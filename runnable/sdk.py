from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, computed_field, field_validator, model_validator
from rich import print
from typing_extensions import Self

from runnable import defaults, entrypoints, graph, utils
from runnable.extensions.nodes import FailNode, MapNode, ParallelNode, StubNode, SuccessNode, TaskNode
from runnable.nodes import TraversalNode
from runnable.tasks import TaskReturns

logger = logging.getLogger(defaults.LOGGER_NAME)

StepType = Union["Stub", "PythonTask", "NotebookTask", "ShellTask", "Success", "Fail", "Parallel", "Map"]
TraversalTypes = Union["Stub", "PythonTask", "NotebookTask", "ShellTask", "Parallel", "Map"]


def pickled(name: str) -> TaskReturns:
    return TaskReturns(name=name, kind="object")


class Catalog(BaseModel):
    """
    Use to instruct a task to sync data from/to the central catalog.
    Please refer to [concepts](concepts/catalog.md) for more information.

    Attributes:
        get (List[str]): List of glob patterns to get from central catalog to the compute data folder.
        put (List[str]): List of glob patterns to put into central catalog from the compute data folder.

    Examples:
        >>> from runnable import Catalog, Task
        >>> catalog = Catalog(compute_data_folder="/path/to/data", get=["*.csv"], put=["*.csv"])

        >>> task = Task(name="task", catalog=catalog, command="echo 'hello'")

    """

    model_config = ConfigDict(extra="forbid")  # Need to be for command, would be validated later
    # Note: compute_data_folder was confusing to explain, might be introduced later.
    # compute_data_folder: str = Field(default="", alias="compute_data_folder")
    get: List[str] = Field(default_factory=list, alias="get")
    put: List[str] = Field(default_factory=list, alias="put")


class BaseTraversal(ABC, BaseModel):
    name: str
    next_node: str = Field(default="", alias="next")
    terminate_with_success: bool = Field(default=False, exclude=True)
    terminate_with_failure: bool = Field(default=False, exclude=True)
    on_failure: str = Field(default="", alias="on_failure")

    model_config = ConfigDict(extra="forbid")

    @computed_field  # type: ignore
    @property
    def internal_name(self) -> str:
        return self.name

    def __rshift__(self, other: StepType) -> StepType:
        if self.next_node:
            raise Exception(f"The node {self} already has a next node: {self.next_node}")
        self.next_node = other.name

        return other

    def __lshift__(self, other: TraversalNode) -> TraversalNode:
        if other.next_node:
            raise Exception(f"The {other} node already has a next node: {other.next_node}")
        other.next_node = self.name

        return other

    def depends_on(self, node: StepType) -> Self:
        assert not isinstance(node, Success)
        assert not isinstance(node, Fail)

        if node.next_node:
            raise Exception(f"The {node} node already has a next node: {node.next_node}")

        node.next_node = self.name
        return self

    @model_validator(mode="after")
    def validate_terminations(self) -> Self:
        if self.terminate_with_failure and self.terminate_with_success:
            raise AssertionError("A node cannot terminate with success and failure")

        if self.terminate_with_failure or self.terminate_with_success:
            if self.next_node and self.next_node not in ["success", "fail"]:
                raise AssertionError("A node being terminated cannot have a user defined next node")

        if self.terminate_with_failure:
            self.next_node = "fail"

        if self.terminate_with_success:
            self.next_node = "success"

        return self

    @abstractmethod
    def create_node(self) -> TraversalNode:
        ...


class BaseTask(BaseTraversal):
    """
    An execution node of the pipeline.
    Please refer to [concepts](concepts/task.md) for more information.

    Attributes:
        name (str): The name of the node.
        command (str): The command to execute.

            - For python functions, [dotted path](concepts/task.md/#python_functions) to the function.
            - For shell commands: command to execute in the shell.
            - For notebooks: path to the notebook.
        command_type (str): The type of command to execute.
            Can be one of "shell", "python", or "notebook".
        catalog (Optional[Catalog]): The catalog to sync data from/to.
            Please see Catalog about the structure of the catalog.
        overrides (Dict[str, Any]): Any overrides to the command.
            Individual tasks can override the global configuration config by referring to the
            specific override.

            For example,
            ### Global configuration
            ```yaml
            executor:
              type: local-container
              config:
                docker_image: "runnable/runnable:latest"
                overrides:
                  custom_docker_image:
                    docker_image: "runnable/runnable:custom"
            ```
            ### Task specific configuration
            ```python
            task = Task(name="task", command="echo 'hello'", command_type="shell",
                    overrides={'local-container': custom_docker_image})
            ```
        notebook_output_path (Optional[str]): The path to save the notebook output.
            Only used when command_type is 'notebook', defaults to command+_out.ipynb
        optional_ploomber_args (Optional[Dict[str, Any]]): Any optional ploomber args.
            Only used when command_type is 'notebook', defaults to {}
        output_cell_tag (Optional[str]): The tag of the output cell.
            Only used when command_type is 'notebook', defaults to "runnable_output"
        terminate_with_failure (bool): Whether to terminate the pipeline with a failure after this node.
        terminate_with_success (bool): Whether to terminate the pipeline with a success after this node.
        on_failure (str): The name of the node to execute if the step fails.

    """

    catalog: Optional[Catalog] = Field(default=None, alias="catalog")
    overrides: Dict[str, Any] = Field(default_factory=dict, alias="overrides")
    returns: List[Union[str, TaskReturns]] = Field(default_factory=list, alias="returns")

    @field_validator("returns", mode="before")
    @classmethod
    def serialize_returns(cls, returns: List[Union[str, TaskReturns]]) -> List[TaskReturns]:
        task_returns = []

        for x in returns:
            if isinstance(x, str):
                task_returns.append(TaskReturns(name=x, kind="json"))
                continue

            # Its already task returns
            task_returns.append(x)

        return task_returns

    def create_node(self) -> TaskNode:
        if not self.next_node:
            if not (self.terminate_with_failure or self.terminate_with_success):
                raise AssertionError("A node not being terminated must have a user defined next node")

        return TaskNode.parse_from_config(self.model_dump(exclude_none=True))


class PythonTask(BaseTask):
    """
    An execution node of the pipeline of python functions.
    Please refer to [concepts](concepts/task.md) for more information.

    Attributes:
        name (str): The name of the node.
        function (callable): The function to execute.
        catalog (Optional[Catalog]): The catalog to sync data from/to.
            Please see Catalog about the structure of the catalog.
        overrides (Dict[str, Any]): Any overrides to the command.
            Individual tasks can override the global configuration config by referring to the
            specific override.

            For example,
            ### Global configuration
            ```yaml
            executor:
              type: local-container
              config:
                docker_image: "runnable/runnable:latest"
                overrides:
                  custom_docker_image:
                    docker_image: "runnable/runnable:custom"
            ```
            ### Task specific configuration
            ```python
            task = PythonTask(name="task", function="function'",
                    overrides={'local-container': custom_docker_image})
            ```

        terminate_with_failure (bool): Whether to terminate the pipeline with a failure after this node.
        terminate_with_success (bool): Whether to terminate the pipeline with a success after this node.
        on_failure (str): The name of the node to execute if the step fails.

    """

    function: Callable = Field(exclude=True)

    @computed_field
    def command_type(self) -> str:
        return "python"

    @computed_field
    def command(self) -> str:
        module = self.function.__module__
        name = self.function.__name__

        return f"{module}.{name}"


class NotebookTask(BaseTask):
    """
    An execution node of the pipeline of type notebook.
    Please refer to [concepts](concepts/task.md) for more information.

    Attributes:
        name (str): The name of the node.
        notebook: The path to the notebook
        catalog (Optional[Catalog]): The catalog to sync data from/to.
            Please see Catalog about the structure of the catalog.
        returns: A list of the names of variables to return from the notebook.
        overrides (Dict[str, Any]): Any overrides to the command.
            Individual tasks can override the global configuration config by referring to the
            specific override.

            For example,
            ### Global configuration
            ```yaml
            executor:
              type: local-container
              config:
                docker_image: "runnable/runnable:latest"
                overrides:
                  custom_docker_image:
                    docker_image: "runnable/runnable:custom"
            ```
            ### Task specific configuration
            ```python
            task = NotebookTask(name="task", notebook="evaluation.ipynb",
                    overrides={'local-container': custom_docker_image})
            ```
        notebook_output_path (Optional[str]): The path to save the notebook output.
            Only used when command_type is 'notebook', defaults to command+_out.ipynb
        optional_ploomber_args (Optional[Dict[str, Any]]): Any optional ploomber args.
            Only used when command_type is 'notebook', defaults to {}

        terminate_with_failure (bool): Whether to terminate the pipeline with a failure after this node.
        terminate_with_success (bool): Whether to terminate the pipeline with a success after this node.
        on_failure (str): The name of the node to execute if the step fails.

    """

    notebook: str = Field(alias="command")

    notebook_output_path: Optional[str] = Field(default=None, alias="notebook_output_path")
    optional_ploomber_args: Optional[Dict[str, Any]] = Field(default=None, alias="optional_ploomber_args")

    @computed_field
    def command_type(self) -> str:
        return "notebook"


class ShellTask(BaseTask):
    """
    An execution node of the pipeline of type shell.
    Please refer to [concepts](concepts/task.md) for more information.

    Attributes:
        name (str): The name of the node.
        command: The shell command to execute.
        catalog (Optional[Catalog]): The catalog to sync data from/to.
            Please see Catalog about the structure of the catalog.
        returns: A list of the names of variables to capture from environment variables of shell.
        overrides (Dict[str, Any]): Any overrides to the command.
            Individual tasks can override the global configuration config by referring to the
            specific override.

            For example,
            ### Global configuration
            ```yaml
            executor:
              type: local-container
              config:
                docker_image: "runnable/runnable:latest"
                overrides:
                  custom_docker_image:
                    docker_image: "runnable/runnable:custom"
            ```
            ### Task specific configuration
            ```python
            task = ShellTask(name="task", command="exit 0",
                    overrides={'local-container': custom_docker_image})
            ```

        terminate_with_failure (bool): Whether to terminate the pipeline with a failure after this node.
        terminate_with_success (bool): Whether to terminate the pipeline with a success after this node.
        on_failure (str): The name of the node to execute if the step fails.

    """

    command: str = Field(alias="command")

    @computed_field
    def command_type(self) -> str:
        return "shell"


class Stub(BaseTraversal):
    """
    A node that does nothing.

    A stub node can tak arbitrary number of arguments.
    Please refer to [concepts](concepts/stub.md) for more information.

    Attributes:
        name (str): The name of the node.
        terminate_with_failure (bool): Whether to terminate the pipeline with a failure after this node.
        terminate_with_success (bool): Whether to terminate the pipeline with a success after this node.

    """

    model_config = ConfigDict(extra="allow")
    catalog: Optional[Catalog] = Field(default=None, alias="catalog")

    def create_node(self) -> StubNode:
        if not self.next_node:
            if not (self.terminate_with_failure or self.terminate_with_success):
                raise AssertionError("A node not being terminated must have a user defined next node")

        return StubNode.parse_from_config(self.model_dump(exclude_none=True))


class Parallel(BaseTraversal):
    """
    A node that executes multiple branches in parallel.
    Please refer to [concepts](concepts/parallel.md) for more information.

    Attributes:
        name (str): The name of the node.
        branches (Dict[str, Pipeline]): A dictionary of branches to execute in parallel.
        terminate_with_failure (bool): Whether to terminate the pipeline with a failure after this node.
        terminate_with_success (bool): Whether to terminate the pipeline with a success after this node.
        on_failure (str): The name of the node to execute if any of the branches fail.
    """

    branches: Dict[str, "Pipeline"]

    @computed_field  # type: ignore
    @property
    def graph_branches(self) -> Dict[str, graph.Graph]:
        return {name: pipeline._dag.model_copy() for name, pipeline in self.branches.items()}

    def create_node(self) -> ParallelNode:
        if not self.next_node:
            if not (self.terminate_with_failure or self.terminate_with_success):
                raise AssertionError("A node not being terminated must have a user defined next node")

        node = ParallelNode(name=self.name, branches=self.graph_branches, internal_name="", next_node=self.next_node)
        return node


class Map(BaseTraversal):
    """
    A node that iterates over a list of items and executes a pipeline for each item.
    Please refer to [concepts](concepts/map.md) for more information.

    Attributes:
        branch: The pipeline to execute for each item.

        iterate_on: The name of the parameter to iterate over.
            The parameter should be defined either by previous steps or statically at the start of execution.

        iterate_as: The name of the iterable to be passed to functions.


        overrides (Dict[str, Any]): Any overrides to the command.

    """

    branch: "Pipeline"
    iterate_on: str
    iterate_as: str
    overrides: Dict[str, Any] = Field(default_factory=dict)

    @computed_field  # type: ignore
    @property
    def graph_branch(self) -> graph.Graph:
        return self.branch._dag.model_copy()

    def create_node(self) -> MapNode:
        if not self.next_node:
            if not (self.terminate_with_failure or self.terminate_with_success):
                raise AssertionError("A node not being terminated must have a user defined next node")

        node = MapNode(
            name=self.name,
            branch=self.graph_branch,
            internal_name="",
            next_node=self.next_node,
            iterate_on=self.iterate_on,
            iterate_as=self.iterate_as,
            overrides=self.overrides,
        )

        return node


class Success(BaseModel):
    """
    A node that represents a successful execution of the pipeline.

    Most often, there is no need to use this node as nodes can be instructed to
    terminate_with_success and pipeline with add_terminal_nodes=True.

    Attributes:
        name (str): The name of the node.
    """

    name: str = "success"

    @computed_field  # type: ignore
    @property
    def internal_name(self) -> str:
        return self.name

    def create_node(self) -> SuccessNode:
        return SuccessNode.parse_from_config(self.model_dump())


class Fail(BaseModel):
    """
    A node that represents a failed execution of the pipeline.

    Most often, there is no need to use this node as nodes can be instructed to
    terminate_with_failure and pipeline with add_terminal_nodes=True.

    Attributes:
        name (str): The name of the node.
    """

    name: str = "fail"

    @computed_field  # type: ignore
    @property
    def internal_name(self) -> str:
        return self.name

    def create_node(self) -> FailNode:
        return FailNode.parse_from_config(self.model_dump())


class Pipeline(BaseModel):
    """
    A Pipeline is a directed acyclic graph of Steps that define a workflow.

    Attributes:
        steps (List[Stub | PythonTask | NotebookTask | ShellTask | Parallel | Map | Success | Fail]):
            A list of Steps that make up the Pipeline.
        start_at (Stub | Task | Parallel | Map): The name of the first Step in the Pipeline.
        name (str, optional): The name of the Pipeline. Defaults to "".
        description (str, optional): A description of the Pipeline. Defaults to "".
        add_terminal_nodes (bool, optional): Whether to add terminal nodes to the Pipeline. Defaults to True.

    The default behavior is to add "success" and "fail" nodes to the Pipeline.
    To add custom success and fail nodes, set add_terminal_nodes=False and create success
    and fail nodes manually.

    """

    steps: List[StepType]
    start_at: TraversalTypes
    name: str = ""
    description: str = ""
    add_terminal_nodes: bool = True  # Adds "success" and "fail" nodes

    internal_branch_name: str = ""

    _dag: graph.Graph = PrivateAttr()
    model_config = ConfigDict(extra="forbid")

    def model_post_init(self, __context: Any) -> None:
        self.steps = [model.model_copy(deep=True) for model in self.steps]

        self._dag = graph.Graph(
            start_at=self.start_at.name,
            description=self.description,
            internal_branch_name=self.internal_branch_name,
        )

        for step in self.steps:
            if step.name == self.start_at.name:
                if isinstance(step, Success) or isinstance(step, Fail):
                    raise Exception("A success or fail node cannot be the start_at of the graph")
                assert step.next_node
            self._dag.add_node(step.create_node())

        if self.add_terminal_nodes:
            self._dag.add_terminal_nodes()

        self._dag.check_graph()

    def return_dag(self) -> graph.Graph:
        return self._dag

    def execute(
        self,
        configuration_file: str = "",
        run_id: str = "",
        tag: str = "",
        parameters_file: str = "",
        use_cached: str = "",
        log_level: str = defaults.LOG_LEVEL,
    ):
        """
        *Execute* the Pipeline.

        Execution of pipeline could either be:

        Traverse and execute all the steps of the pipeline, eg. [local execution](configurations/executors/local.md).

        Or create the ```yaml``` representation of the pipeline for other executors.

        Please refer to [concepts](concepts/executor.md) for more information.

        Args:
            configuration_file (str, optional): The path to the configuration file. Defaults to "".
                The configuration file can be overridden by the environment variable runnable_CONFIGURATION_FILE.

            run_id (str, optional): The ID of the run. Defaults to "".
            tag (str, optional): The tag of the run. Defaults to "".
                Use to group multiple runs.

            parameters_file (str, optional): The path to the parameters file. Defaults to "".
            use_cached (str, optional): Whether to use cached results. Defaults to "".
                Provide the run_id of the older execution to recover.

            log_level (str, optional): The log level. Defaults to defaults.LOG_LEVEL.
        """

        # py_to_yaml is used by non local executors to generate the yaml representation of the pipeline.
        py_to_yaml = os.environ.get("RUNNABLE_PY_TO_YAML", "false")

        if py_to_yaml == "true":
            return

        logger.setLevel(log_level)

        run_id = utils.generate_run_id(run_id=run_id)
        configuration_file = os.environ.get("RUNNABLE_CONFIGURATION_FILE", configuration_file)
        run_context = entrypoints.prepare_configurations(
            configuration_file=configuration_file,
            run_id=run_id,
            tag=tag,
            parameters_file=parameters_file,
        )

        run_context.execution_plan = defaults.EXECUTION_PLAN.CHAINED.value
        utils.set_runnable_environment_variables(run_id=run_id, configuration_file=configuration_file, tag=tag)

        dag_definition = self._dag.model_dump(by_alias=True, exclude_none=True)

        run_context.dag = graph.create_graph(dag_definition)

        print("Working with context:")
        print(run_context)

        if not run_context.executor._local:
            # We are working with non local executor
            import inspect

            caller_stack = inspect.stack()[1]
            module_to_call = f"{caller_stack.filename.replace('/', '.').replace('.py', '')}.{caller_stack.function}"

            run_context.pipeline_file = f"{module_to_call}.py"

        # Prepare for graph execution
        run_context.executor.prepare_for_graph_execution()

        logger.info("Executing the graph")
        run_context.executor.execute_graph(dag=run_context.dag)

        if run_context.executor._local:
            return run_context.run_log_store.get_run_log_by_id(run_id=run_context.run_id)
