from __future__ import annotations

import logging
import os
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    computed_field,
    field_validator,
    model_validator,
)
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Column
from typing_extensions import Self

from extensions.nodes.nodes import (
    FailNode,
    MapNode,
    ParallelNode,
    StubNode,
    SuccessNode,
    TaskNode,
)
from runnable import console, defaults, entrypoints, exceptions, graph, utils
from runnable.executor import BaseJobExecutor, BasePipelineExecutor
from runnable.nodes import TraversalNode
from runnable.tasks import BaseTaskType as RunnableTask
from runnable.tasks import TaskReturns

# TODO: This might have to be an extension

logger = logging.getLogger(defaults.LOGGER_NAME)

StepType = Union["Stub", "PythonTask", "NotebookTask", "ShellTask", "Parallel", "Map"]


def pickled(name: str) -> TaskReturns:
    return TaskReturns(name=name, kind="object")


def metric(name: str) -> TaskReturns:
    return TaskReturns(name=name, kind="metric")


class Catalog(BaseModel):
    """
    Use to instruct a task to sync data from/to the central catalog.
    Please refer to [concepts](concepts/catalog.md) for more information.

    Attributes:
        get (List[str]): List of glob patterns to get from central catalog to the compute data folder.
        put (List[str]): List of glob patterns to put into central catalog from the compute data folder.

    Examples:
        >>> from runnable import Catalog
        >>> catalog = Catalog(compute_data_folder="/path/to/data", get=["*.csv"], put=["*.csv"])

    """

    model_config = ConfigDict(
        extra="forbid"
    )  # Need to be for command, would be validated later
    # Note: compute_data_folder was confusing to explain, might be introduced later.
    # compute_data_folder: str = Field(default="", alias="compute_data_folder")
    get: List[str] = Field(default_factory=list, alias="get")
    put: List[str] = Field(default_factory=list, alias="put")


class BaseTraversal(ABC, BaseModel):
    name: str
    next_node: str = Field(default="", serialization_alias="next_node")
    terminate_with_success: bool = Field(default=False, exclude=True)
    terminate_with_failure: bool = Field(default=False, exclude=True)
    on_failure: str = Field(default="", alias="on_failure")

    model_config = ConfigDict(extra="forbid")

    @computed_field  # type: ignore
    @property
    def internal_name(self) -> str:
        return self.name

    def __hash__(self):
        """
        Needed to Uniqueize DataCatalog objects.
        """
        return hash(self.name)

    def __rshift__(self, other: StepType) -> StepType:
        if self.next_node:
            raise Exception(
                f"The node {self} already has a next node: {self.next_node}"
            )
        self.next_node = other.name

        return other

    def __lshift__(self, other: TraversalNode) -> TraversalNode:
        if other.next_node:
            raise Exception(
                f"The {other} node already has a next node: {other.next_node}"
            )
        other.next_node = self.name

        return other

    def depends_on(self, node: StepType) -> Self:
        assert not isinstance(node, Success)
        assert not isinstance(node, Fail)

        if node.next_node:
            raise Exception(
                f"The {node} node already has a next node: {node.next_node}"
            )

        node.next_node = self.name
        return self

    @model_validator(mode="after")
    def validate_terminations(self) -> Self:
        if self.terminate_with_failure and self.terminate_with_success:
            raise AssertionError("A node cannot terminate with success and failure")

        if self.terminate_with_failure or self.terminate_with_success:
            if self.next_node and self.next_node not in ["success", "fail"]:
                raise AssertionError(
                    "A node being terminated cannot have a user defined next node"
                )

        if self.terminate_with_failure:
            self.next_node = "fail"

        if self.terminate_with_success:
            self.next_node = "success"

        return self

    @abstractmethod
    def create_node(self) -> TraversalNode: ...


class BaseTask(BaseTraversal):
    """
    Base task type which has catalog, overrides, returns and secrets.
    """

    catalog: Optional[Catalog] = Field(default=None, alias="catalog")
    overrides: Dict[str, Any] = Field(default_factory=dict, alias="overrides")
    returns: List[Union[str, TaskReturns]] = Field(
        default_factory=list, alias="returns"
    )
    secrets: List[str] = Field(default_factory=list)

    @field_validator("returns", mode="before")
    @classmethod
    def serialize_returns(
        cls, returns: List[Union[str, TaskReturns]]
    ) -> List[TaskReturns]:
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
                raise AssertionError(
                    "A node not being terminated must have a user defined next node"
                )

        return TaskNode.parse_from_config(
            self.model_dump(exclude_none=True, by_alias=True)
        )

    def create_job(self) -> RunnableTask:
        raise NotImplementedError(
            "This method should be implemented in the child class"
        )


class PythonTask(BaseTask):
    """
    An execution node of the pipeline of python functions.
    Please refer to [concepts](concepts/task.md/#python_functions) for more information.

    Attributes:
        name (str): The name of the node.
        function (callable): The function to execute.

        terminate_with_success (bool): Whether to terminate the pipeline with a success after this node.
                Defaults to False.
        terminate_with_failure (bool): Whether to terminate the pipeline with a failure after this node.
                Defaults to False.

        on_failure (str): The name of the node to execute if the step fails.

        returns List[Union[str, TaskReturns]] : A list of the names of variables to return from the task.
            The names should match the order of the variables returned by the function.

            ```TaskReturns```: can be JSON friendly variables, objects or metrics.

            By default, all variables are assumed to be JSON friendly and will be serialized to JSON.
            Pydantic models are readily supported and will be serialized to JSON.

            To return a python object, please use ```pickled(<name>)```.
            It is advised to use ```pickled(<name>)``` for big JSON friendly variables.

            For example,
            ```python
            from runnable import pickled

            def f():
                ...
                x = 1
                return x, df # A simple JSON friendly variable and a python object.

            task = PythonTask(name="task", function=f, returns=["x", pickled(df)]))
            ```

            To mark any JSON friendly variable as a ```metric```, please use ```metric(x)```.
            Metric variables should be JSON friendly and can be treated just like any other parameter.

        catalog Optional[Catalog]: The files sync data from/to, refer to Catalog.

        secrets List[str]: List of secrets to pass to the task. They are exposed as environment variables
            and removed after execution.

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

    def create_job(self) -> RunnableTask:
        self.terminate_with_success = True
        node = self.create_node()
        return node.executable


class NotebookTask(BaseTask):
    """
    An execution node of the pipeline of notebook.
    Please refer to [concepts](concepts/task.md/#notebooks) for more information.

    We internally use [Ploomber engine](https://github.com/ploomber/ploomber-engine) to execute the notebook.

    Attributes:
        name (str): The name of the node.
        notebook (str): The path to the notebook relative the project root.
        optional_ploomber_args (Dict[str, Any]): Any optional ploomber args, please refer to
            [Ploomber engine](https://github.com/ploomber/ploomber-engine) for more information.

        terminate_with_success (bool): Whether to terminate the pipeline with a success after this node.
                Defaults to False.
        terminate_with_failure (bool): Whether to terminate the pipeline with a failure after this node.
                Defaults to False.

        on_failure (str): The name of the node to execute if the step fails.

        returns List[Union[str, TaskReturns]] : A list of the names of variables to return from the task.
            The names should match the order of the variables returned by the function.

            ```TaskReturns```: can be JSON friendly variables, objects or metrics.

            By default, all variables are assumed to be JSON friendly and will be serialized to JSON.
            Pydantic models are readily supported and will be serialized to JSON.

            To return a python object, please use ```pickled(<name>)```.
            It is advised to use ```pickled(<name>)``` for big JSON friendly variables.

            For example,
            ```python
            from runnable import pickled

            # assume, example.ipynb is the notebook with df and x as variables in some cells.

            task = Notebook(name="task", notebook="example.ipynb", returns=["x", pickled(df)]))
            ```

            To mark any JSON friendly variable as a ```metric```, please use ```metric(x)```.
            Metric variables should be JSON friendly and can be treated just like any other parameter.

        catalog Optional[Catalog]: The files sync data from/to, refer to Catalog.

        secrets List[str]: List of secrets to pass to the task. They are exposed as environment variables
        and removed after execution.

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
            task = NotebookTask(name="task", notebook="example.ipynb",
                    overrides={'local-container': custom_docker_image})
            ```
    """

    notebook: str = Field(serialization_alias="command")
    optional_ploomber_args: Optional[Dict[str, Any]] = Field(
        default=None, alias="optional_ploomber_args"
    )

    @computed_field
    def command_type(self) -> str:
        return "notebook"

    def create_job(self) -> RunnableTask:
        self.terminate_with_success = True
        node = self.create_node()
        return node.executable


class ShellTask(BaseTask):
    """
    An execution node of the pipeline of shell script.
    Please refer to [concepts](concepts/task.md/#shell) for more information.


    Attributes:
        name (str): The name of the node.
        command (str): The path to the notebook relative the project root.
        terminate_with_success (bool): Whether to terminate the pipeline with a success after this node.
                Defaults to False.
        terminate_with_failure (bool): Whether to terminate the pipeline with a failure after this node.
                Defaults to False.

        on_failure (str): The name of the node to execute if the step fails.

        returns List[str] : A list of the names of environment variables to collect from the task.

            The names should match the order of the variables returned by the function.
            Shell based tasks can only return JSON friendly variables.

            To mark any JSON friendly variable as a ```metric```, please use ```metric(x)```.
            Metric variables should be JSON friendly and can be treated just like any other parameter.

        catalog Optional[Catalog]: The files sync data from/to, refer to Catalog.

        secrets List[str]: List of secrets to pass to the task. They are exposed as environment variables
        and removed after execution.

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
            task = ShellTask(name="task", command="export x=1",
                    overrides={'local-container': custom_docker_image})
            ```

    """

    command: str = Field(alias="command")

    @computed_field
    def command_type(self) -> str:
        return "shell"


class Stub(BaseTraversal):
    """
    A node that passes through the pipeline with no action. Just like ```pass``` in Python.
    Please refer to [concepts](concepts/task.md/#stub) for more information.

    A stub node can tak arbitrary number of arguments.

    Attributes:
        name (str): The name of the node.
        command (str): The path to the notebook relative the project root.
        terminate_with_success (bool): Whether to terminate the pipeline with a success after this node.
                Defaults to False.
        terminate_with_failure (bool): Whether to terminate the pipeline with a failure after this node.
                Defaults to False.

        on_failure (str): The name of the node to execute if the step fails.
    """

    model_config = ConfigDict(extra="ignore")
    catalog: Optional[Catalog] = Field(default=None, alias="catalog")

    def create_node(self) -> StubNode:
        if not self.next_node:
            if not (self.terminate_with_failure or self.terminate_with_success):
                raise AssertionError(
                    "A node not being terminated must have a user defined next node"
                )

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
        return {
            name: pipeline._dag.model_copy() for name, pipeline in self.branches.items()
        }

    def create_node(self) -> ParallelNode:
        if not self.next_node:
            if not (self.terminate_with_failure or self.terminate_with_success):
                raise AssertionError(
                    "A node not being terminated must have a user defined next node"
                )

        node = ParallelNode(
            name=self.name,
            branches=self.graph_branches,
            internal_name="",
            next_node=self.next_node,
        )
        return node


class Map(BaseTraversal):
    """
    A node that iterates over a list of items and executes a pipeline for each item.
    Please refer to [concepts](concepts/map.md) for more information.

    Attributes:
        branch (Pipeline): The pipeline to execute for each item.

        iterate_on (str): The name of the parameter to iterate over.
            The parameter should be defined either by previous steps or statically at the start of execution.

        iterate_as (str): The name of the iterable to be passed to functions.
        reducer (Callable): The function to reduce the results of the branches.


        overrides (Dict[str, Any]): Any overrides to the command.

    """

    branch: "Pipeline"
    iterate_on: str
    iterate_as: str
    reducer: Optional[str] = Field(default=None, alias="reducer")
    overrides: Dict[str, Any] = Field(default_factory=dict)

    @computed_field  # type: ignore
    @property
    def graph_branch(self) -> graph.Graph:
        return self.branch._dag.model_copy()

    def create_node(self) -> MapNode:
        if not self.next_node:
            if not (self.terminate_with_failure or self.terminate_with_success):
                raise AssertionError(
                    "A node not being terminated must have a user defined next node"
                )

        node = MapNode(
            name=self.name,
            branch=self.graph_branch,
            internal_name="",
            next_node=self.next_node,
            iterate_on=self.iterate_on,
            iterate_as=self.iterate_as,
            overrides=self.overrides,
            reducer=self.reducer,
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
    A Pipeline is a sequence of Steps.

    Attributes:
        steps (List[Stub | PythonTask | NotebookTask | ShellTask | Parallel | Map]]):
            A list of Steps that make up the Pipeline.

            The order of steps is important as it determines the order of execution.
            Any on failure behavior should the first step in ```on_failure``` pipelines.



        on_failure (List[List[Pipeline], optional): A list of Pipelines to execute in case of failure.

            For example, for the below pipeline:
                step1 >> step2
                and step1 to reach step3 in case of failure.

                failure_pipeline = Pipeline(steps=[step1, step3])

                pipeline = Pipeline(steps=[step1, step2, on_failure=[failure_pipeline])

        name (str, optional): The name of the Pipeline. Defaults to "".
        description (str, optional): A description of the Pipeline. Defaults to "".

    The pipeline implicitly add success and fail nodes.

    """

    steps: List[Union[StepType, List["Pipeline"]]]
    name: str = ""
    description: str = ""

    internal_branch_name: str = ""

    @property
    def add_terminal_nodes(self) -> bool:
        return True

    _dag: graph.Graph = PrivateAttr()
    model_config = ConfigDict(extra="forbid")

    def _validate_path(self, path: List[StepType], failure_path: bool = False) -> None:
        # TODO: Drastically simplify this
        # Check if one and only one step terminates with success
        # Check no more than one step terminates with failure

        reached_success = False
        reached_failure = False

        for step in path:
            if step.terminate_with_success:
                if reached_success:
                    raise Exception(
                        "A pipeline cannot have more than one step that terminates with success"
                    )
                reached_success = True
                continue
            if step.terminate_with_failure:
                if reached_failure:
                    raise Exception(
                        "A pipeline cannot have more than one step that terminates with failure"
                    )
                reached_failure = True

        if not reached_success and not reached_failure:
            raise Exception(
                "A pipeline must have at least one step that terminates with success"
            )

    def _construct_path(self, path: List[StepType]) -> None:
        prev_step = path[0]

        for step in path:
            if step == prev_step:
                continue

            if prev_step.terminate_with_success or prev_step.terminate_with_failure:
                raise Exception(
                    f"A step that terminates with success/failure cannot have a next step: {prev_step}"
                )

            if prev_step.next_node and prev_step.next_node not in ["success", "fail"]:
                raise Exception(f"Step already has a next node: {prev_step} ")

            prev_step.next_node = step.name
            prev_step = step

    def model_post_init(self, __context: Any) -> None:
        """
        The sequence of steps can either be:
            [step1, step2,..., stepN, [step11, step12,..., step1N], [step21, step22,...,]]
            indicates:
                - step1 > step2 > ... > stepN
                - We expect terminate with success or fail to be explicitly stated on a step.
                    - If it is stated, the step cannot have a next step defined apart from "success" and "fail".

                The inner list of steps is only to accommodate on-failure behaviors.
                    - For sake of simplicity, lets assume that it has the same behavior as the happy pipeline.
                    - A task which was already seen should not be part of this.
                    - There should be at least one step which terminates with success

                Any definition of pipeline should have one node that terminates with success.
        """
        # TODO: Bug with repeat names
        # TODO: https://github.com/AstraZeneca/runnable/issues/156

        success_path: List[StepType] = []
        on_failure_paths: List[List[StepType]] = []

        for step in self.steps:
            if isinstance(
                step, (Stub, PythonTask, NotebookTask, ShellTask, Parallel, Map)
            ):
                success_path.append(step)
                continue
            # on_failure_paths.append(step)

        if not success_path:
            raise Exception("There should be some success path")

        # Check all paths are valid and construct the path
        paths = [success_path] + on_failure_paths
        failure_path = False
        for path in paths:
            self._validate_path(path, failure_path)
            self._construct_path(path)

            failure_path = True

        all_steps: List[StepType] = []

        for path in paths:
            for step in path:
                all_steps.append(step)

        seen = set()
        unique = [x for x in all_steps if not (x in seen or seen.add(x))]  # type: ignore

        self._dag = graph.Graph(
            start_at=all_steps[0].name,
            description=self.description,
            internal_branch_name=self.internal_branch_name,
        )

        for step in unique:
            self._dag.add_node(step.create_node())

        if self.add_terminal_nodes:
            self._dag.add_terminal_nodes()

        self._dag.check_graph()

    def return_dag(self) -> graph.Graph:
        dag_definition = self._dag.model_dump(by_alias=True, exclude_none=True)
        return graph.create_graph(dag_definition)

    def _is_called_for_definition(self) -> bool:
        """
        If the run context is set, we are coming in only to get the pipeline definition.
        """
        from runnable.context import run_context

        if run_context is None:
            return False
        return True

    def execute(
        self,
        configuration_file: str = "",
        run_id: str = "",
        tag: str = "",
        parameters_file: str = "",
        log_level: str = defaults.LOG_LEVEL,
    ):
        """
        Overloaded method:
        - Could be called by the user when executing the pipeline via SDK
        - Could be called by the system itself when getting the pipeline definition
        """
        if self._is_called_for_definition():
            # Immediately return as this call is only for getting the pipeline definition
            return {}

        logger.setLevel(log_level)

        run_id = utils.generate_run_id(run_id=run_id)
        configuration_file = os.environ.get(
            "RUNNABLE_CONFIGURATION_FILE", configuration_file
        )
        run_context = entrypoints.prepare_configurations(
            configuration_file=configuration_file,
            run_id=run_id,
            tag=tag,
            parameters_file=parameters_file,
        )

        assert isinstance(run_context.executor, BasePipelineExecutor)

        utils.set_runnable_environment_variables(
            run_id=run_id, configuration_file=configuration_file, tag=tag
        )

        dag_definition = self._dag.model_dump(by_alias=True, exclude_none=True)
        run_context.from_sdk = True
        run_context.dag = graph.create_graph(dag_definition)

        console.print("Working with context:")
        console.print(run_context)
        console.rule(style="[dark orange]")

        if not run_context.executor._is_local:
            # We are not working with executor that does not work in local environment
            import inspect

            caller_stack = inspect.stack()[1]
            relative_to_root = str(Path(caller_stack.filename).relative_to(Path.cwd()))

            module_name = re.sub(r"\b.py\b", "", relative_to_root.replace("/", "."))
            module_to_call = f"{module_name}.{caller_stack.function}"

            run_context.pipeline_file = f"{module_to_call}.py"
            run_context.from_sdk = True

        # Prepare for graph execution
        run_context.executor._set_up_run_log(exists_ok=False)

        with Progress(
            SpinnerColumn(spinner_name="runner"),
            TextColumn(
                "[progress.description]{task.description}", table_column=Column(ratio=2)
            ),
            BarColumn(table_column=Column(ratio=1), style="dark_orange"),
            TimeElapsedColumn(table_column=Column(ratio=1)),
            console=console,
            expand=True,
        ) as progress:
            pipeline_execution_task = progress.add_task(
                "[dark_orange] Starting execution .. ", total=1
            )
            try:
                run_context.progress = progress

                run_context.executor.execute_graph(dag=run_context.dag)

                if not run_context.executor._is_local:
                    # non local executors just traverse the graph and do nothing
                    return {}

                run_log = run_context.run_log_store.get_run_log_by_id(
                    run_id=run_context.run_id, full=False
                )

                if run_log.status == defaults.SUCCESS:
                    progress.update(
                        pipeline_execution_task,
                        description="[green] Success",
                        completed=True,
                    )
                else:
                    progress.update(
                        pipeline_execution_task,
                        description="[red] Failed",
                        completed=True,
                    )
                    raise exceptions.ExecutionFailedError(run_context.run_id)
            except Exception as e:  # noqa: E722
                console.print(e, style=defaults.error_style)
                progress.update(
                    pipeline_execution_task,
                    description="[red] Errored execution",
                    completed=True,
                )
                raise

        if run_context.executor._is_local:
            return run_context.run_log_store.get_run_log_by_id(
                run_id=run_context.run_id
            )


class Job(BaseModel):
    name: str
    task: BaseTask

    def return_task(self) -> RunnableTask:
        return self.task.create_job()

    def return_catalog_settings(self) -> Optional[List[str]]:
        if self.task.catalog is None:
            return []
        return self.task.catalog.put

    def _is_called_for_definition(self) -> bool:
        """
        If the run context is set, we are coming in only to get the pipeline definition.
        """
        from runnable.context import run_context

        if run_context is None:
            return False
        return True

    def execute(
        self,
        configuration_file: str = "",
        job_id: str = "",
        tag: str = "",
        parameters_file: str = "",
        log_level: str = defaults.LOG_LEVEL,
    ):
        if self._is_called_for_definition():
            # Immediately return as this call is only for getting the job definition
            return {}
        logger.setLevel(log_level)

        run_id = utils.generate_run_id(run_id=job_id)
        configuration_file = os.environ.get(
            "RUNNABLE_CONFIGURATION_FILE", configuration_file
        )
        run_context = entrypoints.prepare_configurations(
            configuration_file=configuration_file,
            run_id=run_id,
            tag=tag,
            parameters_file=parameters_file,
            is_job=True,
        )

        assert isinstance(run_context.executor, BaseJobExecutor)
        run_context.from_sdk = True

        utils.set_runnable_environment_variables(
            run_id=run_id, configuration_file=configuration_file, tag=tag
        )

        console.print("Working with context:")
        console.print(run_context)
        console.rule(style="[dark orange]")

        if not run_context.executor._is_local:
            # We are not working with executor that does not work in local environment
            import inspect

            caller_stack = inspect.stack()[1]
            relative_to_root = str(Path(caller_stack.filename).relative_to(Path.cwd()))

            module_name = re.sub(r"\b.py\b", "", relative_to_root.replace("/", "."))
            module_to_call = f"{module_name}.{caller_stack.function}"

            run_context.job_definition_file = f"{module_to_call}.py"

        job = self.task.create_job()
        catalog_settings = self.return_catalog_settings()

        run_context.executor.submit_job(job, catalog_settings=catalog_settings)

        logger.info(
            "Executing the job from the user. We are still in the caller's compute environment"
        )

        if run_context.executor._is_local:
            return run_context.run_log_store.get_run_log_by_id(
                run_id=run_context.run_id
            )
