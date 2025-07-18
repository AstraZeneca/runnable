from __future__ import annotations

import inspect
import logging
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
from typing_extensions import Self

from extensions.nodes.conditional import ConditionalNode
from extensions.nodes.fail import FailNode
from extensions.nodes.map import MapNode
from extensions.nodes.parallel import ParallelNode
from extensions.nodes.stub import StubNode
from extensions.nodes.success import SuccessNode
from extensions.nodes.task import TaskNode
from runnable import defaults, graph
from runnable.executor import BaseJobExecutor
from runnable.nodes import TraversalNode
from runnable.tasks import BaseTaskType as RunnableTask
from runnable.tasks import TaskReturns

logger = logging.getLogger(defaults.LOGGER_NAME)

StepType = Union[
    "Stub",
    "PythonTask",
    "NotebookTask",
    "ShellTask",
    "Parallel",
    "Map",
    "Conditional",
]


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
    on_failure: Optional[Pipeline] = Field(default=None)

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

        if self.on_failure:
            self.on_failure = self.on_failure.steps[0].name  # type: ignore

        return TaskNode.parse_from_config(
            self.model_dump(exclude_none=True, by_alias=True)
        )

    def create_job(self) -> RunnableTask:
        raise NotImplementedError(
            "This method should be implemented in the child class"
        )

    def as_pipeline(self) -> "Pipeline":
        return Pipeline(steps=[self], name=self.internal_name)  # type: ignore


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

    def create_job(self) -> RunnableTask:
        self.terminate_with_success = True
        node = self.create_node()
        return node.executable


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

    def as_pipeline(self) -> "Pipeline":
        return Pipeline(steps=[self])


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


class Conditional(BaseTraversal):
    branches: Dict[str, "Pipeline"]
    parameter: str  # the name of the parameter should be isalnum

    @field_validator("parameter")
    @classmethod
    def validate_parameter(cls, parameter: str) -> str:
        if not parameter.isalnum():
            raise AssertionError(
                "The parameter name should be alphanumeric and not empty"
            )
        return parameter

    @field_validator("branches")
    @classmethod
    def validate_branches(
        cls, branches: Dict[str, "Pipeline"]
    ) -> Dict[str, "Pipeline"]:
        for branch_name in branches.keys():
            if not branch_name.isalnum():
                raise ValueError(f"Branch '{branch_name}' must be alphanumeric.")
        return branches

    @computed_field  # type: ignore
    @property
    def graph_branches(self) -> Dict[str, graph.Graph]:
        return {
            name: pipeline._dag.model_copy() for name, pipeline in self.branches.items()
        }

    def create_node(self) -> ConditionalNode:
        if not self.next_node:
            if not (self.terminate_with_failure or self.terminate_with_success):
                raise AssertionError(
                    "A node not being terminated must have a user defined next node"
                )

        node = ConditionalNode(
            name=self.name,
            branches=self.graph_branches,
            internal_name="",
            next_node=self.next_node,
            parameter=self.parameter,
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

    steps: List[StepType]
    name: str = ""
    description: str = ""

    internal_branch_name: str = ""

    @property
    def add_terminal_nodes(self) -> bool:
        return True

    _dag: graph.Graph = PrivateAttr()
    model_config = ConfigDict(extra="forbid")

    def model_post_init(self, __context: Any) -> None:
        """
        The sequence of steps can either be:
            [step1, step2,..., stepN]
            indicates:
                - step1 > step2 > ... > stepN
                - We expect terminate with success or fail to be explicitly stated on a step.
                    - If it is stated, the step cannot have a next step defined apart from "success" and "fail".
                Any definition of pipeline should have one node that terminates with success.
        """
        # The last step of the pipeline is defaulted to be a success step
        # unless it is explicitly stated to terminate with failure.
        terminal_step: StepType = self.steps[-1]
        if not terminal_step.terminate_with_failure:
            terminal_step.terminate_with_success = True
            terminal_step.next_node = "success"

        # assert that there is only one termination node with success or failure
        # Assert that there are no duplicate step names
        observed: Dict[str, str] = {}
        count_termination: int = 0

        for step in self.steps:
            if isinstance(
                step, (Stub, PythonTask, NotebookTask, ShellTask, Parallel, Map)
            ):
                if step.terminate_with_success or step.terminate_with_failure:
                    count_termination += 1
            if step.name in observed:
                raise Exception(
                    f"Step names should be unique. Found duplicate: {step.name}"
                )
            observed[step.name] = step.name

        if count_termination > 1:
            raise AssertionError(
                "A pipeline can only have one termination node with success or failure"
            )

        # link the steps by assigning the next_node name to be that name of the node
        # immediately after it.
        for i in range(len(self.steps) - 1):
            self.steps[i] >> self.steps[i + 1]

        # Add any on_failure pipelines to the steps
        gathered_on_failure: List[StepType] = []
        for step in self.steps:
            if step.on_failure:
                gathered_on_failure.extend(step.on_failure.steps)

        self._dag = graph.Graph(
            start_at=self.steps[0].name,
            description=self.description,
            internal_branch_name=self.internal_branch_name,
        )

        self.steps.extend(gathered_on_failure)

        for step in self.steps:
            self._dag.add_node(step.create_node())

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

    def get_caller(self) -> str:
        caller_stack = inspect.stack()[2]
        relative_to_root = str(Path(caller_stack.filename).relative_to(Path.cwd()))

        module_name = re.sub(r"\b.py\b", "", relative_to_root.replace("/", "."))
        module_to_call = f"{module_name}.{caller_stack.function}"

        return module_to_call

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

        from runnable import context

        logger.setLevel(log_level)

        service_configurations = context.ServiceConfigurations(
            configuration_file=configuration_file,
            execution_context=context.ExecutionContext.PIPELINE,
        )

        configurations = {
            "pipeline_definition_file": self.get_caller(),
            "parameters_file": parameters_file,
            "tag": tag,
            "run_id": run_id,
            "execution_mode": context.ExecutionMode.PYTHON,
            "configuration_file": configuration_file,
            **service_configurations.services,
        }

        run_context = context.PipelineContext.model_validate(configurations)
        context.run_context = run_context

        assert isinstance(run_context, context.PipelineContext)

        run_context.execute()


class BaseJob(BaseModel):
    catalog: Optional[Catalog] = Field(default=None, alias="catalog")
    returns: List[Union[str, TaskReturns]] = Field(
        default_factory=list, alias="returns"
    )
    secrets: List[str] = Field(default_factory=list)

    @field_validator("catalog", mode="after")
    @classmethod
    def validate_catalog(cls, catalog: Optional[Catalog]) -> Optional[Catalog]:
        if catalog is None:
            return None

        if catalog.get:
            raise Exception("Catalog get is not supported for jobs")

        return catalog

    def get_task(self) -> RunnableTask:
        raise NotImplementedError

    def get_caller(self) -> str:
        caller_stack = inspect.stack()[2]
        relative_to_root = str(Path(caller_stack.filename).relative_to(Path.cwd()))

        module_name = re.sub(r"\b.py\b", "", relative_to_root.replace("/", "."))
        module_to_call = f"{module_name}.{caller_stack.function}"

        return module_to_call

    def return_catalog_settings(self) -> Optional[List[str]]:
        if self.catalog is None:
            return []
        return self.catalog.put

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
        from runnable import context

        logger.setLevel(log_level)

        service_configurations = context.ServiceConfigurations(
            configuration_file=configuration_file,
            execution_context=context.ExecutionContext.JOB,
        )

        configurations = {
            "job_definition_file": self.get_caller(),
            "parameters_file": parameters_file,
            "tag": tag,
            "run_id": job_id,
            "execution_mode": context.ExecutionMode.PYTHON,
            "configuration_file": configuration_file,
            "job": self.get_task(),
            "catalog_settings": self.return_catalog_settings(),
            **service_configurations.services,
        }

        run_context = context.JobContext.model_validate(configurations)

        assert isinstance(run_context.job_executor, BaseJobExecutor)

        run_context.execute()


class PythonJob(BaseJob):
    function: Callable = Field()

    @property
    @computed_field
    def command(self) -> str:
        module = self.function.__module__
        name = self.function.__name__

        return f"{module}.{name}"

    def get_task(self) -> RunnableTask:
        # Piggy bank on existing tasks as a hack
        task = PythonTask(
            name="dummy",
            terminate_with_success=True,
            **self.model_dump(exclude_defaults=True, exclude_none=True),
        )
        return task.create_node().executable


class NotebookJob(BaseJob):
    notebook: str = Field(serialization_alias="command")
    optional_ploomber_args: Optional[Dict[str, Any]] = Field(
        default=None, alias="optional_ploomber_args"
    )

    def get_task(self) -> RunnableTask:
        # Piggy bank on existing tasks as a hack
        task = NotebookTask(
            name="dummy",
            terminate_with_success=True,
            **self.model_dump(exclude_defaults=True, exclude_none=True),
        )
        return task.create_node().executable


class ShellJob(BaseJob):
    command: str = Field(alias="command")

    def get_task(self) -> RunnableTask:
        # Piggy bank on existing tasks as a hack
        task = ShellTask(
            name="dummy",
            terminate_with_success=True,
            **self.model_dump(exclude_defaults=True, exclude_none=True),
        )
        return task.create_node().executable
