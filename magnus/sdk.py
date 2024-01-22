from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, computed_field, field_validator, model_validator
from rich import print
from ruamel.yaml import YAML
from typing_extensions import Self

from magnus import defaults, entrypoints, graph, utils
from magnus.extensions.nodes import FailNode, MapNode, ParallelNode, StubNode, SuccessNode, TaskNode
from magnus.nodes import TraversalNode

logger = logging.getLogger(defaults.LOGGER_NAME)

StepType = Union["Stub", "Task", "Success", "Fail", "Parallel", "Map"]
TraversalTypes = Union["Stub", "Task", "Parallel", "Map"]


ALLOWED_COMMAND_TYPES = ["shell", "python", "notebook"]


class Catalog(BaseModel):
    """
    Use to instruct a task to sync data from/to the central catalog.

    Args:
        get (List[str]): List of glob patterns to get from central catalog to the compute data folder.
        put (List[str]): List of glob patterns to put into central catalog from the compute data folder.

    Examples:
        >>> from magnus import Catalog, Task
        >>> catalog = Catalog(compute_data_folder="/path/to/data", get=["*.csv"], put=["*.csv"])

        >>> task = Task(name="task", catalog=catalog)

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


class Task(BaseTraversal):
    model_config = ConfigDict(use_enum_values=True)  # Need to be for command, would be validated later
    command: str = Field(alias="command")
    command_type: str = Field(default="python")
    catalog: Optional[Catalog] = Field(default=None, alias="catalog")

    notebook_output_path: Optional[str] = Field(default=None, alias="notebook_output_path")
    optional_ploomber_args: Optional[Dict[str, Any]] = Field(default=None, alias="optional_ploomber_args")
    output_cell_tag: Optional[str] = Field(default=None, alias="output_cell_tag")

    @field_validator("command_type", mode="after")
    @classmethod
    def validate_command_type(cls, value: str) -> str:
        if value not in ALLOWED_COMMAND_TYPES:
            raise ValueError(f"Invalid command_type: {value}")
        return value

    @model_validator(mode="after")
    def check_notebook_args(self) -> "Task":
        if self.command_type != "notebook":
            assert (
                self.notebook_output_path is None
            ), "Only command_types of 'notebook' can be used with notebook_output_path"

            assert (
                self.optional_ploomber_args is None
            ), "Only command_types of 'notebook' can be used with optional_ploomber_args"

            assert self.output_cell_tag is None, "Only command_types of 'notebook' can be used with output_cell_tag"
        return self

    def create_node(self) -> TaskNode:
        if not self.next_node:
            if not (self.terminate_with_failure or self.terminate_with_success):
                raise AssertionError("A node not being terminated must have a user defined next node")
        return TaskNode.parse_from_config(self.model_dump(exclude_none=True))


class Stub(BaseTraversal):
    model_config = ConfigDict(extra="allow")
    catalog: Optional[Catalog] = Field(default=None, alias="catalog")

    def create_node(self) -> StubNode:
        if not self.next_node:
            if not (self.terminate_with_failure or self.terminate_with_success):
                raise AssertionError("A node not being terminated must have a user defined next node")

        return StubNode.parse_from_config(self.model_dump(exclude_none=True))


class Parallel(BaseTraversal):
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
    branch: "Pipeline"
    iterate_on: str
    iterate_as: str

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
        )

        return node


class Success(BaseModel):
    name: str = "success"

    @computed_field  # type: ignore
    @property
    def internal_name(self) -> str:
        return self.name

    def create_node(self) -> SuccessNode:
        return SuccessNode.parse_from_config(self.model_dump())


class Fail(BaseModel):
    name: str = "fail"

    @computed_field  # type: ignore
    @property
    def internal_name(self) -> str:
        return self.name

    def create_node(self) -> FailNode:
        return FailNode.parse_from_config(self.model_dump())


class Pipeline(BaseModel):
    """An exposed magnus pipeline to be used in SDK."""

    steps: List[StepType]  # TODO: Add map and dag nodes
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

    # TODO: There is a need for variables to be part of the process.
    def execute(
        self,
        configuration_file: str = "",
        run_id: str = "",
        tag: str = "",
        parameters_file: str = "",
        use_cached: str = "",
        log_level: str = defaults.LOG_LEVEL,
        output_pipeline_definition: str = "magnus-pipeline.yaml",
    ):
        """Execute the pipeline.

        This method should be beefed up as the use cases grow.
        """
        from magnus.extensions.executor.local.implementation import LocalExecutor

        logger.setLevel(log_level)

        run_id = utils.generate_run_id(run_id=run_id)
        configuration_file = os.environ.get("MAGNUS_CONFIGURATION_FILE", configuration_file)
        run_context = entrypoints.prepare_configurations(
            configuration_file=configuration_file,
            run_id=run_id,
            tag=tag,
            parameters_file=parameters_file,
            use_cached=use_cached,
        )

        run_context.execution_plan = defaults.EXECUTION_PLAN.CHAINED.value
        utils.set_magnus_environment_variables(run_id=run_id, configuration_file=configuration_file, tag=tag)

        dag_definition = self._dag.model_dump(by_alias=True, exclude_none=True)

        run_context.dag = graph.create_graph(dag_definition)

        print("Working with context:")
        print(run_context)

        if not isinstance(run_context.executor, LocalExecutor):
            logger.debug(run_context.dag.model_dump(by_alias=True))
            yaml = YAML()

            with open(output_pipeline_definition, "w", encoding="utf-8") as f:
                yaml.dump(
                    {"dag": run_context.dag.model_dump(by_alias=True, exclude_none=True)},
                    f,
                )

            return

        # Prepare for graph execution
        run_context.executor.prepare_for_graph_execution()

        logger.info("Executing the graph")
        run_context.executor.execute_graph(dag=run_context.dag)

        return run_context.run_log_store.get_run_log_by_id(run_id=run_context.run_id)


# class step(object):

#     def __init__(
#             self, name: Union[str, FunctionType],
#             catalog_config: dict = None, magnus_config: str = None,
#             parameters_file: str = None):
#         """
#         This decorator could be used to make the function within the scope of magnus.

#         Since we are not orchestrating, it is expected that resource management happens outside this scope.

#         Args:
#             name (str, callable): The name of the step. The step log would have the same name
#             catalog_config (dict): The configuration of the catalog per step.
#             magnus_config (str): The name of the file having the magnus config, defaults to None.
#         """
#         if isinstance(name, FunctionType):
#             name = name()

#         self.name = name
#         self.catalog_config = catalog_config
#         self.active = True  # Check if we are executing the function via pipeline

#         if pipeline.global_executor \
#                 and pipeline.global_executor.execution_plan == defaults.EXECUTION_PLAN.CHAINED.value:
#             self.active = False
#             return

#         self.executor = pipeline.prepare_configurations(
#             configuration_file=magnus_config, parameters_file=parameters_file)

#         self.executor.execution_plan = defaults.EXECUTION_PLAN.UNCHAINED.value
#         run_id = self.executor.step_decorator_run_id
#         if not run_id:
#             msg = (
#                 f'Step decorator expects run id from environment.'
#             )
#             raise Exception(msg)

#         self.executor.run_id = run_id
#         utils.set_magnus_environment_variables(run_id=run_id, configuration_file=magnus_config, tag=get_tag())

#         try:
#             # Try to get it if previous steps have created it
#             # TODO: Can call the set_up_runlog now.
#             run_log = self.executor.run_log_store.get_run_log_by_id(self.executor.run_id)
#             if run_log.status in [defaults.FAIL, defaults.SUCCESS]:  # TODO: Remove this in preference to defaults
#                 """
#                 This check is mostly useless as we do not know when the graph ends as they are created dynamically.
#                 This only prevents from using a run_id which has reached a final state.
#                 #TODO: There is a need to create a status called step_success
#                 """
#                 msg = (
#                     f'The run_log for run_id: {run_id} already exists and is in {run_log.status} state.'
#                     ' Make sure that this was not run before.'
#                 )
#                 raise Exception(msg)
#         except exceptions.RunLogNotFoundError:
#             # Create one if they are not created
#             self.executor._set_up_run_log()

#     def __call__(self, func):
#         """
#         The function is converted into a node and called via the magnus framework.
#         """
#         @functools.wraps(func)
#         def wrapped_f(*args, **kwargs):
#             if not self.active:
#                 # If we are not running via decorator, execute the function
#                 return func(*args, **kwargs)

#             step_config = {
#                 'command': func,
#                 'command_type': 'python-function',
#                 'type': 'task',
#                 'next': 'not defined',
#                 'catalog': self.catalog_config
#             }
#             node = graph.create_node(name=self.name, step_config=step_config)
#             self.executor.execute_from_graph(node=node)
#             run_log = self.executor.run_log_store.get_run_log_by_id(run_id=self.executor.run_id, full=False)
#             # TODO: If the previous step succeeded, make the status of the run log step_success
#             print(json.dumps(run_log.dict(), indent=4))
#         return wrapped_f
