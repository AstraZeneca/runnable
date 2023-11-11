from __future__ import annotations

import logging
from logging.config import dictConfig
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, cast

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, computed_field, model_validator
from typing_extensions import Self

from magnus import defaults, entrypoints, graph, utils
from magnus.extensions.nodes import FailNode, ParallelNode, StubNode, SuccessNode, TaskNode
from magnus.nodes import TraversalNode

logger = logging.getLogger(defaults.LOGGER_NAME)

StepType = Union["Stub", "Success", "Fail", "Parallel"]
TraversalTypes = Union["Stub", "Parallel"]


class Catalog(BaseModel):
    compute_folder_name: str = Field(default="data", alias="compute_folder_name")
    get: List[Union[str, Path]] = Field(default=["*"], alias="get")
    put: List[Union[str, Path]] = Field(default=["*"], alias="put")


class BaseTraversal(BaseModel):
    name: str
    next_node: str = Field(default="", alias="next")
    terminate_with_success: bool = Field(default=False, exclude=True)
    terminate_with_fail: bool = Field(default=False, exclude=True)
    on_failure: str = Field(default="", alias="on_failure")

    @computed_field  # type: ignore
    @property
    def internal_name(self) -> str:
        return self.name

    def __rshift__(self, other) -> None:
        if self.next_node:
            raise Exception(f"The node {self} already has a next node: {self.next_node}")
        self.next_node = other.name

    def __lshift__(self, other) -> None:
        if other.next_node:
            raise Exception(f"The {other} node already has a next node: {other.next_node}")
        other.next_node = self.name

    def depends_on(self, node: StepType) -> Self:
        assert not isinstance(node, Success)
        assert not isinstance(node, Fail)

        if node.next_node:
            raise Exception(f"The {node} node already has a next node: {node.next_node}")

        node.next_node = self.name
        return self

    @model_validator(mode="after")
    def validate_terminations(self) -> Self:
        if self.terminate_with_fail and self.terminate_with_success:
            raise AssertionError("A node cannot terminate with success and failure")

        if self.terminate_with_fail or self.terminate_with_success:
            if self.next_node and self.next_node not in ["success", "fail"]:
                raise AssertionError("A node being terminated cannot have a user defined next node")

        if self.terminate_with_fail:
            self.next_node = "fail"

        if self.terminate_with_success:
            self.next_node = "success"

        return self

    def create_node(self) -> TraversalNode:
        if not self.next_node:
            if not (self.terminate_with_fail or self.terminate_with_success):
                raise AssertionError("A node not being terminated must have a user defined next node")

        return StubNode.parse_from_config(self.model_dump(exclude_none=True))


class Task(BaseTraversal):
    model_config = ConfigDict(extra="allow")  # Need to be for command, would be validated later
    catalog: Optional[Catalog] = Field(default=None, alias="catalog")

    def create_node(self) -> TaskNode:
        return cast(TaskNode, super().create_node())


class Stub(BaseTraversal):
    model_config = ConfigDict(extra="allow")
    catalog: Optional[Catalog] = Field(default=None, alias="catalog")

    def create_node(self) -> StubNode:
        return cast(StubNode, super().create_node())


class Parallel(BaseTraversal):
    branches: Dict[str, "Pipeline"]

    @computed_field  # type: ignore
    @property
    def graph_branches(self) -> Dict[str, graph.Graph]:
        return {name: pipeline._dag.model_copy() for name, pipeline in self.branches.items()}

    def create_node(self) -> ParallelNode:
        if not self.next_node:
            if not (self.terminate_with_fail or self.terminate_with_success):
                raise AssertionError("A node not being terminated must have a user defined next node")

        node = ParallelNode(name=self.name, branches=self.graph_branches, internal_name="", next_node=self.next_node)

        node.add_parent(self.name)
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
    max_time: int = defaults.MAX_TIME
    add_terminal_nodes: bool = True  # Adds "success" and "fail" nodes

    internal_branch_name: str = ""

    _dag: graph.Graph = PrivateAttr()
    model_config = ConfigDict(extra="forbid")

    def model_post_init(self, __context: Any) -> None:
        self.steps = [model.model_copy(deep=True) for model in self.steps]

        self._dag = graph.Graph(
            start_at=self.start_at.name,
            description=self.description,
            max_time=self.max_time,
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

    def execute(
        self,
        configuration_file: str = "",
        run_id: str = "",
        tag: str = "",
        parameters_file: str = "",
        log_level: str = defaults.LOG_LEVEL,
    ):
        """Execute the pipeline.

        This method should be beefed up as the use cases grow.
        """
        dictConfig(defaults.LOGGING_CONFIG)
        logger = logging.getLogger(defaults.LOGGER_NAME)
        logger.setLevel(log_level)

        run_id = utils.generate_run_id(run_id=run_id)
        run_context = entrypoints.prepare_configurations(
            configuration_file=configuration_file,
            run_id=run_id,
            tag=tag,
            parameters_file=parameters_file,
        )

        run_context.execution_plan = defaults.EXECUTION_PLAN.CHAINED.value
        utils.set_magnus_environment_variables(run_id=run_id, configuration_file=configuration_file, tag=tag)

        run_context.dag = self._dag
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
