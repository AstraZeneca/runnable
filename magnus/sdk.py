from __future__ import annotations

import logging
from logging.config import dictConfig
from typing import Any, Callable, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, FieldValidationInfo, PrivateAttr, computed_field, field_validator

from magnus import defaults, entrypoints, graph, utils
from magnus.extensions.nodes import FailNode, ParallelNode, StubNode, SuccessNode, TaskNode

logger = logging.getLogger(defaults.LOGGER_NAME)


class Success(BaseModel):
    name: str = "success"

    _node: SuccessNode = PrivateAttr()

    @computed_field  # type: ignore
    @property
    def internal_name(self) -> str:
        return self.name

    def model_post_init(self, __context: Any) -> None:
        self._node = SuccessNode.parse_from_config(self.model_dump())


class Fail(BaseModel):
    name: str = "fail"

    _node: FailNode = PrivateAttr()

    @computed_field  # type: ignore
    @property
    def internal_name(self) -> str:
        return self.name

    def model_post_init(self, __context: Any) -> None:
        self._node = FailNode.parse_from_config(self.model_dump())


class Task(BaseModel):
    name: str
    command: Union[str, Callable]
    command_type: str = Field(default="python")
    next_node: str = Field(alias="next")
    terminate_with_success: bool = Field(default=False, exclude=True)

    model_config = ConfigDict(extra="allow")  # Need to be for command, would be validated later

    _node: TaskNode = PrivateAttr()

    @computed_field  # type: ignore
    @property
    def internal_name(self) -> str:
        return self.name

    def model_post_init(self, __context: Any) -> None:
        self._node = TaskNode.parse_from_config(config=self.model_dump())


class Stub(BaseModel):
    name: str
    terminate_with_success: bool = Field(default=False, exclude=True)
    next_node: Optional[str] = Field(default="", alias="next", validate_default=True)

    model_config = ConfigDict(extra="allow")

    _node: StubNode = PrivateAttr()

    @computed_field  # type: ignore
    @property
    def internal_name(self) -> str:
        return self.name

    @field_validator("next_node", mode="before")
    @classmethod
    def next_node_validator(cls, next_node: Optional[str], info: FieldValidationInfo) -> str:
        if next_node:
            return next_node

        if info.data["terminate_with_success"]:
            return "success"

        raise ValueError("Next node is required or can be terminated with success")

    def model_post_init(self, __context: Any) -> None:
        self._node = StubNode.parse_from_config(self.model_dump())


class Parallel(BaseModel):
    name: str
    branches: Dict[str, "Pipeline"]
    next_node: str = Field(alias="next")
    terminate_with_success: bool = Field(default=False, exclude=True)

    @computed_field  # type: ignore
    @property
    def internal_name(self) -> str:
        return self.name

    @field_validator("next_node", mode="before")
    @classmethod
    def next_node_validator(cls, next_node: Optional[str], info: FieldValidationInfo) -> str:
        if next_node:
            return next_node

        if info.data["terminate_with_success"]:
            return "success"

        raise ValueError("Next node is required or can be terminated with success")

    def model_post_init(self, __context: Any) -> None:
        node_branches = {name: pipeline._dag for name, pipeline in self.branches.items()}
        node_config = {**self.model_dump(), **{"branches": node_branches}}

        self._node = ParallelNode.parse_from_config(config=node_config, internal_name=self.name)


class Pipeline(BaseModel):
    # TODO: Allow for nodes other than Task, AsIs
    """An exposed magnus pipeline to be used in SDK."""

    steps: List[Union[Task, Stub]]
    start_at: str  # TODO would be nicer to refer to nodes here
    name: str = ""
    description: str = ""
    max_time: int = defaults.MAX_TIME
    add_terminal_nodes: bool = True

    internal_branch_name: str = ""

    _dag: Optional[graph.Graph] = PrivateAttr()
    model_config = ConfigDict(extra="forbid")

    def model_post_init(self, __context: Any) -> None:
        self._dag = graph.Graph(
            start_at=self.start_at,
            description=self.description,
            max_time=self.max_time,
            internal_branch_name=self.internal_branch_name,
        )

        for step in self.steps:
            self._dag.add_node(step._node)

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
        run_context.executor.execute_graph(dag=run_context.dag)  # type: ignore

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
