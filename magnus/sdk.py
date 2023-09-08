from __future__ import annotations

import logging
from logging.config import fileConfig
from typing import List, Optional, Union

from pkg_resources import resource_filename
from pydantic import BaseModel, Extra, Field

from magnus import defaults, entrypoints, graph, utils

logger = logging.getLogger(defaults.LOGGER_NAME)


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


class BaseStep(BaseModel):
    name: str
    next_node: str = ""
    on_failure: Optional[str]
    _node: Optional[BaseStep]
    _is_frozen: bool = False  # Once the graph is constructed, it is frozen for any changes.
    # Could be interesting to see this:
    # https://stackoverflow.com/questions/67078207/is-it-possible-to-dynamically-change-the-mutability-of-a-pydantic-class

    class Config:
        extra = Extra.allow
        underscore_attrs_are_private = True

    def _construct_node(self):
        """Construct a node of the graph."""
        # The node will temporarily have invalid branch names
        step_config = self.dict(by_alias=True)
        step_config.pop("name")
        self._node = graph.create_node(name=self.name, step_config=step_config, internal_branch_name="")

    def set_on_failure(self, name: str):
        if self._is_frozen:
            raise Exception(f"Cannot modify a node: {self.name} after the graph has been constructed.")

        self.on_failure = name

    def set_next_node(self, next_node: str):
        if self._is_frozen:
            raise Exception(f"Cannot modify a node: {self.name} after the graph has been constructed.")

        self.next_node = next_node

    def go_to_success(self):
        if self._is_frozen:
            raise Exception(f"Cannot modify a node: {self.name} after the graph has been constructed.")

        self.next_node = "success"

    def go_to_failure(self):
        if self._is_frozen:
            raise Exception(f"Cannot modify a node: {self.name} after the graph has been constructed.")

        self.next_node = "fail"

    def _fix_internal_name(self):
        """Should be done after the parallel's are implemented."""
        pass


# BaseStep.update_forward_refs()


class Task(BaseStep):
    """A exposed magnus task to be used in SDK."""

    ref_type: str = Field("task", alias="type")


class AsIs(BaseStep):
    """A exposed magnus as-is to be used in SDK."""

    ref_type: str = Field("as-is", alias="type")


class Pipeline(BaseModel):
    # TODO: Allow for nodes other than Task, AsIs
    """An exposed magnus pipeline to be used in SDK."""

    steps: List[Union[Task, AsIs]]
    additional_steps: List[Union[Task, AsIs]] = []
    name: str = ""
    description: str = ""
    max_time: int = defaults.MAX_TIME
    internal_branch_name: str = ""
    _dag: Optional[graph.Graph] = None
    _start_at: Optional[Union[Task, AsIs]] = None

    class Config:
        extra = Extra.forbid
        underscore_attrs_are_private = True

    def __init__(self, **data):
        super().__init__(**data)
        self._construct()

    def _construct(self):
        """Construct a pipeline from a list of tasks."""

        prev_step = None
        messages: List[str] = []
        for step in self.steps:
            if not self._start_at:
                # The first step is always the start_at
                self._start_at = step

            # Freeze the step from any alterations
            step._is_frozen = True

            # Construct the previous named step of the graph
            if prev_step:
                # Link to the next node only if it is asked to be done
                if not prev_step.next_node:
                    prev_step.next_node = step.name

                prev_step._construct_node()

                messages.extend(prev_step._node.validate())

            prev_step = step

        # construct the last named step of the graph
        if not step.next_node:
            step.next_node = "success"
        step._construct_node()

        # Add the additional steps of the graph
        for step in self.additional_steps:
            if not step.next_node:
                messages.append(f"The step {step.name} has no next node. Additional nodes should have a next node.")
            step._construct_node()
            messages.extend(step._node.validate())  # type: ignore

        if messages:
            raise Exception(", ".join(messages))

        graph_config = self.dict()
        graph_config["start_at"] = self._start_at.name  # type: ignore

        graph_config.pop("steps")
        graph_config.pop("additional_steps")

        self._dag = graph.Graph(**graph_config)
        self._dag.nodes = [step._node for step in self.steps]  # type: ignore

        self._dag.add_terminal_nodes()

        self._dag.validate()

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
        fileConfig(resource_filename(__name__, "log_config.ini"))
        logger = logging.getLogger(defaults.LOGGER_NAME)
        logger.setLevel(log_level)

        run_id = utils.generate_run_id(run_id=run_id)
        mode_executor = entrypoints.prepare_configurations(
            configuration_file=configuration_file,
            run_id=run_id,
            tag=tag,
            parameters_file=parameters_file,
        )

        mode_executor.execution_plan = defaults.EXECUTION_PLAN.CHAINED.value
        utils.set_magnus_environment_variables(run_id=run_id, configuration_file=configuration_file, tag=tag)

        mode_executor.dag = self._dag
        # Prepare for graph execution
        mode_executor.prepare_for_graph_execution()

        logger.info("Executing the graph")
        mode_executor.execute_graph(dag=mode_executor.dag)

        mode_executor.send_return_code()
