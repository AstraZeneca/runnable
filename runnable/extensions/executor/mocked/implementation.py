import copy
import logging
from typing import Any, Dict, Type, cast

from pydantic import ConfigDict, Field

from runnable import context, defaults
from runnable.defaults import TypeMapVariable
from runnable.extensions.executor import GenericExecutor
from runnable.extensions.nodes import TaskNode
from runnable.integration import BaseIntegration
from runnable.nodes import BaseNode
from runnable.tasks import BaseTaskType

logger = logging.getLogger(defaults.LOGGER_NAME)


def create_executable(params: Dict[str, Any], model: Type[BaseTaskType], node_name: str) -> BaseTaskType:
    class EasyModel(model):  # type: ignore
        model_config = ConfigDict(extra="ignore")

    swallow_all = EasyModel(**params, node_name=node_name)
    return swallow_all


class MockedExecutor(GenericExecutor):
    service_name: str = "mocked"

    enable_parallel: bool = defaults.ENABLE_PARALLEL

    patches: Dict[str, Any] = Field(default_factory=dict)

    @property
    def _context(self):
        return context.run_context

    def _set_up_for_re_run(self, parameters: Dict[str, Any]) -> None:
        raise Exception("MockedExecutor does not support re-run")

    def execute_from_graph(self, node: BaseNode, map_variable: TypeMapVariable = None, **kwargs):
        """
        This is the entry point to from the graph execution.

        While the self.execute_graph is responsible for traversing the graph, this function is responsible for
        actual execution of the node.

        If the node type is:
            * task : We can delegate to _execute_node after checking the eligibility for re-run in cases of a re-run
            * success: We can delegate to _execute_node
            * fail: We can delegate to _execute_node

        For nodes that are internally graphs:
            * parallel: Delegate the responsibility of execution to the node.execute_as_graph()
            * dag: Delegate the responsibility of execution to the node.execute_as_graph()
            * map: Delegate the responsibility of execution to the node.execute_as_graph()

        Transpilers will NEVER use this method and will NEVER call ths method.
        This method should only be used by interactive executors.

        Args:
            node (Node): The node to execute
            map_variable (dict, optional): If the node if of a map state, this corresponds to the value of iterable.
                    Defaults to None.
        """
        step_log = self._context.run_log_store.create_step_log(node.name, node._get_step_log_name(map_variable))

        self.add_code_identities(node=node, step_log=step_log)

        step_log.step_type = node.node_type
        step_log.status = defaults.PROCESSING

        # Add the step log to the database as per the situation.
        # If its a terminal node, complete it now
        if node.node_type in ["success", "fail"]:
            self._context.run_log_store.add_step_log(step_log, self._context.run_id)
            self._execute_node(node, map_variable=map_variable, **kwargs)
            return

        # We call an internal function to iterate the sub graphs and execute them
        if node.is_composite:
            self._context.run_log_store.add_step_log(step_log, self._context.run_id)
            node.execute_as_graph(map_variable=map_variable, **kwargs)
            return

        node_to_send: TaskNode = cast(TaskNode, node).model_copy(deep=True)
        if node.name not in self.patches:
            # node is not patched, so mock it
            step_log.mock = True
        else:
            # node is mocked, change the executable to python with the
            # command as the patch value
            executable_type = node_to_send.executable.__class__
            executable = create_executable(
                self.patches[node.name],
                executable_type,
                node_name=node.name,
            )
            node_to_send.executable = executable
            pass

        # Executor specific way to trigger a job
        self._context.run_log_store.add_step_log(step_log, self._context.run_id)
        self.trigger_job(node=node_to_send, map_variable=map_variable, **kwargs)

    def trigger_job(self, node: BaseNode, map_variable: TypeMapVariable = None, **kwargs):
        """
        Call this method only if we are responsible for traversing the graph via
        execute_from_graph().

        We are not prepared to execute node as of now.

        Args:
            node (BaseNode): The node to execute
            map_variable (str, optional): If the node if of a map state, this corresponds to the value of iterable.
                    Defaults to ''.

        NOTE: We do not raise an exception as this method is not required by many extensions
        """
        self.prepare_for_node_execution()
        self.execute_node(node=node, map_variable=map_variable, **kwargs)

    def _is_step_eligible_for_rerun(self, node: BaseNode, map_variable: TypeMapVariable = None):
        """
        In case of a re-run, this method checks to see if the previous run step status to determine if a re-run is
        necessary.
            * True: If its not a re-run.
            * True: If its a re-run and we failed in the last run or the corresponding logs do not exist.
            * False: If its a re-run and we succeeded in the last run.

        Most cases, this logic need not be touched

        Args:
            node (Node): The node to check against re-run
            map_variable (dict, optional): If the node if of a map state, this corresponds to the value of iterable..
                        Defaults to None.

        Returns:
            bool: Eligibility for re-run. True means re-run, False means skip to the next step.
        """
        return True

    def _resolve_executor_config(self, node: BaseNode):
        """
        The overrides section can contain specific over-rides to an global executor config.
        To avoid too much clutter in the dag definition, we allow the configuration file to have overrides block.
        The nodes can over-ride the global config by referring to key in the overrides.

        This function also applies variables to the effective node config.

        For example:
        # configuration.yaml
        execution:
          type: cloud-implementation
          config:
            k1: v1
            k3: v3
            overrides:
             custom_config:
                k1: v11
                k2: v2 # Could be a mapping internally.

        # in pipeline definition.yaml
        dag:
          steps:
            step1:
              overrides:
                cloud-implementation: custom_config

        This method should resolve the node_config to {'k1': 'v11', 'k2': 'v2', 'k3': 'v3'}

        Args:
            node (BaseNode): The current node being processed.

        """
        effective_node_config = copy.deepcopy(self.model_dump())

        return effective_node_config

    def execute_job(self, node: TaskNode):
        pass

    def execute_node(self, node: BaseNode, map_variable: TypeMapVariable = None, **kwargs):
        """
        For local execution, we just execute the node.

        Args:
            node (BaseNode): _description_
            map_variable (dict[str, str], optional): _description_. Defaults to None.
        """
        self._execute_node(node=node, map_variable=map_variable, **kwargs)


class LocalContainerComputeFileSystemRunLogstore(BaseIntegration):
    """
    Integration between local container and file system run log store
    """

    executor_type = "local-container"
    service_type = "run_log_store"  # One of secret, catalog, datastore
    service_provider = "file-system"  # The actual implementation of the service

    def validate(self, **kwargs):
        if self.executor._is_parallel_execution():  # pragma: no branch
            msg = "Mocked executor does not support parallel execution. "
            logger.warning(msg)


class LocalContainerComputeChunkedFSRunLogstore(BaseIntegration):
    """
    Integration between local container and file system run log store
    """

    executor_type = "local-container"
    service_type = "run_log_store"  # One of secret, catalog, datastore
    service_provider = "chunked-fs"  # The actual implementation of the service

    def validate(self, **kwargs):
        if self.executor._is_parallel_execution():  # pragma: no branch
            msg = "Mocked executor does not support parallel execution. "
            logger.warning(msg)
