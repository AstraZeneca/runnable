import copy
import logging
from typing import Any, Dict, Type, cast

from pydantic import ConfigDict, Field

from extensions.nodes.nodes import TaskNode
from extensions.pipeline_executor import GenericPipelineExecutor
from runnable import context, defaults
from runnable.defaults import TypeMapVariable
from runnable.nodes import BaseNode
from runnable.tasks import BaseTaskType

logger = logging.getLogger(defaults.LOGGER_NAME)


def create_executable(
    params: Dict[str, Any], model: Type[BaseTaskType], node_name: str
) -> BaseTaskType:
    class EasyModel(model):  # type: ignore
        model_config = ConfigDict(extra="ignore")

    swallow_all = EasyModel(node_name=node_name, **params)
    return swallow_all


class MockedExecutor(GenericPipelineExecutor):
    service_name: str = "mocked"
    _is_local: bool = True

    model_config = ConfigDict(extra="ignore")

    patches: Dict[str, Any] = Field(default_factory=dict)

    @property
    def _context(self):
        return context.run_context

    def execute_from_graph(
        self, node: BaseNode, map_variable: TypeMapVariable = None, **kwargs
    ):
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
        step_log = self._context.run_log_store.create_step_log(
            node.name, node._get_step_log_name(map_variable)
        )

        self.add_code_identities(node=node, step_log=step_log)

        step_log.step_type = node.node_type
        step_log.status = defaults.PROCESSING

        self._context.run_log_store.add_step_log(step_log, self._context.run_id)

        logger.info(f"Executing node: {node.get_summary()}")

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

        if node.name not in self.patches:
            # node is not patched, so mock it
            self._execute_node(node, map_variable=map_variable, mock=True, **kwargs)
        else:
            # node is patched
            # command as the patch value
            node_to_send: TaskNode = cast(TaskNode, node).model_copy(deep=True)
            executable_type = node_to_send.executable.__class__
            executable = create_executable(
                self.patches[node.name],
                executable_type,
                node_name=node.name,
            )
            node_to_send.executable = executable
            self._execute_node(
                node_to_send, map_variable=map_variable, mock=False, **kwargs
            )

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

    def execute_node(
        self, node: BaseNode, map_variable: TypeMapVariable = None, **kwargs
    ):
        """
        The entry point for all executors apart from local.
        We have already prepared for node execution.

        Args:
            node (BaseNode): The node to execute
            map_variable (dict, optional): If the node is part of a map, send in the map dictionary. Defaults to None.

        Raises:
            NotImplementedError: _description_
        """
        ...
