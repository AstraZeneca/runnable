import logging

from runnable import defaults
from runnable.defaults import TypeMapVariable
from runnable.extensions.executor import GenericExecutor
from runnable.extensions.nodes import TaskNode
from runnable.nodes import BaseNode

logger = logging.getLogger(defaults.LOGGER_NAME)


class LocalExecutor(GenericExecutor):
    """
    In the mode of local execution, we run everything on the local computer.

    This has some serious implications on the amount of time it would take to complete the run.
    Also ensure that the local compute is good enough for the compute to happen of all the steps.

    Example config:
    execution:
      type: local
      config:
        enable_parallel: True or False to enable parallel.

    """

    service_name: str = "local"
    _local: bool = True

    def trigger_job(self, node: BaseNode, map_variable: TypeMapVariable = None, **kwargs):
        """
        In this mode of execution, we prepare for the node execution and execute the node

        Args:
            node (BaseNode): [description]
            map_variable (str, optional): [description]. Defaults to ''.
        """
        self.prepare_for_node_execution()
        self.execute_node(node=node, map_variable=map_variable, **kwargs)

    def execute_node(self, node: BaseNode, map_variable: TypeMapVariable = None, **kwargs):
        """
        For local execution, we just execute the node.

        Args:
            node (BaseNode): _description_
            map_variable (dict[str, str], optional): _description_. Defaults to None.
        """
        self._execute_node(node=node, map_variable=map_variable, **kwargs)

    def execute_job(self, node: TaskNode):
        """
        Set up the step log and call the execute node

        Args:
            node (BaseNode): _description_
        """

        step_log = self._context.run_log_store.create_step_log(node.name, node._get_step_log_name(map_variable=None))

        self.add_code_identities(node=node, step_log=step_log)

        step_log.step_type = node.node_type
        step_log.status = defaults.PROCESSING
        self._context.run_log_store.add_step_log(step_log, self._context.run_id)
        self.execute_node(node=node)

        # Update the run log status
        step_log = self._context.run_log_store.get_step_log(node._get_step_log_name(), self._context.run_id)
        self._context.run_log_store.update_run_log_status(run_id=self._context.run_id, status=step_log.status)
