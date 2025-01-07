import logging
from functools import cached_property
from typing import Any, Dict, Optional

from extensions.pipeline_executor import GenericPipelineExecutor
from runnable import context, defaults, exceptions
from runnable.datastore import RunLog
from runnable.defaults import TypeMapVariable
from runnable.nodes import BaseNode

logger = logging.getLogger(defaults.LOGGER_NAME)


class RetryExecutor(GenericPipelineExecutor):
    """
    The skeleton of an executor class.
    Any implementation of an executor should inherit this class and over-ride accordingly.

    This is a loaded base class which has a lot of methods already implemented for "typical" executions.
    Look at the function docs to understand how to use them appropriately.

    For any implementation:
    1). Who/when should the run log be set up?
    2). Who/When should the step log be set up?

    """

    service_name: str = "retry"
    service_type: str = "executor"
    run_id: str

    _is_local: bool = True
    _original_run_log: Optional[RunLog] = None
    _restart_initiated: bool = False

    @property
    def _context(self):
        return context.run_context

    @cached_property
    def original_run_log(self):
        return self._context.run_log_store.get_run_log_by_id(
            run_id=self.run_id,
            full=True,
        )

    def _set_up_for_re_run(self, params: Dict[str, Any]) -> None:
        # Sync the previous run log catalog to this one.
        self._context.catalog_handler.sync_between_runs(
            previous_run_id=self.run_id, run_id=self._context.run_id
        )

        params.update(self.original_run_log.parameters)

    def _set_up_run_log(self, exists_ok=False):
        """
        Create a run log and put that in the run log store

        If exists_ok, we allow the run log to be already present in the run log store.
        """
        super()._set_up_run_log(exists_ok=exists_ok)

        # Should the parameters be copied from previous execution
        # self._set_up_for_re_run(params=params)

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

        # Add the step log to the database as per the situation.
        # If its a terminal node, complete it now
        if node.node_type in ["success", "fail"]:
            self._context.run_log_store.add_step_log(step_log, self._context.run_id)
            self._execute_node(node, map_variable=map_variable, **kwargs)
            return

        # In retry step
        if not self._is_step_eligible_for_rerun(node, map_variable=map_variable):
            # If the node name does not match, we move on to the next node.
            # If previous run was successful, move on to the next step
            step_log.mock = True
            step_log.status = defaults.SUCCESS
            self._context.run_log_store.add_step_log(step_log, self._context.run_id)
            return

        # We call an internal function to iterate the sub graphs and execute them
        if node.is_composite:
            self._context.run_log_store.add_step_log(step_log, self._context.run_id)
            node.execute_as_graph(map_variable=map_variable, **kwargs)
            return

        # Executor specific way to trigger a job
        self._context.run_log_store.add_step_log(step_log, self._context.run_id)
        self.execute_node(node=node, map_variable=map_variable, **kwargs)

    def _is_step_eligible_for_rerun(
        self, node: BaseNode, map_variable: TypeMapVariable = None
    ):
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

        node_step_log_name = node._get_step_log_name(map_variable=map_variable)
        logger.info(
            f"Scanning previous run logs for node logs of: {node_step_log_name}"
        )

        if self._restart_initiated:
            return True

        try:
            previous_attempt_log, _ = (
                self.original_run_log.search_step_by_internal_name(node_step_log_name)
            )
        except exceptions.StepLogNotFoundError:
            logger.warning(f"Did not find the node {node.name} in previous run log")
            self._restart_initiated = True
            return True  # We should re-run the node.

        logger.info(f"The original step status: {previous_attempt_log.status}")

        if previous_attempt_log.status == defaults.SUCCESS:
            return False  # We need not run the node

        logger.info(
            f"The new execution should start executing graph from this node {node.name}"
        )
        self._restart_initiated = True
        return True

    def execute_node(
        self, node: BaseNode, map_variable: TypeMapVariable = None, **kwargs
    ):
        self._execute_node(node, map_variable=map_variable, **kwargs)
