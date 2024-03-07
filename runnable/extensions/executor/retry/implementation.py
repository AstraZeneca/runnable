import copy
import json
import logging
from functools import cached_property
from typing import Any, Dict, List, Optional

from rich import print

from runnable import context, defaults, exceptions, parameters, utils
from runnable.datastore import DataCatalog, RunLog
from runnable.defaults import TypeMapVariable
from runnable.experiment_tracker import get_tracked_data
from runnable.extensions.executor import GenericExecutor
from runnable.graph import Graph
from runnable.nodes import BaseNode

logger = logging.getLogger(defaults.LOGGER_NAME)


class RetryExecutor(GenericExecutor):
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

    _local: bool = True
    _original_run_log: Optional[RunLog] = None

    @property
    def _context(self):
        return context.run_context

    @cached_property
    def original_run_log(self):
        self.original_run_log = self._context.run_log_store.get_run_log_by_id(
            run_id=self.run_id,
            full=True,
        )

    def _set_up_for_re_run(self, params: Dict[str, Any]) -> None:
        # Sync the previous run log catalog to this one.
        self._context.catalog_handler.sync_between_runs(previous_run_id=self.run_id, run_id=self._context.run_id)

        params.update(self.original_run_log.parameters)

    def _set_up_run_log(self, exists_ok=False):
        """
        Create a run log and put that in the run log store

        If exists_ok, we allow the run log to be already present in the run log store.
        """
        super()._set_up_run_log(exists_ok=exists_ok)

        params = self._get_parameters()

        self._set_up_for_re_run(params=params)

    def _execute_node(self, node: BaseNode, map_variable: TypeMapVariable = None, **kwargs):
        """
        This is the entry point when we do the actual execution of the function.
        DO NOT Over-ride this function.

        While in interactive execution, we just compute, in 3rd party interactive execution, we need to reach
        this function.

        In most cases,
            * We get the corresponding step_log of the node and the parameters.
            * We sync the catalog to GET any data sets that are in the catalog
            * We call the execute method of the node for the actual compute and retry it as many times as asked.
            * If the node succeeds, we get any of the user defined metrics provided by the user.
            * We sync the catalog to PUT any data sets that are in the catalog.

        Args:
            node (Node): The node to execute
            map_variable (dict, optional): If the node is of a map state, map_variable is the value of the iterable.
                        Defaults to None.
        """
        step_log = self._context.run_log_store.get_step_log(node._get_step_log_name(map_variable), self._context.run_id)
        """
        By now, all the parameters are part of the run log as a dictionary.
        We set them as environment variables, serialized as json strings.
        """
        params = self._context.run_log_store.get_parameters(run_id=self._context.run_id)
        params_copy = copy.deepcopy(params)
        # This is only for the API to work.
        parameters.set_user_defined_params_as_environment_variables(params)

        attempt = self.step_attempt_number
        logger.info(f"Trying to execute node: {node.internal_name}, attempt : {attempt}")

        attempt_log = self._context.run_log_store.create_attempt_log()
        self._context_step_log = step_log
        self._context_node = node

        data_catalogs_get: Optional[List[DataCatalog]] = self._sync_catalog(step_log, stage="get")
        try:
            attempt_log = node.execute(
                executor=self,
                mock=step_log.mock,
                map_variable=map_variable,
                params=params,
                **kwargs,
            )
        except Exception as e:
            # Any exception here is a runnable exception as node suppresses exceptions.
            msg = "This is clearly runnable fault, please report a bug and the logs"
            logger.exception(msg)
            raise Exception(msg) from e
        finally:
            attempt_log.attempt_number = attempt
            step_log.attempts.append(attempt_log)

            tracked_data = get_tracked_data()

            self._context.experiment_tracker.publish_data(tracked_data)
            parameters_out = attempt_log.output_parameters

            if attempt_log.status == defaults.FAIL:
                logger.exception(f"Node: {node} failed")
                step_log.status = defaults.FAIL
            else:
                # Mock is always set to False, bad design??
                # TODO: Stub nodes should not sync back data
                # TODO: Errors in catalog syncing should point to Fail step
                # TODO: Even for a failed execution, the catalog can happen
                step_log.status = defaults.SUCCESS
                self._sync_catalog(step_log, stage="put", synced_catalogs=data_catalogs_get)
                step_log.user_defined_metrics = tracked_data

                diff_parameters = utils.diff_dict(params_copy, parameters_out)
                self._context.run_log_store.set_parameters(self._context.run_id, diff_parameters)

            # Remove the step context
            parameters.get_user_set_parameters(remove=True)
            self._context_step_log = None
            self._context_node = None  # type: ignore
            self._context_metrics = {}

            self._context.run_log_store.add_step_log(step_log, self._context.run_id)

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

        # In single step
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

    def execute_graph(self, dag: Graph, map_variable: TypeMapVariable = None, **kwargs):
        """
        The parallelization is controlled by the nodes and not by this function.

        Transpilers should over ride this method to do the translation of dag to the platform specific way.
        Interactive methods should use this to traverse and execute the dag.
            - Use execute_from_graph to handle sub-graphs

        Logically the method should:
            * Start at the dag.start_at of the dag.
            * Call the self.execute_from_graph(node)
            * depending upon the status of the execution, either move to the success node or failure node.

        Args:
            dag (Graph): The directed acyclic graph to traverse and execute.
            map_variable (dict, optional): If the node if of a map state, this corresponds to the value of the iterable.
                    Defaults to None.
        """
        current_node = dag.start_at
        previous_node = None
        logger.info(f"Running the execution with {current_node}")

        while True:
            working_on = dag.get_node_by_name(current_node)

            if previous_node == current_node:
                raise Exception("Potentially running in a infinite loop")

            previous_node = current_node

            logger.info(f"Creating execution log for {working_on}")
            self.execute_from_graph(working_on, map_variable=map_variable, **kwargs)

            _, next_node_name = self._get_status_and_next_node_name(
                current_node=working_on, dag=dag, map_variable=map_variable
            )

            if working_on.node_type in ["success", "fail"]:
                break

            current_node = next_node_name

        run_log = self._context.run_log_store.get_branch_log(
            working_on._get_branch_log_name(map_variable), self._context.run_id
        )

        branch = "graph"
        if working_on.internal_branch_name:
            branch = working_on.internal_branch_name

        logger.info(f"Finished execution of the {branch} with status {run_log.status}")

        # get the final run log
        if branch == "graph":
            run_log = self._context.run_log_store.get_run_log_by_id(run_id=self._context.run_id, full=True)
        print(json.dumps(run_log.model_dump(), indent=4))

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

        node_step_log_name = node._get_step_log_name(map_variable=map_variable)
        logger.info(f"Scanning previous run logs for node logs of: {node_step_log_name}")

        try:
            previous_attempt_log, _ = self.original_run_log.search_step_by_internal_name(node_step_log_name)
        except exceptions.StepLogNotFoundError:
            logger.warning(f"Did not find the node {node.name} in previous run log")
            return True  # We should re-run the node.

        logger.info(f"The original step status: {previous_attempt_log.status}")

        if previous_attempt_log.status == defaults.SUCCESS:
            return False  # We need not run the node

        logger.info(f"The new execution should start executing graph from this node {node.name}")
        return True

    def execute_node(self, node: BaseNode, map_variable: TypeMapVariable = None, **kwargs):
        self._execute_node(node, map_variable=map_variable, **kwargs)
