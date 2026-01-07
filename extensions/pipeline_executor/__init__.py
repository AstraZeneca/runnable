import copy
import logging
import os
from typing import Any, Dict, List, Optional, cast

from runnable import (
    console,
    context,
    defaults,
    exceptions,
    parameters,
    task_console,
    utils,
)
from runnable.datastore import DataCatalog, JsonParameter, RunLog, StepAttempt
from runnable.defaults import IterableParameterModel
from runnable.executor import BasePipelineExecutor
from runnable.graph import Graph
from runnable.nodes import BaseNode

logger = logging.getLogger(defaults.LOGGER_NAME)


class GenericPipelineExecutor(BasePipelineExecutor):
    """
    The skeleton of an executor class.
    Any implementation of an executor should inherit this class and over-ride accordingly.

    This is a loaded base class which has a lot of methods already implemented for "typical" executions.
    Look at the function docs to understand how to use them appropriately.

    For any implementation:
    1). Who/when should the run log be set up?
    2). Who/When should the step log be set up?

    """

    service_name: str = ""
    service_type: str = "pipeline_executor"

    @property
    def _context(self):
        current_context = context.get_run_context()
        if current_context is None:
            raise RuntimeError("No run context available")
        if not isinstance(
            current_context, (context.PipelineContext, context.AsyncPipelineContext)
        ):
            raise TypeError(
                f"Expected PipelineContext or AsyncPipelineContext, got {type(current_context).__name__}"
            )
        return current_context

    def _get_parameters(self) -> Dict[str, JsonParameter]:
        """
        Consolidate the parameters from the environment variables
        and the parameters file.

        The parameters defined in the environment variables take precedence over the parameters file.

        Returns:
            _type_: _description_
        """
        params: Dict[str, JsonParameter] = {}
        if self._context.parameters_file:
            user_defined = utils.load_yaml(self._context.parameters_file) or {}

            for key, value in user_defined.items():
                params[key] = JsonParameter(value=value, kind="json")

        # Update these with some from the environment variables
        params.update(parameters.get_user_set_parameters())
        logger.debug(f"parameters as seen by executor: {params}")
        return params

    def _get_parameters_for_retry(self) -> Dict[str, JsonParameter]:
        """
        Get parameters for execution, handling retry logic.

        For retry runs, loads parameters from original run metadata.
        For normal runs, uses standard parameter loading logic.

        Returns:
            Dict[str, JsonParameter]: Parameters for execution
        """
        if not self._context.is_retry:
            return self._get_parameters()

        # Load original run log to get parameters
        original_run_log = self._context.run_log_store.get_run_log_by_id(
            run_id=self._context.run_id, full=True
        )

        # Warn if user provided new parameters file
        if self._context.parameters_file:
            console.print(
                f"⚠️  [bold yellow]RETRY MODE:[/bold yellow] Ignoring provided parameters file "
                f"'{self._context.parameters_file}'. Using parameters from original run.",
                style="yellow",
            )

        # Check for environment variable parameter overrides
        env_params = {
            key.replace(defaults.PARAMETER_PREFIX, ""): value
            for key, value in os.environ.items()
            if key.startswith(defaults.PARAMETER_PREFIX)
        }

        if env_params:
            console.print(
                f"⚠️  [bold yellow]RETRY MODE:[/bold yellow] Ignoring {len(env_params)} environment "
                f"parameter overrides. Using parameters from original run.",
                style="yellow",
            )

        console.print(
            f"📋 [bold green]RETRY MODE:[/bold green] Using parameters from original run "
            f"'{self._context.run_id}' with {len(original_run_log.parameters or {})} parameters.",
            style="green",
        )

        return original_run_log.parameters or {}

    def _validate_retry_prerequisites(self):
        """
        Validate prerequisites for retry execution.

        Raises:
            RetryValidationError: If retry cannot proceed due to validation failures
        """
        if not self._context.is_retry:
            return  # Not a retry, skip validation

        try:
            # Check if original run log exists
            original_run_log = self._context.run_log_store.get_run_log_by_id(
                run_id=self._context.run_id, full=True
            )
        except exceptions.RunLogNotFoundError:
            raise exceptions.RetryValidationError(
                f"Original run log not found for run_id: {self._context.run_id}. "
                f"Cannot retry a run that doesn't exist.",
                run_id=self._context.run_id,
            )

        # Validate DAG structure hasn't changed
        if original_run_log.dag_hash != self._context.dag_hash:
            raise exceptions.RetryValidationError(
                f"DAG structure has changed since original run. "
                f"Original hash: {original_run_log.dag_hash}, "
                f"Current hash: {self._context.dag_hash}. "
                f"Retry is not allowed when DAG structure changes.",
                run_id=self._context.run_id,
            )

        logger.info(f"Retry validation passed for run_id: {self._context.run_id}")

    def _should_skip_step_in_retry(
        self, node: BaseNode, iter_variable: Optional[IterableParameterModel] = None
    ) -> bool:
        """
        Determine if a step should be skipped during retry execution.

        Steps are skipped if:
        - This is not a retry run AND
        - The step was previously executed AND
        - The last attempt was successful

        Args:
            node: The node to check
            iter_variable: Optional iterable variable if in map context

        Returns:
            bool: True if step should be skipped, False otherwise
        """
        if not self._context.is_retry:
            return False

        step_log_name = node._get_step_log_name(iter_variable)

        try:
            # Get step log from original run
            step_log = self._context.run_log_store.get_step_log(
                step_log_name, self._context.run_id
            )

            # Composite nodes do not have attempts, their status is consolidated from child nodes
            if node.is_composite and step_log.status == defaults.SUCCESS:
                logger.info(
                    f"Skipping composite step '{node.internal_name}' - already successful in original run"
                )
                return True

            if node._is_terminal_node():
                logger.info(
                    f"Terminal step '{node.internal_name}' will always execute in retry runs"
                )
                return False

            # Check if last attempt was successful
            if step_log.attempts:
                last_attempt = step_log.attempts[-1]
                is_successful = last_attempt.status == defaults.SUCCESS

                if is_successful:
                    logger.info(
                        f"Skipping step '{node.internal_name}' - already successful in original run"
                    )
                    return True

            return False

        except exceptions.StepLogNotFoundError:
            # Step was never executed, don't skip
            logger.info(
                f"Step '{node.internal_name}' was never executed in original run - will execute"
            )
            return False

    def _set_up_run_log(self, exists_ok=False):
        """
        Create a run log and put that in the run log store

        If exists_ok, we allow the run log to be already present in the run log store.
        Enhanced to support retry execution with validation only.
        """
        # For retry runs: validate prerequisites and return early
        if self._context.is_retry:
            logger.info(
                f"Validating retry prerequisites for run_id: {self._context.run_id}"
            )
            self._validate_retry_prerequisites()
            logger.info(
                f"Retry validation passed. Reusing existing run log: {self._context.run_id}"
            )
            return  # Don't create new run log, reuse existing one

        # Normal run log creation logic (unchanged)
        try:
            attempt_run_log = self._context.run_log_store.get_run_log_by_id(
                run_id=self._context.run_id, full=False
            )

            logger.warning(
                f"The run log by id: {self._context.run_id} already exists, is this designed?"
            )
            raise exceptions.RunLogExistsError(
                f"The run log by id: {self._context.run_id} already exists and is {attempt_run_log.status}"
            )
        except exceptions.RunLogNotFoundError:
            pass
        except exceptions.RunLogExistsError:
            if exists_ok:
                return
            raise

        # Consolidate and get the parameters
        params = self._get_parameters()

        self._context.run_log_store.create_run_log(
            run_id=self._context.run_id,
            tag=self._context.tag,
            status=defaults.PROCESSING,
            dag_hash=self._context.dag_hash,
        )
        # Any interaction with run log store attributes should happen via API if available.
        self._context.run_log_store.set_parameters(
            run_id=self._context.run_id, parameters=params
        )

        # Update run_config
        run_config = self._context.model_dump()
        logger.debug(f"run_config as seen by executor: {run_config}")
        self._context.run_log_store.set_run_config(
            run_id=self._context.run_id, run_config=run_config
        )

    def _sync_catalog(
        self, stage: str, synced_catalogs=None, allow_file_no_found_exc: bool = False
    ) -> Optional[List[DataCatalog]]:
        """
        1). Identify the catalog settings by over-riding node settings with the global settings.
        2). For stage = get:
                Identify the catalog items that are being asked to get from the catalog
                And copy them to the local compute data folder
        3). For stage = put:
                Identify the catalog items that are being asked to put into the catalog
                Copy the items from local compute folder to the catalog
        4). Add the items onto the step log according to the stage

        Args:
            node (Node): The current node being processed
            step_log (StepLog): The step log corresponding to that node
            stage (str): One of get or put

        Raises:
            Exception: If the stage is not in one of get/put

        """
        assert isinstance(self._context_node, BaseNode)
        if stage not in ["get", "put"]:
            msg = (
                "Catalog service only accepts get/put possible actions as part of node execution."
                f"Sync catalog of the executor: {self.service_name} asks for {stage} which is not accepted"
            )
            logger.exception(msg)
            raise Exception(msg)

        try:
            node_catalog_settings = self._context_node._get_catalog_settings()
        except exceptions.TerminalNodeError:
            return None

        if not (node_catalog_settings and stage in node_catalog_settings):
            logger.info("No catalog settings found for stage: %s", stage)
            # Nothing to get/put from the catalog
            return None

        data_catalogs = []
        for name_pattern in node_catalog_settings.get(stage) or []:
            if stage == "get":
                data_catalog = self._context.catalog.get(
                    name=name_pattern,
                )

            elif stage == "put":
                data_catalog = self._context.catalog.put(
                    name=name_pattern,
                    allow_file_not_found_exc=allow_file_no_found_exc,
                    store_copy=node_catalog_settings.get("store_copy", True),
                )
            else:
                raise Exception(f"Stage {stage} not supported")

            logger.debug(f"Added data catalog: {data_catalog} to step log")
            data_catalogs.extend(data_catalog)

        return data_catalogs

    @property
    def step_attempt_number(self) -> int:
        """
        The attempt number of the current step.
        Orchestrators should use this step to submit multiple attempts of the job.

        Returns:
            int: The attempt number of the current step. Defaults to 1.
        """
        return int(os.environ.get(defaults.ATTEMPT_NUMBER, 1))

    def add_task_log_to_catalog(
        self,
        name: str,
        iter_variable: Optional[IterableParameterModel] = None,
    ):
        log_file_name = utils.make_log_file_name(
            name=name,
            iter_variable=iter_variable,
        )
        task_console.save_text(log_file_name)
        task_console.export_text(clear=True)
        # Put the log file in the catalog
        self._context.catalog.put(name=log_file_name)
        os.remove(log_file_name)

    def _calculate_attempt_number(
        self,
        node: BaseNode,
        iter_variable: Optional[IterableParameterModel] = None,
    ) -> int:
        """
        Calculate the attempt number for a node based on existing attempts in the run log.

        Args:
            node: The node to calculate attempt number for
            iter_variable: Optional iteration variable if node is in a map state

        Returns:
            int: The attempt number (starting from 1)
        """
        step_log_name = node._get_step_log_name(iter_variable)

        try:
            existing_step_log = self._context.run_log_store.get_step_log(
                step_log_name, self._context.run_id
            )
            # If step log exists, increment attempt number based on existing attempts
            return len(existing_step_log.attempts) + 1
        except exceptions.StepLogNotFoundError:
            # This is the first attempt, use attempt number 1
            return 1

    def _execute_node(
        self,
        node: BaseNode,
        iter_variable: Optional[IterableParameterModel] = None,
        mock: bool = False,
    ):
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
            iter_variable: Optional iteration variable if the node is of a map state.
                        Defaults to None.
        """
        # Calculate attempt number based on existing attempts in run log
        current_attempt_number = self._calculate_attempt_number(node, iter_variable)

        # Set the environment variable for this attempt
        os.environ[defaults.ATTEMPT_NUMBER] = str(current_attempt_number)

        logger.info(
            f"Trying to execute node: {node.internal_name}, attempt : {current_attempt_number}"
        )

        self._context_node = node

        data_catalogs_get: Optional[List[DataCatalog]] = self._sync_catalog(stage="get")
        logger.debug(f"data_catalogs_get: {data_catalogs_get}")

        step_log = node.execute(
            iter_variable=iter_variable,
            attempt_number=current_attempt_number,
            mock=mock,
        )

        allow_file_not_found_exc = True
        if step_log.status == defaults.SUCCESS:
            # raise exception if we succeeded but the file was not found
            allow_file_not_found_exc = False

        data_catalogs_put: Optional[List[DataCatalog]] = self._sync_catalog(
            stage="put", allow_file_no_found_exc=allow_file_not_found_exc
        )
        logger.debug(f"data_catalogs_put: {data_catalogs_put}")
        step_log.add_data_catalogs(data_catalogs_put or [])

        # get catalog should always be added to the step log
        step_log.add_data_catalogs(data_catalogs_get or [])

        console.print(f"Summary of the step: {step_log.internal_name}")
        console.print(step_log.get_summary(), style=defaults.info_style)

        self.add_task_log_to_catalog(
            name=self._context_node.internal_name, iter_variable=iter_variable
        )

        self._context_node = None

        self._context.run_log_store.add_step_log(step_log, self._context.run_id)

    def add_code_identities(self, node: BaseNode, attempt_log: StepAttempt):
        """
        Add code identities specific to the implementation.

        The Base class has an implementation of adding git code identities.

        Args:
            attempt_log (StepAttempt): The step attempt log object
            node (BaseNode): The node we are adding the code identities for
        """
        attempt_log.code_identities.append(utils.get_git_code_identity())

    # ═══════════════════════════════════════════════════════════════
    # Shared helpers - called by both sync and async execution paths
    # ═══════════════════════════════════════════════════════════════

    def _prepare_node_for_execution(
        self,
        node: BaseNode,
        iter_variable: Optional[IterableParameterModel] = None,
    ):
        """
        Setup before node execution - shared by sync/async paths.

        Returns None if node should be skipped (retry logic).
        """
        if self._should_skip_step_in_retry(node, iter_variable):
            logger.info(
                f"Skipping execution of '{node.internal_name}' due to retry logic"
            )
            console.print(
                f":fast_forward: Skipping node {node.internal_name} - already successful",
                style="bold yellow",
            )
            return None

        # Handle step log creation for retry vs normal runs
        if self._context.is_retry:
            try:
                step_log = self._context.run_log_store.get_step_log(
                    node._get_step_log_name(iter_variable), self._context.run_id
                )
                logger.info(
                    f"Reusing existing step log for retry: {node.internal_name}"
                )
            except exceptions.StepLogNotFoundError:
                step_log = self._context.run_log_store.create_step_log(
                    node.name, node._get_step_log_name(iter_variable)
                )
                logger.info(f"Creating new step log for retry: {node.internal_name}")
        else:
            step_log = self._context.run_log_store.create_step_log(
                node.name, node._get_step_log_name(iter_variable)
            )

        step_log.step_type = node.node_type
        step_log.status = defaults.PROCESSING
        self._context.run_log_store.add_step_log(step_log, self._context.run_id)

        return step_log

    def _finalize_graph_execution(
        self,
        node: BaseNode,
        dag: Graph,
        iter_variable: Optional[IterableParameterModel] = None,
    ):
        """Finalize after graph traversal - shared by sync/async paths."""
        run_log = self._context.run_log_store.get_branch_log(
            node._get_branch_log_name(iter_variable), self._context.run_id
        )

        branch = "graph"
        if node.internal_branch_name:
            branch = node.internal_branch_name

        logger.info(f"Finished execution of {branch} with status {run_log.status}")

        if dag == self._context.dag:
            run_log = cast(RunLog, run_log)
            console.print("Completed Execution, Summary:", style="bold color(208)")
            console.print(run_log.get_summary(), style=defaults.info_style)

    def execute_from_graph(
        self,
        node: BaseNode,
        iter_variable: Optional[IterableParameterModel] = None,
    ):
        """
        Sync node execution entry point.

        Uses _prepare_node_for_execution helper for setup (shared with async path).
        """
        step_log = self._prepare_node_for_execution(node, iter_variable)
        if step_log is None:
            return  # Skipped due to retry logic

        logger.info(f"Executing node: {node.get_summary()}")

        # Terminal nodes
        if node.node_type in ["success", "fail"]:
            self._execute_node(node, iter_variable=iter_variable)
            return

        # Composite nodes delegate to their sub-graph
        if node.is_composite:
            node.execute_as_graph(iter_variable=iter_variable)
            return

        # Task nodes
        task_name = node._resolve_map_placeholders(node.internal_name, iter_variable)
        console.print(
            f":runner: Executing the node {task_name} ... ", style="bold color(208)"
        )
        self.trigger_node_execution(node=node, iter_variable=iter_variable)

    def trigger_node_execution(
        self,
        node: BaseNode,
        iter_variable: Optional[IterableParameterModel] = None,
    ):
        """
        Call this method only if we are responsible for traversing the graph via
        execute_from_graph().

        We are not prepared to execute node as of now.

        Args:
            node (BaseNode): The node to execute
            iter_variable (str, optional): If the node if of a map state, this corresponds to the value of iterable.
                    Defaults to ''.

        NOTE: We do not raise an exception as this method is not required by many extensions
        """
        pass

    def _get_status_and_next_node_name(
        self,
        current_node: BaseNode,
        dag: Graph,
        iter_variable: Optional[IterableParameterModel] = None,
    ) -> tuple[str, str]:
        """
        Given the current node and the graph, returns the name of the next node to execute.

        The name is always relative the graph that the node resides in.

        If the current node succeeded, we return the next node as per the graph.
        If the current node failed, we return the on failure node of the node (if provided) or the global one.

        This method is only used by interactive executors i.e local and local-container

        Args:
            current_node (BaseNode): The current node.
            dag (Graph): The dag we are traversing.
            iter_variable (dict): If the node belongs to a map branch.

        """

        step_log = self._context.run_log_store.get_step_log(
            current_node._get_step_log_name(iter_variable), self._context.run_id
        )
        logger.info(
            f"Finished executing the node {current_node} with status {step_log.status}"
        )

        try:
            next_node_name = current_node._get_next_node()
        except exceptions.TerminalNodeError:
            next_node_name = ""

        if step_log.status == defaults.FAIL:
            next_node_name = dag.get_fail_node().name
            if current_node._get_on_failure_node():
                next_node_name = current_node._get_on_failure_node()

        return step_log.status, next_node_name

    def execute_graph(
        self,
        dag: Graph,
        iter_variable: Optional[IterableParameterModel] = None,
    ):
        """
        The parallelization is controlled by the nodes and not by this function.

        Transpilers should over ride this method to do the translation of dag to the
        platform specific way.

        Interactive methods should use this to traverse and execute the dag.
            - Use execute_from_graph to handle sub-graphs

        Logically the method should:
            * Start at the dag.start_at of the dag.
            * Call the self.execute_from_graph(node)
            * depending upon the status of the execution, either move to the
            success node or failure node.

        Args:
            dag (Graph): The directed acyclic graph to traverse and execute.
            iter_variable (dict, optional): If the node if of a map state, this
                corresponds to the value of the iterable.
            Defaults to None.
        """
        current_node = dag.start_at
        previous_node = None
        logger.info(f"Running the execution with {current_node}")
        logger.info(
            "iter_variable: %s",
            iter_variable.model_dump_json() if iter_variable else None,
        )

        branch_task_name: str = ""
        if dag.internal_branch_name:
            branch_task_name = BaseNode._resolve_map_placeholders(
                dag.internal_branch_name or "Graph",
                iter_variable,
            )
            console.print(
                f":runner: Executing the branch {branch_task_name} ... ",
                style="bold color(208)",
            )

        while True:
            working_on = dag.get_node_by_name(current_node)
            task_name = working_on._resolve_map_placeholders(
                working_on.internal_name, iter_variable
            )

            if previous_node == current_node:
                raise Exception("Potentially running in a infinite loop")

            previous_node = current_node

            try:
                self.execute_from_graph(working_on, iter_variable=iter_variable)
                status, next_node_name = self._get_status_and_next_node_name(
                    current_node=working_on, dag=dag, iter_variable=iter_variable
                )

                if status == defaults.SUCCESS:
                    console.print(
                        f":white_check_mark: Node {task_name} succeeded",
                    )
                else:
                    console.print(
                        f":x: Node {task_name} failed",
                    )
            except Exception as e:  # noqa: E722
                console.print(":x: Error during execution", style="bold red")
                console.print(e, style=defaults.error_style)
                logger.exception(e)
                raise

            console.rule(style="[dark orange]")

            if working_on.node_type in ["success", "fail"]:
                break

            current_node = next_node_name

        # Use shared helper for finalization
        self._finalize_graph_execution(working_on, dag, iter_variable)

    def send_return_code(self, stage="traversal"):
        """
        Convenience function used by pipeline to send return code to the caller of the cli

        Raises:
            Exception: If the pipeline execution failed
        """
        run_id = self._context.run_id

        run_log = self._context.run_log_store.get_run_log_by_id(
            run_id=run_id, full=False
        )
        if run_log.status == defaults.FAIL:
            raise exceptions.ExecutionFailedError(run_id=run_id)

    def _resolve_executor_config(self, node: BaseNode) -> Dict[str, Any]:
        """
        The overrides section can contain specific over-rides to an global executor config.
        To avoid too much clutter in the dag definition, we allow the configuration file to
        have overrides block.

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
        try:
            ctx_node_config = node._get_executor_config(self.service_name)
        except exceptions.TerminalNodeError:
            # Some modes request for effective node config even for success or fail nodes
            return utils.apply_variables(effective_node_config, self._context.variables)

        if ctx_node_config:
            if ctx_node_config not in self.overrides:
                raise Exception(
                    f"No override of key: {ctx_node_config} found in the overrides section"
                )

            effective_node_config.update(self.overrides[ctx_node_config])

        effective_node_config = utils.apply_variables(
            effective_node_config, self._context.variables
        )
        logger.debug(f"Effective node config: {effective_node_config}")

        return effective_node_config

    def fan_out(
        self,
        node: BaseNode,
        iter_variable: Optional[IterableParameterModel] = None,
    ):
        """
        This method is used to appropriately fan-out the execution of a composite node.
        This is only useful when we want to execute a composite node during 3rd party orchestrators.

        Reason: Transpilers typically try to run the leaf nodes but do not have any capacity
        to do anything for the step which is composite. By calling this fan-out before calling the
        leaf nodes, we have an opportunity to do the right set up (creating the step log,
        exposing the parameters, etc.) for the composite step.

        All 3rd party orchestrators should use this method to fan-out the execution of
        a composite node.
        This ensures:
            - The dot path notation is preserved, this method should create the step and
            call the node's fan out to create the branch logs and let the 3rd party do the
            actual step execution.
            - Gives 3rd party orchestrators an opportunity to set out the required
            for running a composite node.

        Args:
            node (BaseNode): The node to fan-out
            iter_variable (dict, optional): If the node if of a map state,.Defaults to None.

        """
        step_log = self._context.run_log_store.create_step_log(
            node.name, node._get_step_log_name(iter_variable=iter_variable)
        )

        step_log.step_type = node.node_type
        step_log.status = defaults.PROCESSING
        self._context.run_log_store.add_step_log(step_log, self._context.run_id)

        node.fan_out(iter_variable=iter_variable)

    def fan_in(
        self,
        node: BaseNode,
        iter_variable: Optional[IterableParameterModel] = None,
    ):
        """
        This method is used to appropriately fan-in after the execution of a composite node.
        This is only useful when we want to execute a composite node during 3rd party orchestrators.

        Reason: Transpilers typically try to run the leaf nodes but do not have any capacity
        to do anything for the step which is composite. By calling this fan-in after calling
        the leaf nodes, we have an opportunity to act depending upon the status of the
        individual branches.

        All 3rd party orchestrators should use this method to fan-in the execution of a
        composite node.
        This ensures:
            - Gives the renderer's the control on where to go depending upon the state of
                the composite node.
            - The status of the step and its underlying branches are correctly updated.

        Args:
            node (BaseNode): The node to fan-in
            iter_variable (dict, optional): If the node if of a map state,.Defaults to None.

        """
        node.fan_in(iter_variable=iter_variable)

        step_log = self._context.run_log_store.get_step_log(
            node._get_step_log_name(iter_variable=iter_variable), self._context.run_id
        )

        if step_log.status == defaults.FAIL:
            raise Exception(f"Step {node.name} failed")
