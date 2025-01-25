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
from runnable.datastore import DataCatalog, JsonParameter, RunLog, StepLog
from runnable.defaults import TypeMapVariable
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
        assert context.run_context
        return context.run_context

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

    def _set_up_run_log(self, exists_ok=False):
        """
        Create a run log and put that in the run log store

        If exists_ok, we allow the run log to be already present in the run log store.
        """
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
        run_config = utils.get_run_config()
        logger.debug(f"run_config as seen by executor: {run_config}")
        self._context.run_log_store.set_run_config(
            run_id=self._context.run_id, run_config=run_config
        )

    def _sync_catalog(
        self, stage: str, synced_catalogs=None
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

        compute_data_folder = self.get_effective_compute_data_folder()

        data_catalogs = []
        for name_pattern in node_catalog_settings.get(stage) or []:
            if stage == "get":
                data_catalog = self._context.catalog_handler.get(
                    name=name_pattern,
                    run_id=self._context.run_id,
                    compute_data_folder=compute_data_folder,
                )

            elif stage == "put":
                data_catalog = self._context.catalog_handler.put(
                    name=name_pattern,
                    run_id=self._context.run_id,
                    compute_data_folder=compute_data_folder,
                    synced_catalogs=synced_catalogs,
                )

            logger.debug(f"Added data catalog: {data_catalog} to step log")
            data_catalogs.extend(data_catalog)

        return data_catalogs

    def get_effective_compute_data_folder(self) -> str:
        """
        Get the effective compute data folder for the given stage.
        If there is nothing to catalog, we return None.

        The default is the compute data folder of the catalog but this can be over-ridden by the node.

        Args:
            stage (str): The stage we are in the process of cataloging


        Returns:
            str: The compute data folder as defined by the node defaulting to catalog handler
        """
        assert isinstance(self._context_node, BaseNode)
        compute_data_folder = self._context.catalog_handler.compute_data_folder

        catalog_settings = self._context_node._get_catalog_settings()
        effective_compute_data_folder = (
            catalog_settings.get("compute_data_folder", "") or compute_data_folder
        )

        return effective_compute_data_folder

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
        self, name: str, map_variable: Dict[str, str | int | float] | None = None
    ):
        log_file_name = utils.make_log_file_name(
            name=name,
            map_variable=map_variable,
        )
        task_console.save_text(log_file_name)
        # Put the log file in the catalog
        self._context.catalog_handler.put(
            name=log_file_name, run_id=self._context.run_id
        )
        os.remove(log_file_name)

    def _execute_node(
        self,
        node: BaseNode,
        map_variable: TypeMapVariable = None,
        mock: bool = False,
        **kwargs,
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
            map_variable (dict, optional): If the node is of a map state, map_variable is the value of the iterable.
                        Defaults to None.
        """
        logger.info(
            f"Trying to execute node: {node.internal_name}, attempt : {self.step_attempt_number}"
        )

        self._context_node = node

        data_catalogs_get: Optional[List[DataCatalog]] = self._sync_catalog(stage="get")
        logger.debug(f"data_catalogs_get: {data_catalogs_get}")

        step_log = node.execute(
            map_variable=map_variable,
            attempt_number=self.step_attempt_number,
            mock=mock,
            **kwargs,
        )

        data_catalogs_put: Optional[List[DataCatalog]] = self._sync_catalog(stage="put")
        logger.debug(f"data_catalogs_put: {data_catalogs_put}")

        step_log.add_data_catalogs(data_catalogs_get or [])
        step_log.add_data_catalogs(data_catalogs_put or [])

        console.print(f"Summary of the step: {step_log.internal_name}")
        console.print(step_log.get_summary(), style=defaults.info_style)

        self._context_node = None

        self._context.run_log_store.add_step_log(step_log, self._context.run_id)

    def add_code_identities(self, node: BaseNode, step_log: StepLog, **kwargs):
        """
        Add code identities specific to the implementation.

        The Base class has an implementation of adding git code identities.

        Args:
            step_log (object): The step log object
            node (BaseNode): The node we are adding the step log for
        """
        step_log.code_identities.append(utils.get_git_code_identity())

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
            self._execute_node(node, map_variable=map_variable, **kwargs)
            return

        # We call an internal function to iterate the sub graphs and execute them
        if node.is_composite:
            node.execute_as_graph(map_variable=map_variable, **kwargs)
            return

        task_console.export_text(clear=True)

        task_name = node._resolve_map_placeholders(node.internal_name, map_variable)
        console.print(
            f":runner: Executing the node {task_name} ... ", style="bold color(208)"
        )
        self.trigger_node_execution(node=node, map_variable=map_variable, **kwargs)

    def trigger_node_execution(
        self, node: BaseNode, map_variable: TypeMapVariable = None, **kwargs
    ):
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
        pass

    def _get_status_and_next_node_name(
        self, current_node: BaseNode, dag: Graph, map_variable: TypeMapVariable = None
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
            map_variable (dict): If the node belongs to a map branch.

        """

        step_log = self._context.run_log_store.get_step_log(
            current_node._get_step_log_name(map_variable), self._context.run_id
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

        branch_execution_task = None
        branch_task_name: str = ""
        if dag.internal_branch_name:
            branch_task_name = BaseNode._resolve_map_placeholders(
                dag.internal_branch_name or "Graph",
                map_variable,
            )
            branch_execution_task = self._context.progress.add_task(
                f"[dark_orange]Executing {branch_task_name}",
                total=1,
            )

        while True:
            working_on = dag.get_node_by_name(current_node)
            task_name = working_on._resolve_map_placeholders(
                working_on.internal_name, map_variable
            )

            if previous_node == current_node:
                raise Exception("Potentially running in a infinite loop")

            previous_node = current_node

            logger.debug(f"Creating execution log for {working_on}")

            depth = " " * ((task_name.count(".")) or 1 - 1)

            task_execution = self._context.progress.add_task(
                f"{depth}Executing {task_name}", total=1
            )

            try:
                self.execute_from_graph(working_on, map_variable=map_variable, **kwargs)
                status, next_node_name = self._get_status_and_next_node_name(
                    current_node=working_on, dag=dag, map_variable=map_variable
                )

                if status == defaults.SUCCESS:
                    self._context.progress.update(
                        task_execution,
                        description=f"{depth}[green] {task_name} Completed",
                        completed=True,
                        overflow="fold",
                    )
                else:
                    self._context.progress.update(
                        task_execution,
                        description=f"{depth}[red] {task_name} Failed",
                        completed=True,
                    )  # type ignore
            except Exception as e:  # noqa: E722
                self._context.progress.update(
                    task_execution,
                    description=f"{depth}[red] {task_name} Errored",
                    completed=True,
                )
                console.print(e, style=defaults.error_style)
                logger.exception(e)
                raise
            finally:
                # Add task log to the catalog
                self.add_task_log_to_catalog(
                    name=working_on.internal_name, map_variable=map_variable
                )

            console.rule(style="[dark orange]")

            if working_on.node_type in ["success", "fail"]:
                break

            current_node = next_node_name

        if branch_execution_task:
            self._context.progress.update(
                branch_execution_task,
                description=f"[green3] {branch_task_name} completed",
                completed=True,
            )

        run_log = self._context.run_log_store.get_branch_log(
            working_on._get_branch_log_name(map_variable), self._context.run_id
        )

        branch = "graph"
        if working_on.internal_branch_name:
            branch = working_on.internal_branch_name

        logger.info(f"Finished execution of the {branch} with status {run_log.status}")

        # We are in the root dag
        if dag == self._context.dag:
            run_log = cast(RunLog, run_log)
            console.print("Completed Execution, Summary:", style="bold color(208)")
            console.print(run_log.get_summary(), style=defaults.info_style)

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

    def fan_out(self, node: BaseNode, map_variable: TypeMapVariable = None):
        """
        This method is used to appropriately fan-out the execution of a composite node.
        This is only useful when we want to execute a composite node during 3rd party orchestrators.

        Reason: Transpilers typically try to run the leaf nodes but do not have any capacity to do anything for the
        step which is composite. By calling this fan-out before calling the leaf nodes, we have an opportunity to
        do the right set up (creating the step log, exposing the parameters, etc.) for the composite step.

        All 3rd party orchestrators should use this method to fan-out the execution of a composite node.
        This ensures:
            - The dot path notation is preserved, this method should create the step and call the node's fan out to
            create the branch logs and let the 3rd party do the actual step execution.
            - Gives 3rd party orchestrators an opportunity to set out the required for running a composite node.

        Args:
            node (BaseNode): The node to fan-out
            map_variable (dict, optional): If the node if of a map state,.Defaults to None.

        """
        step_log = self._context.run_log_store.create_step_log(
            node.name, node._get_step_log_name(map_variable=map_variable)
        )

        self.add_code_identities(node=node, step_log=step_log)

        step_log.step_type = node.node_type
        step_log.status = defaults.PROCESSING
        self._context.run_log_store.add_step_log(step_log, self._context.run_id)

        node.fan_out(executor=self, map_variable=map_variable)

    def fan_in(self, node: BaseNode, map_variable: TypeMapVariable = None):
        """
        This method is used to appropriately fan-in after the execution of a composite node.
        This is only useful when we want to execute a composite node during 3rd party orchestrators.

        Reason: Transpilers typically try to run the leaf nodes but do not have any capacity to do anything for the
        step which is composite. By calling this fan-in after calling the leaf nodes, we have an opportunity to
        act depending upon the status of the individual branches.

        All 3rd party orchestrators should use this method to fan-in the execution of a composite node.
        This ensures:
            - Gives the renderer's the control on where to go depending upon the state of the composite node.
            - The status of the step and its underlying branches are correctly updated.

        Args:
            node (BaseNode): The node to fan-in
            map_variable (dict, optional): If the node if of a map state,.Defaults to None.

        """
        node.fan_in(executor=self, map_variable=map_variable)

        step_log = self._context.run_log_store.get_step_log(
            node._get_step_log_name(map_variable=map_variable), self._context.run_id
        )

        if step_log.status == defaults.FAIL:
            raise Exception(f"Step {node.name} failed")
