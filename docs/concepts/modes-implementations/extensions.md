To extend and implement a custom compute mode, you need to over-ride the appropriate methods of the ```Base``` class.

Most of the methods of the ```BaseRunLogStore``` have default implementations and need not be over-written in a few 
situations.

Please refer to [*Guide to extensions* ](../../../extensions/extensions/) for a detailed explanation and the need for
implementing a *Integration* pattern along with the extension along with understanding the right example for your
extension.

In summary, the extension will fall into one of the four possible possibilities:

- Magnus traverses, execution environment same as traversal. eg: local
- Magnus traverses, execution environment not same as traversal. eg: local-container
- Magnus does not traverse, execution environment not same as traversal. eg: demo-renderer
- Magnus does not traverse, execution environment same as traversal. eg: 
[advanced use of as-is](../../../examples/#advanced_use_as-is)




Extensions that are being actively worked on and listed to be released as part of ```magnus-extensions```

- local-aws-batch : A decentralized AWS batch compute
- aws-step-function: Translates the dag into a Step function.

```python
# You can find this in the source code at: magnus/executor.py along with a few example 
# implementations of local, local-container, demo-renderer
class BaseExecutor:
    """
    The skeleton of an executor class.
    Any implementation of an executor should inherit this class and over-ride accordingly.

    The logic of any dag execution is a play between three methods of this class.

    execute_graph:
        This method is responsible for traversing A graph.
        The core logic is start at the start_at the graph and traverse according to the state of the execution.
        execute_graph hands over the actual execution of the node to self.execute_from_graph

        Helper method: prepare_for_graph_execution would be called prior to calling execute_graph.
            Use it to modify settings if needed.

    execute_from_graph:
        This method is responsible for executing a node.
        But given that a node itself could be a dag in cases of parallel, map and dag, this method handles the cases.
        If the node is of type task, success, fail: we can pretty much execute it and we call self.trigger_job.
        If the node is of type dag, map, parallel: We call the node's execute_as_graph function which internally
        triggers execute_graph in-turn iteratively traverses the graph.


    execute_node:
        This method is where the actual execution of the work happens.
        This method is already in the compute environment of the mode.
        Use prepare_node_for_execution to adjust settings.
        The base class is given an implementation and in most cases should not be touched.

        Helper method: prepare_for_node_execution would be called prior to calling execute_node.
            Use it to modify settings if needed

    The above logic holds good when we are in interactive compute mode i.e. local, local-container, local-aws-batch

    But in 3rd party orchestration mode, we might have to render the job specifications and the roles might be different

    Please see the implementations of local, local-container, local-aws-batch to perform interactive compute.
    And demo-renderer to see an example of what a 3rd party executor looks like.
    """
    service_name = ''

    def __init__(self, config):
        # pylint: disable=R0914,R0913
        self.config = config
        # The remaining would be attached later
        self.pipeline_file = None
        self.variables_file = None
        self.run_id = None
        self.dag = None
        self.use_cached = None
        self.tag = None
        self.run_log_store = None
        self.previous_run_log = None
        self.dag_hash = None
        self.catalog_handler = None
        self.secrets_handler = None
        self.variables_file = None
        self.configuration_file = None
        self.cmd_line_arguments = {}

    def is_parallel_execution(self) -> bool:  # pylint: disable=R0201
        """
        Controls the parallelization of branches in map and parallel state.
        Defaults to False and left for the compute modes to decide.

        Returns:
            bool: True if the mode allows parallel execution of branches.
        """
        return defaults.ENABLE_PARALLEL

    def set_up_run_log(self):
        """
        Create a run log and put that in the run log store
        """
        run_log = self.run_log_store.create_run_log(self.run_id)
        run_log.tag = self.tag
        run_log.use_cached = False
        run_log.status = defaults.PROCESSING
        run_log.dag_hash = self.dag_hash

        parameters = self.cmd_line_arguments
        if self.previous_run_log:
            run_log.original_run_id = self.previous_run_log.run_id
            # Sync the previous run log catalog to this one.
            self.catalog_handler.sync_between_runs(previous_run_id=run_log.original_run_id, run_id=self.run_id)
            run_log.use_cached = True
            parameters.update(self.previous_run_log.parameters)

        run_log.parameters = parameters

        # Update run_config
        run_log.run_config = utils.get_run_config(self)

        self.run_log_store.put_run_log(run_log)

    def prepare_for_graph_execution(self):
        """
        This method would be called prior to calling execute_graph.
        Perform any steps required before doing the graph execution.

        The most common implementation is to prepare a run log for the run if the run uses local interactive compute.

        But in cases of actual rendering the job specs (eg: AWS step functions, K8's) we need not do anything.
        """

        integration.validate(self, self.run_log_store)
        integration.configure_for_traversal(self, self.run_log_store)

        integration.validate(self, self.catalog_handler)
        integration.configure_for_traversal(self, self.catalog_handler)

        integration.validate(self, self.secrets_handler)
        integration.configure_for_traversal(self, self.secrets_handler)

        self.set_up_run_log()

    def prepare_for_node_execution(self, node: BaseNode, map_variable: dict = None):
        """
        Perform any modifications to the services prior to execution of the node.

        Args:
            node (Node): [description]
            map_variable (dict, optional): [description]. Defaults to None.
        """

        integration.validate(self, self.run_log_store)
        integration.configure_for_execution(self, self.run_log_store)

        integration.validate(self, self.catalog_handler)
        integration.configure_for_execution(self, self.catalog_handler)

        integration.validate(self, self.secrets_handler)
        integration.configure_for_execution(self, self.secrets_handler)

    def sync_catalog(self, node: BaseNode, step_log: datastore.StepLog, stage: str, synced_catalogs=None):
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
            step_log (datastore.StepLog): The step log corresponding to that node
            stage (str): One of get or put
        """
        if stage not in ['get', 'put']:
            msg = (
                'Catalog service only accepts get/put possible actions as part of node execution.'
                f'Sync catalog of the executor: {self.service_name} asks for {stage} which is not accepted'
            )
            raise Exception(msg)

        node_catalog_settings = node.get_catalog_settings()
        if not (node_catalog_settings and stage in node_catalog_settings):
            # Nothing to get/put from the catalog
            return None

        # Local compute data folder over rides the global one
        compute_data_folder = self.catalog_handler.compute_data_folder
        if 'compute_data_folder' in node_catalog_settings and node_catalog_settings['compute_data_folder']:
            compute_data_folder = node_catalog_settings['compute_data_folder']

        data_catalogs = []
        for name_pattern in node_catalog_settings.get(stage) or []:  #  Assumes a list
            data_catalogs = getattr(
                self.catalog_handler, stage)(
                name=name_pattern, run_id=self.run_id, compute_data_folder=compute_data_folder,
                synced_catalogs=synced_catalogs)

        if data_catalogs:
            step_log.add_data_catalogs(data_catalogs)

        return data_catalogs

    def execute_node(self, node: BaseNode, map_variable: dict = None, **kwargs):
        """
        This is the entry point when we do the actual execution of the function.
        Over-ride this function to do what the executor has to do for the actual function call
        Most frequently, this core logic should not be touched either in interactive mode or 3rd party orchestration
        mode.

        While in interactive mode, we just compute, in 3rd party interactive mode, we call this function from the CLI

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
        max_attempts = node.get_max_attempts()
        attempts = 0
        step_log = self.run_log_store.get_step_log(node.get_step_log_name(map_variable), self.run_id)

        parameters = self.run_log_store.get_parameters(run_id=self.run_id)
        interaction.store_parameter(**parameters)

        data_catalogs_get = self.sync_catalog(node, step_log, stage='get')

        mock = step_log.mock
        logger.info(f'Trying to execute node: {node.internal_name}, attempt : {attempts}, max_attempts: {max_attempts}')
        while attempts < max_attempts:
            try:
                attempt_log = node.execute(executor=self, mock=mock,
                                           map_variable=map_variable, **kwargs)
                attempt_log.attempt_number = attempts
                step_log.attempts.append(attempt_log)
                if attempt_log.status == defaults.FAIL:
                    raise Exception()

                step_log.status = defaults.SUCCESS
                step_log.user_defined_metrics = utils.get_tracked_data()
                self.run_log_store.set_parameters(self.run_id, utils.get_user_set_parameters(remove=True))
                break
            except Exception as _e:  # pylint: disable=W0703
                attempts += 1
                logger.exception(f'Node: {node} failed with exception {_e}')
                # Remove any steps data
                utils.get_tracked_data()
                utils.get_user_set_parameters(remove=True)

            if attempts == max_attempts:
                step_log.status = defaults.FAIL
                logger.error(f'Node {node} failed, max retries of {max_attempts} reached')

        self.sync_catalog(node, step_log, stage='put', synced_catalogs=data_catalogs_get)
        self.run_log_store.add_step_log(step_log, self.run_id)

    def add_code_identities(self, node: BaseNode, step_log: datastore.StepLog, **kwargs):
        """
        Add code identities specific to the implementation.

        The Base class has an implementation of adding git code identities.

        Args:
            step_log (object): The step log object
            node (BaseNode): The node we are adding the step log for
        """
        step_log.code_identities.append(utils.get_git_code_identity(self.run_log_store))

    def execute_from_graph(self, node: BaseNode, map_variable: dict = None, **kwargs):
        """
        This is the entry point to from the graph execution.

        While the self.execute_graph is responsible for traversing the graph, this function is responsible for
        actual execution of the node.

        If the node type is:
            * task : We can delegate to execute_node after checking the eligibility for re-run in cases of a re-run
            * success: We can delegate to execute_node
            * fail: We can delegate to execute_node

        For nodes that are internally graphs:
            * parallel: Delegate the responsibility of execution to the node.execute_as_graph()
            * dag: Delegate the responsibility of execution to the node.execute_as_graph()
            * map: Delegate the responsibility of execution to the node.execute_as_graph()

        Check the implementations of local, local-container, local-aws-batch for different examples of implementation

        Args:
            node (Node): The node to execute
            map_variable (dict, optional): If the node if of a map state, this corresponds to the value of iterable.
                    Defaults to None.
        """
        step_log = self.run_log_store.create_step_log(node.name, node.get_step_log_name(map_variable))

        self.add_code_identities(node=node, step_log=step_log)

        step_log.step_type = node.node_type
        step_log.status = defaults.PROCESSING

        # Add the step log to the database as per the situation.
        # If its a terminal node, complete it now
        if node.node_type in ['success', 'fail']:
            self.run_log_store.add_step_log(step_log, self.run_id)
            self.execute_node(node, map_variable=map_variable, **kwargs)
            return

        # If previous run was successful, move on to the next step
        if not self.is_eligible_for_rerun(node, map_variable=map_variable):
            step_log.mock = True
            step_log.status = defaults.SUCCESS
            self.run_log_store.add_step_log(step_log, self.run_id)
            return

        # We call an internal function to iterate the sub graphs and execute them
        if node.node_type in ['parallel', 'dag', 'map']:
            self.run_log_store.add_step_log(step_log, self.run_id)
            node.execute_as_graph(self, map_variable=map_variable, **kwargs)
            return

        # Executor specific way to trigger a job
        self.run_log_store.add_step_log(step_log, self.run_id)
        self.trigger_job(node=node, map_variable=map_variable, **kwargs)

    def trigger_job(self, node: BaseNode, map_variable: dict = None, **kwargs):
        """
        Executor specific way of triggering jobs.

        Args:
            node (BaseNode): The node to execute
            map_variable (str, optional): If the node if of a map state, this corresponds to the value of iterable.
                    Defaults to ''.

        Raises: NotImplementedError Base class hence not implemented
        """
        raise NotImplementedError

    def get_status_and_next_node_name(self, current_node: BaseNode, dag: Graph, map_variable: dict = None):
        """
        Given the current node and the graph, returns the name of the next node to execute.

        The name is always relative the graph that the node resides in.

        If the current node succeeded, we return the next node as per the graph.
        If the current node failed, we return the on failure node of the node (if provided) or the global one.

        Args:
            current_node (BaseNode): The current node.
            dag (Graph): The dag we are traversing.
            map_variable (dict): If the node belongs to a map branch.
        """

        step_log = self.run_log_store.get_step_log(current_node.get_step_log_name(map_variable), self.run_id)
        logger.info(
            f'Finished executing the node {current_node} with status {step_log.status}')

        next_node_name = current_node.get_next_node()

        if step_log.status == defaults.FAIL:
            next_node_name = dag.get_fail_node().name
            if current_node.get_on_failure_node():
                next_node_name = current_node.get_on_failure_node()

        return step_log.status, next_node_name

    def execute_graph(self, dag: Graph, map_variable: dict = None, **kwargs):
        """
        The parallelization is controlled by the nodes and not by this function.

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
        logger.info(f'Running the execution with {current_node}')
        while True:
            working_on = dag.get_node_by_name(current_node)

            if previous_node == current_node:
                raise Exception('Potentially running in a infinite loop')

            previous_node = current_node

            logger.info(f'Creating execution log for {working_on}')
            self.execute_from_graph(working_on, map_variable=map_variable, **kwargs)

            status, next_node_name = self.get_status_and_next_node_name(
                current_node=working_on, dag=dag, map_variable=map_variable)

            if status == defaults.TRIGGERED:
                # Some nodes go into triggered state and self traverse
                logger.info(f'Triggered the job to execute the node {current_node}')
                break

            if working_on.node_type in ['success', 'fail']:
                break

            current_node = next_node_name

        run_log = self.run_log_store.get_branch_log(working_on.get_branch_log_name(map_variable), self.run_id)

        branch = 'graph'
        if working_on.internal_branch_name:
            branch = working_on.internal_branch_name

        logger.info(f'Finished execution of the {branch} with status {run_log.status}')
        print(json.dumps(run_log.dict(), indent=4))

    def is_eligible_for_rerun(self, node, map_variable: dict = None):
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
        if self.previous_run_log:
            node_step_log_name = node.get_step_log_name(map_variable=map_variable)
            logger.info(f'Scanning previous run logs for node logs of: {node_step_log_name}')

            previous_node_log = None
            try:
                previous_node_log, _ = self.previous_run_log.search_step_by_internal_name(node_step_log_name)
            except exceptions.StepLogNotFoundError:
                logger.warning(f'Did not find the node {node.name} in previous run log')
                return True  # We should re-run the node.

            step_log = self.run_log_store.get_step_log(node.get_step_log_name(map_variable), self.run_id)
            logger.info(f'The original step status: {previous_node_log.status}')

            if previous_node_log.status == defaults.SUCCESS:
                logger.info(f'The step {node.name} is marked success, not executing it')
                step_log.status = defaults.SUCCESS
                step_log.message = 'Node execution successful in previous run, skipping it'
                self.run_log_store.add_step_log(step_log, self.run_id)
                return False  # We need not run the node

            # Remove previous run log to start execution from this step
            logger.info(f'The new execution should start executing graph from this node {node.name}')
            self.previous_run_log = None
        return True

    def send_return_code(self, stage='traversal'):
        """
        Convenience function used by pipeline to send return code to the caller of the cli

        Raises:
            Exception: If the pipeline execution failed
        """
        run_id = self.run_id

        run_log = self.run_log_store.get_run_log_by_id(run_id=run_id, full=False)
        if run_log.status == defaults.FAIL:
            raise Exception('Pipeline execution failed')
```

