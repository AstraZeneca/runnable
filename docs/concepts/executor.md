# Executors

In magnus, executors essentially represent either the compute resource or the orchestration framework.

Conceptually, a executor can be one of two types:

- **Interactive executors**: In this mode, magnus does the work of executing the pipeline/function/notebook

```shell
magnus execute --file my-project.yaml --config-file config.yaml

magnus execute_notebook interesting.ipynb --config-file config.yaml

magnus execute_function my_module.my_function --config-file config.yaml
```

Magnus takes care of the execution of the pipeline/function or notebook in the compute you requested. Examples
of this executors or local, local container etc.

- **Orchestration executors**: In this mode, the dag definition is transpiled to your preferred orchestration language
of dag definition. To still achieve the capabilities of interactive executors, the orchestration language is
directed to call an internal method instead of your actual function.

Specifically, the orchestration is asked to call

```shell
magnus execute_single_node --file my-project.yaml --config-file config.yaml --step-name step-to-call
```

The branches of the original dag are also translated to the orchestrators language if its supported. If the
orchestration executors does not support a feature that magnus supports, you could still make it work by a mixed model.

Examples of orchestration modes are aws step functions, kubeflow job specification, argo job specification etc.

## Configuration

As with any system within magnus, configuration of an executor can be done by:

```yaml
executor:
  type:
  config:
```

### type

The type of mode provider you want. This should be one of the executors types already available.

Local executor is provided by default if nothing is provided.

### config

Any configuration parameters the execution provider accepts.

## Parameterized definition

As with any part of the magnus configuration, you can parameterize the configuration of Mode to switch between
Mode providers without changing the base definition.

Please follow the example provided [here](../dag/#parameterized_definition) for more information.

## Extensions

You can easily extend magnus to interpret the dag definition to a orchestration language of choice, if a default
implementation does not exist or you are not happy with the implementation.

The ```BaseExecutor``` implementation is as follows:

```python
# The code can be found at magnus/executor.py
# The "_private" methods should not be touched without significant knowledge about the design

from pydantic import BaseModel

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
        If the node is of type composite: We call the node's execute_as_graph function which internally
        triggers execute_graph in-turn iteratively traverses the graph.


    _execute_node:
        This method is where the actual execution of the work happens.
        This method is already in the compute environment of the mode.
        Use prepare_node_for_execution to adjust settings.
        The base class is given an implementation and in most cases should not be touched.

        Helper method: prepare_for_node_execution would be called prior to calling _execute_node.
            Use it to modify settings if needed

    The above logic holds good when we are in interactive compute mode i.e. local, local-container, local-aws-batch

    But in 3rd party orchestration mode, we might have to render the job specifications and the roles might be different

    Please see the implementations of local, local-container, local-aws-batch to perform interactive compute.
    And demo-renderer to see an example of what a 3rd party executor looks like.
    """
    service_name = ''

    class Config(BaseModel):
        enable_parallel: bool = defaults.ENABLE_PARALLEL
        placeholders: dict = {}

    def __init__(self, config: dict = None):
        # pylint: disable=R0914,R0913
        config = config or {}
        self.config = self.Config(**config)
        # The remaining would be attached later
        # The definition files
        self.pipeline_file = None
        self.variables_file = None
        self.parameters_file = None
        self.configuration_file = None
        # run descriptors
        self.tag: str = ''
        self.run_id: str = ''
        self.single_step: bool = False
        self.variables: Dict[str, str] = {}
        self.use_cached: bool = False
        self.dag: Graph = None  # type: ignore
        self.dag_hash: str = ''
        self.execution_plan: str = ''
        # Services
        self.catalog_handler: BaseCatalog = None  # type: ignore
        self.secrets_handler: BaseSecrets = None  # type: ignore
        self.experiment_tracker: BaseExperimentTracker = None  # type: ignore
        self.run_log_store: BaseRunLogStore = None  # type: ignore
        self.previous_run_log = None

        self.context_step_log: StepLog = None  # type: ignore


    def _is_parallel_execution(self) -> bool:
        """
        Controls the parallelization of branches in map and parallel state.
        Defaults to False and left for the compute modes to decide.

        Returns:
            bool: True if the mode allows parallel execution of branches.
        """
        return self.config.enable_parallel

    def _set_up_run_log(self, exists_ok=False):
        """
        Create a run log and put that in the run log store

        If exists_ok, we allow the run log to be already present in the run log store
        """
        try:
            attempt_run_log = self.run_log_store.get_run_log_by_id(run_id=self.run_id, full=False)
            if attempt_run_log.status in [defaults.FAIL, defaults.SUCCESS]:
                raise Exception(f'The run log by id: {self.run_id} already exists and is {attempt_run_log.status}')

            raise exceptions.RunLogExistsError(self.run_id)
        except exceptions.RunLogNotFoundError:
            pass
        except exceptions.RunLogExistsError:
            if exists_ok:
                return
            raise

        run_log = {}
        run_log['run_id'] = self.run_id
        run_log['tag'] = self.tag
        run_log['use_cached'] = False
        run_log['status'] = defaults.PROCESSING
        run_log['dag_hash'] = self.dag_hash

        parameters = {}
        if self.parameters_file:
            parameters.update(utils.load_yaml(self.parameters_file))

        # Update these with some from the environment variables
        parameters.update(utils.get_user_set_parameters())

        if self.previous_run_log:
            run_log['original_run_id'] = self.previous_run_log.run_id
            # Sync the previous run log catalog to this one.
            self.catalog_handler.sync_between_runs(previous_run_id=self.previous_run_log.run_id, run_id=self.run_id)
            run_log['use_cached'] = True
            parameters.update(self.previous_run_log.parameters)

        run_log = self.run_log_store.create_run_log(**run_log)
        # Any interaction with run log store attributes should happen via API if available.
        self.run_log_store.set_parameters(run_id=self.run_id, parameters=parameters)

        # Update run_config
        run_config = utils.get_run_config(self)
        self.run_log_store.set_run_config(run_id=self.run_id, run_config=run_config)

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

        self._set_up_run_log()

    def prepare_for_node_execution(self):
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

    def _sync_catalog(self, node: BaseNode, step_log: StepLog, stage: str, synced_catalogs=None):
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

        """
        if stage not in ['get', 'put']:
            msg = (
                'Catalog service only accepts get/put possible actions as part of node execution.'
                f'Sync catalog of the executor: {self.service_name} asks for {stage} which is not accepted'
            )
            raise Exception(msg)

        node_catalog_settings = node._get_catalog_settings()
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

    def _execute_node(self, node: BaseNode, map_variable: dict = None, **kwargs):
        """
        This is the entry point when we do the actual execution of the function.
        DO NOT Over-ride this function.

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
        max_attempts = node._get_max_attempts()
        attempts = 0
        step_log = self.run_log_store.get_step_log(node._get_step_log_name(map_variable), self.run_id)

        parameters = self.run_log_store.get_parameters(run_id=self.run_id)
        # Set up environment variables for the execution
        # If the key already exists, do not update it to give priority to parameters set by environment variables
        interaction.store_parameter(update=False, **parameters)

        data_catalogs_get = self._sync_catalog(node, step_log, stage='get')

        mock = step_log.mock
        logger.info(f'Trying to execute node: {node.internal_name}, attempt : {attempts}, max_attempts: {max_attempts}')
        while attempts < max_attempts:
            try:
                self.context_step_log = step_log
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
            finally:
                self.context_step_log = None  # type: ignore

            if attempts == max_attempts:
                step_log.status = defaults.FAIL
                logger.error(f'Node {node} failed, max retries of {max_attempts} reached')

        self._sync_catalog(node, step_log, stage='put', synced_catalogs=data_catalogs_get)
        self.run_log_store.add_step_log(step_log, self.run_id)

    def execute_node(self, node: BaseNode, map_variable: dict = None, **kwargs):
        raise NotImplementedError

    def add_code_identities(self, node: BaseNode, step_log: StepLog, **kwargs):
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
            * task : We can delegate to _execute_node after checking the eligibility for re-run in cases of a re-run
            * success: We can delegate to _execute_node
            * fail: We can delegate to _execute_node

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
        step_log = self.run_log_store.create_step_log(node.name, node._get_step_log_name(map_variable))

        self.add_code_identities(node=node, step_log=step_log)

        step_log.step_type = node.node_type
        step_log.status = defaults.PROCESSING

        # Add the step log to the database as per the situation.
        # If its a terminal node, complete it now
        if node.node_type in ['success', 'fail']:
            self.run_log_store.add_step_log(step_log, self.run_id)
            self._execute_node(node, map_variable=map_variable, **kwargs)
            return

        # In single step
        if self.single_step:
            # If the node name does not match, we move on to the next node.
            if not node.name == self.single_step:
                step_log.mock = True
                step_log.status = defaults.SUCCESS
                self.run_log_store.add_step_log(step_log, self.run_id)
                return
        else:  # We are not in single step mode
            # If previous run was successful, move on to the next step
            if not self._is_eligible_for_rerun(node, map_variable=map_variable):
                step_log.mock = True
                step_log.status = defaults.SUCCESS
                self.run_log_store.add_step_log(step_log, self.run_id)
                return

        # We call an internal function to iterate the sub graphs and execute them
        if node.is_composite:
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

    def _get_status_and_next_node_name(self, current_node: BaseNode, dag: Graph, map_variable: dict = None):
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

        step_log = self.run_log_store.get_step_log(current_node._get_step_log_name(map_variable), self.run_id)
        logger.info(
            f'Finished executing the node {current_node} with status {step_log.status}')

        next_node_name = current_node._get_next_node()

        if step_log.status == defaults.FAIL:
            next_node_name = dag.get_fail_node().name
            if current_node._get_on_failure_node():
                next_node_name = current_node._get_on_failure_node()

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

            status, next_node_name = self._get_status_and_next_node_name(
                current_node=working_on, dag=dag, map_variable=map_variable)

            if status == defaults.TRIGGERED:
                # Some nodes go into triggered state and self traverse
                logger.info(f'Triggered the job to execute the node {current_node}')
                break

            if working_on.node_type in ['success', 'fail']:
                break

            current_node = next_node_name

        run_log = self.run_log_store.get_branch_log(working_on._get_branch_log_name(map_variable), self.run_id)

        branch = 'graph'
        if working_on.internal_branch_name:
            branch = working_on.internal_branch_name

        logger.info(f'Finished execution of the {branch} with status {run_log.status}')
        print(json.dumps(run_log.dict(), indent=4))

    def _is_eligible_for_rerun(self, node: BaseNode, map_variable: dict = None):
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
            node_step_log_name = node._get_step_log_name(map_variable=map_variable)
            logger.info(f'Scanning previous run logs for node logs of: {node_step_log_name}')

            previous_node_log = None
            try:
                previous_node_log, _ = self.previous_run_log.search_step_by_internal_name(node_step_log_name)
            except exceptions.StepLogNotFoundError:
                logger.warning(f'Did not find the node {node.name} in previous run log')
                return True  # We should re-run the node.

            step_log = self.run_log_store.get_step_log(node._get_step_log_name(map_variable), self.run_id)
            logger.info(f'The original step status: {previous_node_log.status}')

            if previous_node_log.status == defaults.SUCCESS:
                logger.info(f'The step {node.name} is marked success, not executing it')
                step_log.status = defaults.SUCCESS
                step_log.message = 'Node execution successful in previous run, skipping it'
                self.run_log_store.add_step_log(step_log, self.run_id)
                return False  # We need not run the node

            #  Remove previous run log to start execution from this step
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

    def _resolve_node_config(self, node: BaseNode):
        """
        The mode_config section can contain specific over-rides to an global executor config.
        To avoid too much clutter in the dag definition, we allow the configuration file to have placeholders block.
        The nodes can over-ride the global config by referring to key in the placeholder.

        For example:
        # configuration.yaml
        mode:
          type: cloud-implementation
          config:
            k1: v1
            k3: v3
            placeholders:
              k2: v2 # Could be a mapping internally.

        # in pipeline definition.yaml
        dag:
          steps:
            step1:
              mode_config:
                cloud-implementation:
                  k1: value_specific_to_node
                  k2:

        This method should resolve the node_config to {'k1': 'value_specific_to_node', 'k2': 'v2', 'k3': 'v3'}

        Args:
            node (BaseNode): The current node being processed.

        """
        effective_node_config = copy.deepcopy(self.config.dict())
        ctx_node_config = node._get_mode_config(self.service_name)

        placeholders = self.config.placeholders

        for key, value in ctx_node_config.items():
            if not value:
                if key in placeholders:  # Update via placeholder only if value is None
                    try:
                        effective_node_config.update(placeholders[key])
                    except TypeError:
                        logger.error(f'Expected value to the {key} to be a mapping but found {type(placeholders[key])}')
                    continue
                logger.info(f"For key: {key} in the {node.name} mode_config, there is no value provided and no \
                    corresponding placeholder was found")

            effective_node_config[key] = value

        effective_node_config.pop('placeholders', None)

        return effective_node_config

```

The custom extensions should be registered as part of the namespace: ```executor``` for it to be
loaded.

```toml
# For example, as part of your pyproject.toml
[tool.poetry.plugins."executor"]
"kubeflow" = "YOUR_PACKAGE:Kubeflow"
```
