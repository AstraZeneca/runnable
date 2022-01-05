# Extensions

To implement your own Run Log store, all you have to do is implement this base class.

The implemented methods work, if the run log store uses a single JSON file.

Please refer to the implementation of ```db``` run log for fragmented run log store.


```python
from magnus.datastore import RunLog, StepLog, StepAttempt, BranchLog, CodeIdentity, DataCatalog

class BaseRunLogStore:
    store_type = None

    def __init__(self, config):
        self.config = config
    
    def accomodate_executor(self, executor, stage='execution'):
        """
        Use this method to change any of the mode (executor) settings.

        Raise an execption if this service provider is not compatible with the compute provider. 

        This function would be called twice:

        * During the traversal of the graph with stage='traversal'.
        * During the execution of the node with stage='execution'.

        Most of the times, the method need not do anything and you can simply pass.

        Args:
            executor (magnus.executor.BaseExecutor): The compute mode 
            stage (str, optional): The stage at which the function is called. Defaults to 'execution'.
        """
        pass


    def create_run_log(self, run_id, **kwargs):
        """
        Creates a Run Log object by using the config

        Logically the method should do the following:
            * Creates a Run log
            * Adds it to the db
            * Return the log
        Raises:
            NotImplementedError: This is a base class and therefore has no default implementation
        """

        raise NotImplementedError

    def get_run_log_by_id(self, run_id, **kwargs):
        """
        Retrieves a Run log from the database using the config and the run_id

        Args:
            run_id (str): The run_id of the run

        Logically the method should:
            * Returns the run_log defined by id from the data store defined by the config

        Raises:
            NotImplementedError: This is a base class and therefore has no default implementation
            RunLogNotFoundError: If the run log for run_id is not found in the datastore
        """

        raise NotImplementedError

    def put_run_log(self, run_log, **kwargs):
        """
        Puts the Run Log in the database as defined by the config

        Args:
            run_log (RunLog): The Run log of the run

        Logically the method should:
            Puts the run_log into the database

        Raises:
            NotImplementedError: This is a base class and therefore has no default implementation
        """
        raise NotImplementedError

    def get_parameters(self, run_id, **kwargs):
        """
        Get the parameters from the Run log defined by the run_id

        Args:
            run_id (str): The run_id of the run

        The method should:
            * Call get_run_log_by_id(run_id) to retrieve the run_log
            * Return the parameters as identified in the run_log

        Returns:
            dict: A dictionary of the run_log parameters
        Raises:
            RunLogNotFoundError: If the run log for run_id is not found in the datastore
        """
        run_log = self.get_run_log_by_id(run_id=run_id)
        return run_log.parameters

    def set_parameters(self, run_id, parameters, **kwargs):
        """
        Update the parameters of the Run log with the new parameters

        This method would over-write the parameters, if the parameter exists in the run log already

        The method should:
            * Call get_run_log_by_id(run_id) to retrieve the run_log
            * Update the parameters of the run_log
            * Call put_run_log(run_log) to put the run_log in the datastore

        Args:
            run_id (str): The run_id of the run
            parameters (dict): The parameters to update in the run log
        Raises:
            RunLogNotFoundError: If the run log for run_id is not found in the datastore
        """
        run_log = self.get_run_log_by_id(run_id=run_id)
        run_log.parameters.update(parameters)
        self.put_run_log(run_log=run_log)

    def create_step_log(self, name, internal_name, **kwargs):
        """
        Create a step log by the name and internal name

        The method does not update the Run Log with the step log at this point in time.
        This method is just an interface for external modules to create a step log


        Args:
            name (str): The friendly name of the step log
            internal_name (str): The internal naming of the step log. The internal naming is a dot path convention

        Returns:
            StepLog: A uncommmited step log object
        """
        logger.info(f'{self.store_type} Creating a Step Log: {name}')
        return StepLog(name=name, internal_name=internal_name, status=defaults.CREATED)

    def get_step_log(self, internal_name, run_id, **kwargs):
        """
        Get a step log from the datastore for run_id and the internal naming of the step log

        The internal naming of the step log is a dot path convention.

        The method should:
            * Call get_run_log_by_id(run_id) to retrieve the run_log
            * Identify the step location by decoding the internal naming
            * Return the step log

        Args:
            internal_name (str): The internal name of the step log
            run_id (str): The run_id of the run

        Returns:
            StepLog: The step log object for the step defined by the internal naming and run_id

        Raises:
            RunLogNotFoundError: If the run log for run_id is not found in the datastore
            StepLogNotFoundError: If the step log for internal_name is not found in the datastore for run_id
        """
        logger.info(f'{self.store_type} Getting the step log: {internal_name} of {run_id}')
        run_log = self.get_run_log_by_id(run_id=run_id)
        step_log, _ = run_log.search_step_by_internal_name(internal_name)
        return step_log

    def add_step_log(self, step_log, run_id, **kwargs):
        """
        Add the step log in the run log as identifed by the run_id in the datastore

        The method should:
             * Call get_run_log_by_id(run_id) to retrieve the run_log
             * Identify the branch to add the step by decoding the step_logs internal name
             * Add the step log to the identified branch log
             * Call put_run_log(run_log) to put the run_log in the datastore

        Args:
            step_log (StepLog): The Step log to add to the database
            run_id (str): The run id of the run

        Raises:
            RunLogNotFoundError: If the run log for run_id is not found in the datastore
            BranchLogNotFoundError: If the branch of the step log for internal_name is not found in the datastore
                                    for run_id
        """
        logger.info(f'{self.store_type} Adding the step log to DB: {step_log.name}')
        run_log = self.get_run_log_by_id(run_id=run_id)

        branch_to_add = '.'.join(step_log.internal_name.split('.')[:-1])
        branch, _ = run_log.search_branch_by_internal_name(branch_to_add)

        if branch is None:
            branch = run_log
        branch.steps[step_log.internal_name] = step_log
        self.put_run_log(run_log=run_log)

    def create_branch_log(self, internal_branch_name, **kwargs):
        """
        Creates a uncommited branch log object by the internal name given

        Args:
            internal_branch_name ([type]): [description]

        Returns:
            [type]: [description]
        """
        # Create a new BranchLog
        logger.info(f'{self.store_type} Creating a Branch Log : {internal_branch_name}')
        return BranchLog(internal_branch_name, status=defaults.CREATED)

    def get_branch_log(self, internal_branch_name, run_id, **kwargs):
        # Should get the run_log
        # Should search for the branch log with the name
        run_log = self.get_run_log_by_id(run_id=run_id)
        if not internal_branch_name:
            return run_log
        branch, _ = run_log.search_branch_by_internal_name(internal_branch_name)
        return branch  # , branch_step

    def add_branch_log(self, branch_log, run_id, **kwargs):
        # Get the run log
        # Get the branch and step containining the branch
        # Add the branch to the step
        # Write the run_log
        internal_branch_name = None

        if hasattr(branch_log, 'internal_name'):
            internal_branch_name = branch_log.internal_name

        if not internal_branch_name:
            self.put_run_log(branch_log)  # We are dealing with base dag here
            return

        run_log = self.get_run_log_by_id(run_id=run_id)

        step_name = '.'.join(internal_branch_name.split('.')[:-1])
        step, _ = run_log.search_step_by_internal_name(step_name)

        step.branches[internal_branch_name] = branch_log
        self.put_run_log(run_log)

    def create_attempt_log(self, **kwargs):
        logger.info(f'{self.store_type} Creating an attempt log')
        return StepAttempt()

    def create_code_identity(self, **kwargs):
        logger.info(f'{self.store_type} Creating Code identity')
        return CodeIdentity()

    def create_data_catalog(self, name: str, **kwargs) -> DataCatalog:
        """
        Create a uncommitted data catalog object

        Args:
            name (str): The name of the data catalog item to put

        Returns:
            DataCatalog: The DataCatalog object. 
        """
        logger.info(f'{self.store_type} Creating Data Catalog for {name}')
        return DataCatalog(name=name)
```


Provide the path to the implementation in magnus config as described here.