# Run Log

In magnus, run log captures all the information required to accurately describe a run. It should not
confused with your application logs, which are project dependent. Independent of the providers of any systems
(compute, secrets, run log, catalog), the structure of the run log would remain the same and should enable
you to compare between runs.

To accurately recreate an older run either for debugging purposes or for reproducibility, it should capture all
the variables of the system and their state during the run. For the purple of data science applications,
it boils down to:

- Data: The source of the data and the version of it.
- Code: The code used to run the the experiment and the version of it.
- Environment: The environment the code ran in with all the system installations.
- Configuration: The pipeline definition and the configuration.

The Run Log helps in storing them systematically for every run with the best possible information on all of the above.

## Structure of Run Log

A typical run log has the following structure, with a few definitions given inline.

```json
{
    "run_id": ,
    "dag_hash": , # The SHA id of the dag definition
    "use_cached": , # True for a re-run, False otherwise
    "tag": , # A friendly name given to a group of runs
    "original_run_id": , # The run id of the older run in case of a re-run
    "status": ,
    "steps": {},
    "parameters": {},
    "variables": {}
}
```

### run_id
Every run in magnus is given a unique ```run_id```.

Magnus creates one based on the timestamp is one is not provided during the run time.

During the execution of the run, the ```run_id``` can be obtained in the following ways:


```python
from magnus import get_run_id

def my_function():
    run_id = get_run_id() # Returns the run_id of the current run
```


or using environmental variable ```MAGNUS_RUN_ID```.

```python
import os

def my_function():
    run_id = os.environ['MAGNUS_RUN_ID'] # Returns the run_id of the current run
```




### dag_hash

The SHA id of the pipeline itself is stored here.

In the case of re-run, we check the newly run pipeline hash against the older run to ensure they are the same. You
can force to re-run too if you are aware of the differences.

### tag

A friendly name that could be used to group multiple runs together. You can ```group``` multiple runs by the tag to
compare and track the experiments done in the group.

### status

A flag to denote the status of the run. The status could be:

- success : If the graph or sub-graph succeeded, i.e reached the success node.
- fail: If the graph or sub-graph reached the fail node. Please note that a  failure of a node does not imply failure of
    the graph as you can configure conditional traversal of the nodes.
- processing: A temporary status if any of the nodes are currently being processed.
- triggered: A temporary status if any of the nodes triggered a remote job (in cloud, for example).

### parameters

A dictionary of key-value pairs available to all the nodes.

Any ```kwargs``` present in the function signature, called as part of the pipeline, are resolved against this
dictionary and the values are set during runtime.

### steps

steps is a dictionary containing step log for every individual step of the pipeline. The structure of step log is
described below.

## Structure of Step Log

Every step of the dag have a corresponding step log.

The general structure follows, with a few explanations given inline.

```json
"step name": {
    "name": , # The name of the step as given in the dag definition
    "internal_name": , # The name of the step log in dot path convention
    "status": ,
    "step_type": , # The type of step as per the dag definition
    "message": , # Any message added to step by the run
    "mock": , # Is True if the step was skipped in case of a re-run
    "code_identities": [
    ],
    "attempts": [
    ],
    "user_defined_metrics": {
    },
    "branches": {},
    "data_catalog": []
}
```

### Naming Step Log
The name of the step log follows a convention, we refer, to as *dot path* convention.

All the steps of the parent dag have the same exact name as the step name provided in the dag.

The naming of the steps of the nested branches like parallel, map or dag are given below.
#### parallel step

The steps of the parallel branch follow parent_step.branch_name.child_step name.

<details>
<summary>Example</summary>

The step log names are given in-line for ease of reading.
```yaml
dag:
  start_at: Simple Step
  steps:
    Simple Step: # dot path name: Simple Step
      type: as-is
      next: Parallel
    Parallel: # dot path name: Parallel
      type: parallel
      next: Success
      branches:
        Branch A:
          start_at: Child Step A
          steps:
            Child Step A: # dot path name: Parallel.Branch A.Child Step A
              type: as-is
              next: Success
            Success: # dot path name: Parallel.Branch A.Success
              type: success
            Fail: # dot path name: Parallel.Branch A.Fail
              type: fail
        Branch B:
          start_at: Child Step B
          steps:
            Child Step B: # dot path name: Parallel.Branch B. Child Step B
              type: as-is
              next: Success
            Success: # dot path name: Parallel.Branch B.Success
              type: success
            Fail:  # dot path name: Parallel.Branch B.Fail
              type: fail
    Success: # dot path name: Success
      type: success
    Fail: # dot path name: Fail
      type: fail
```

</details>
#### dag step

The steps of the dag branch follow parent_step.branch.child_step_name.
Here *branch* is a special name given to keep the naming always consistent.

<details>
<summary>Example</summary>

The step log names are given in-line for ease of reading.
```yaml
dag:
  start_at: Simple Step
  steps:
    Simple Step: # dot path name: Simple Step
      type: as-is
      next: Dag
    Dag: # dot path name: Dag
      type: dag
      next: Success
      branch:
        steps:
          Child Step: # dot path name: Dag.branch.Child Step
            type: as-is
            next: Success
          Success: # dot path name: Dag.branch.Success
            type: success
          Fail: # dot path name: Dag.branch.Fail
            type: fail
    Success: # dot path name: Success
      type: success
    Fail: # dot path name: Fail
      type: fail
```

</details>

#### map step

The steps of the map branch follow parent_step.{value of iter_variable}.child_step_name.

<details>
<summary>Example</summary>

```yaml
dag:
  start_at: Simple Step
  steps:
    Simple Step: # dot path name: Simple Step
      type: as-is
      next: Map
    Map: # dot path name: Map
      type: map
      iterate_on: y
      next: Success
      branch:
        steps:
          Child Step:
            type: as-is
            next: Success
          Success:
            type: success
          Fail:
            type: fail
    Success: # dot path name: Success
      type: success
    Fail: # dot path name: Fail
      type: fail
```

If the value of parameter y turns out to be ['A', 'B'], the step log naming convention would by dynamic and have
Map.A.Child Step, Map.A.Success, Map.A.Fail and Map.B.Child Step, Map.B.Success, Map.B.Fail

</details>


### status

A flag to denote the status of the step. The status could be:

- success : If the step succeeded.
- fail: If the step failed.
- processing: A temporary status if current step is being processed.


### code identity

As part of the log, magnus captures any possible identification of the state of the code and environment.

This section is only present for *Execution* nodes.

An example code identity if the code is git controlled

```json
"code_identities": [
    {
        "code_identifier": "1486bd7fbe27d57ff4a9612e8dabe6a914bc4eb5", # Git commit id
        "code_identifier_type": "git", # Git
        "code_identifier_dependable": true, # A flag to track if git tree is clean
        "code_identifier_url": "ssh://git@##################.git", # The remote URL of the repo
        "code_identifier_message": "" # Lists all the files that were found to be unclean as per git
    }
]
```

If the execution was in a container, we also track the docker identity.
For example:

```json
"code_identities": [
    {
        "code_identifier": "1486bd7fbe27d57ff4a9612e8dabe6a914bc4eb5", # Git commit id
        "code_identifier_type": "git", # Git
        "code_identifier_dependable": true, # A flag to track if git tree is clean
        "code_identifier_url": "ssh://git@##################.git", # The remote URL of the repo
        "code_identifier_message": "" # Lists all the files that were found to be unclean as per git
    },
    {
        "code_identifier": "", # Docker image digest
        "code_identifier_type": "docker", # Git
        "code_identifier_dependable": true, # Always true as docker image id is dependable
        "code_identifier_url": "", # The docker registry URL
        "code_identifier_message": ""
    }
]
```

### attempts

An attempt log capturing meta data about the attempt made to execute the node.
This section is only present for *Execution* nodes.

The structure of attempt log along with inline definitions

```json
"attempts": [
    {
        "attempt_number": 0, # The sequence number of attempt.
        "start_time": "", # The start time of the attempt
        "end_time": "", # The end time of the attempt
        "duration": null, # The duration of the time taken for the command to execute
        "status": "",
        "parameters": "", # The parameters at that point of execution.
        "message": "" # If any exception was raised, this field captures the message of the exception
    }
]
```

The status of an attempt log could be one of:

- success : If the attempt succeeded.
- fail: If the attempt failed.


### user defined metrics

As part of the execution, there is a provision to store metrics in the run log. These metrics would be stored in this
section of the log.

Example of storing metrics:
```python
# in my_module.py

from magnus import track_this

def my_cool_function():
    track_this(number_of_files=102, failed_for=10)
    track_this(number_of_incidents={'mean_value':2, 'variance':0.1})

```

If this function was executed as part of the pipeline, you should see the following in the run log

``` json
{
    ...
    "steps": {
        "step name": {
            ...,
            "number_of_incidents": {
                "mean_value": 2,
                "variance" : 0.1
                },
                "number_of_files": 102,
                "failed_for": 10
            },
            ...
            "
        },
        ...
    }
}
```

The same could also be acheived without ```import magnus``` by exporting environment variables with prefix of
```MAGNUS_TRACK_```

```python
# in my_module.py

import os
import json

def my_cool_function():
    os.environ['MAGNUS_TRACK_' + 'number_of_files'] = 102
    os.environ['MAGNUS_TRACK_' + 'failed_for'] = 10

    os.environ['MAGNUS_TRACK_' + 'number_of_incidents'] = json.dumps({'mean_value':2, 'variance':0.1})

```


### branches

If the step was a composite node of type dag or parallel or map, this section is used to store the logs of the branch
which have a structure similar to the Run Log.


### data catalog

Data generated as part of individual steps of the pipeline can use the catalog to make the data available for the
downstream steps or for reproducibility of the run. The catalog metadata is stored here in this section.

The structure of the data catalog is as follows with inline definition.

```json
"data_catalog":
    [
        {
            "name": "", # The name of the file
            "stored_at": "", # The location at which it is stored
            "data_hash": "", # The SHA id of the data
            "stage": "" # The stage at which the data is cataloged.
        }
    ]
```

More information about cataloging is found [here](../catalog).


## Configuration

Configuration of a Run Log Store is as follows:

```yaml
run_log:
  type:
  config:
```

### type

The type of run log provider you want. This should be one of the run log types already available.

Buffered Run Log is provided as default if nothing is given.

### config

Any configuration parameters the run log provider accepts.

## Parameterized definition

As with any part of the magnus configuration, you can parameterize the configuration of Run Log to switch between
Run Log providers without changing the base definition.

Please follow the example provided [here](../dag/#parameterized_definition) for more information.



## Extensions

You can easily extend magnus to bring in your custom provider, if a default
implementation does not exist or you are not happy with the implementation.

To implement your own run log store, please extend the BaseRunLogStore whose definition is given below.


```python
# Code can be found in magnus/datastore.py

from pydantic import BaseModel

class BaseRunLogStore:
    """
    The base class of a Run Log Store with many common methods implemented.
    """
    service_name = ''

    class Config(BaseModel):
        pass

    def __init__(self, config):
        config = config or {}
        self.config = self.Config(**config)

    def create_run_log(self, run_id: str, dag_hash: str = '', use_cached: bool = False,
                       tag: str = '', original_run_id: str = '', status: str = defaults.CREATED, **kwargs):
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

    def get_run_log_by_id(self, run_id: str, full: bool = True, **kwargs) -> RunLog:
        """
        Retrieves a Run log from the database using the config and the run_id

        Args:
            run_id (str): The run_id of the run
            full (bool): return the full run log store or only the RunLog object

        Returns:
            RunLog: The RunLog object identified by the run_id

        Logically the method should:
            * Returns the run_log defined by id from the data store defined by the config

        Raises:
            NotImplementedError: This is a base class and therefore has no default implementation
            RunLogNotFoundError: If the run log for run_id is not found in the datastore
        """

        raise NotImplementedError

    def put_run_log(self, run_log: RunLog, **kwargs):
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

    def get_parameters(self, run_id: str, **kwargs) -> dict:
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

    def set_parameters(self, run_id: str, parameters: dict, **kwargs):
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

    def get_run_config(self, run_id: str, **kwargs) -> dict:
        """
        Given a run_id, return the run_config used to perform the run.

        Args:
            run_id (str): The run_id of the run

        Returns:
            dict: The run config used for the run
        """

        run_log = self.get_run_log_by_id(run_id=run_id)
        return run_log.run_config

    def set_run_config(self, run_id: str, run_config: dict, **kwargs):
        """ Set the run config used to run the run_id

        Args:
            run_id (str): The run_id of the run
            run_config (dict): The run_config of the run
        """

        run_log = self.get_run_log_by_id(run_id=run_id)
        run_log.run_config.update(run_config)
        self.put_run_log(run_log=run_log)

    def create_step_log(self, name: str, internal_name: str, **kwargs):
        """
        Create a step log by the name and internal name

        The method does not update the Run Log with the step log at this point in time.
        This method is just an interface for external modules to create a step log


        Args:
            name (str): The friendly name of the step log
            internal_name (str): The internal naming of the step log. The internal naming is a dot path convention

        Returns:
            StepLog: A uncommitted step log object
        """
        logger.info(f'{self.service_name} Creating a Step Log: {name}')
        return StepLog(name=name, internal_name=internal_name, status=defaults.CREATED)

    def get_step_log(self, internal_name: str, run_id: str, **kwargs) -> StepLog:
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
        logger.info(f'{self.service_name} Getting the step log: {internal_name} of {run_id}')
        run_log = self.get_run_log_by_id(run_id=run_id)
        step_log, _ = run_log.search_step_by_internal_name(internal_name)
        return step_log

    def add_step_log(self, step_log: StepLog, run_id: str, **kwargs):
        """
        Add the step log in the run log as identified by the run_id in the datastore

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
        logger.info(f'{self.service_name} Adding the step log to DB: {step_log.name}')
        run_log = self.get_run_log_by_id(run_id=run_id)

        branch_to_add = '.'.join(step_log.internal_name.split('.')[:-1])
        branch, _ = run_log.search_branch_by_internal_name(branch_to_add)

        if branch is None:
            branch = run_log
        branch.steps[step_log.internal_name] = step_log
        self.put_run_log(run_log=run_log)

    def create_branch_log(self, internal_branch_name: str, **kwargs) -> BranchLog:
        """
        Creates a uncommitted branch log object by the internal name given

        Args:
            internal_branch_name (str): Creates a branch log by name internal_branch_name

        Returns:
            BranchLog: Uncommitted and initialized with defaults BranchLog object
        """
        # Create a new BranchLog
        logger.info(f'{self.service_name} Creating a Branch Log : {internal_branch_name}')
        return BranchLog(internal_name=internal_branch_name, status=defaults.CREATED)

    def get_branch_log(self, internal_branch_name: str, run_id: str, **kwargs) -> Union[BranchLog, RunLog]:
        """
        Returns the branch log by the internal branch name for the run id

        If the internal branch name is none, returns the run log

        Args:
            internal_branch_name (str): The internal branch name to retrieve.
            run_id (str): The run id of interest

        Returns:
            BranchLog: The branch log or the run log as requested.
        """
        run_log = self.get_run_log_by_id(run_id=run_id)
        if not internal_branch_name:
            return run_log
        branch, _ = run_log.search_branch_by_internal_name(internal_branch_name)
        return branch

    def add_branch_log(self, branch_log: Union[BranchLog, RunLog], run_id: str, **kwargs):
        """
        The method should:
        # Get the run log
        # Get the branch and step containing the branch
        # Add the branch to the step
        # Write the run_log

        The branch log could some times be a Run log and should be handled appropriately

        Args:
            branch_log (BranchLog): The branch log/run log to add to the database
            run_id (str): The run id to which the branch/run log is added
        """

        internal_branch_name = None

        if isinstance(branch_log, BranchLog):
            internal_branch_name = branch_log.internal_name

        if not internal_branch_name:
            self.put_run_log(branch_log)  # type: ignore # We are dealing with base dag here
            return

        run_log = self.get_run_log_by_id(run_id=run_id)

        step_name = '.'.join(internal_branch_name.split('.')[:-1])
        step, _ = run_log.search_step_by_internal_name(step_name)

        step.branches[internal_branch_name] = branch_log  # type: ignore
        self.put_run_log(run_log)

    def create_attempt_log(self, **kwargs) -> StepAttempt:
        """
        Returns an uncommitted step attempt log.

        Returns:
            StepAttempt: An uncommitted step attempt log
        """
        logger.info(f'{self.service_name} Creating an attempt log')
        return StepAttempt()

    def create_code_identity(self, **kwargs) -> CodeIdentity:
        """
        Creates an uncommitted Code identity class

        Returns:
            CodeIdentity: An uncommitted code identity class
        """
        logger.info(f'{self.service_name} Creating Code identity')
        return CodeIdentity()

    def create_data_catalog(self, name: str, **kwargs) -> DataCatalog:
        """
        Create a uncommitted data catalog object

        Args:
            name (str): The name of the data catalog item to put

        Returns:
            DataCatalog: The DataCatalog object.
        """
        logger.info(f'{self.service_name} Creating Data Catalog for {name}')
        return DataCatalog(name=name)
```

The BaseRunLogStore depends upon a lot of other DataModels (pydantic datamodels) that capture and store the information.
These can all be found in ```magnus/datastore.py```. You can alternatively ignore all of them and create your own custom
implementation if desired but be aware of internal code dependencies on the structure of the datamodels.

The custom extensions should be registered as part of the namespace: ```magnus.datastore.BaseRunLogStore``` for it to be
loaded.

```toml
# For example, as part of your pyproject.toml
[tool.poetry.plugins."magnus.datastore.BaseRunLogStore"]
"mlmd" = "YOUR_PACKAGE:MLMDFormat"
```
