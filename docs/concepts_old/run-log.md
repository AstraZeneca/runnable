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

[Extensions are being actively developed and can be found here.](https://github.com/AstraZeneca/magnus-extensions)

To implement your own run log store, please extend the BaseRunLogStore whose definition is given below.


```python
# Code can be found in magnus/datastore.py
--8<-- "magnus/datastore.py:docs"
```

The BaseRunLogStore depends upon a lot of other DataModels (pydantic datamodels) that capture and store the information.
These can all be found in ```magnus/datastore.py```. You can alternatively ignore all of them and create your own custom
implementation if desired but be aware of internal code dependencies on the structure of the datamodels.

The custom extensions should be registered as part of the namespace: ```run_log_store``` for it to be
loaded.

```toml
# For example, as part of your pyproject.toml
[tool.poetry.plugins."run_log_store"]
"mlmd" = "YOUR_PACKAGE:MLMDFormat"
```
