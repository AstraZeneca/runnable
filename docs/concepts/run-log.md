# Run Log

Internally, magnus uses a ```run log``` to keep track of the execution of the pipeline. It
also stores the parameters, experiment tracking metrics and reproducibility information captured during the execution.

It should not be confused with application logs generated during the execution of a ```task``` i.e the stdout and stderr
when running the ```command``` of a task.

## Example

=== "pipeline"

    This is the same example [described in tasks](../task/#shell).

    tl;dr a pipeline that consumes some initial parameters and passes them
    to the next step. Both the steps are ```shell``` based tasks.

    ```yaml
    --8<-- "examples/concepts/task_shell_parameters.yaml"
    ```


=== "Run log"

    Please look into the inline annotations for more information about the different fields.

    ```json linenums="1"
    {
      "run_id": "affable-babbage-0545",
      "dag_hash": "",
      "use_cached": false,
      "tag": "",
      "original_run_id": "",
      "status": "SUCCESS",
      "steps": {
          "simple": {
              "name": "simple",
              "internal_name": "simple",
              "status": "SUCCESS",
              "step_type": "task",
              "message": "",
              "mock": false,
              "code_identities": [
                  {
                      "code_identifier": "2d1951a9126c213cf9db917a4f17a72b51557181",
                      "code_identifier_type": "git",
                      "code_identifier_dependable": true,
                      "code_identifier_url": "https://github.com/AstraZeneca/magnus-core.git",
                      "code_identifier_message": ""
                  }
              ],
              "attempts": [
                  {
                      "attempt_number": 1,
                      "start_time": "2023-12-29 05:45:26.785469",
                      "end_time": "2023-12-29 05:45:26.789541",
                      "duration": "0:00:00.004072",
                      "status": "SUCCESS",
                      "message": "",
                      "parameters": {}
                  }
              ],
              "user_defined_metrics": {},
              "branches": {},
              "data_catalog": [
                  {
                      "name": "simple.execution.log",
                      "data_hash": "03ba204e50d126e4674c005e04d82e84c21366780af1f43bd54a37816b6ab340",
                      "catalog_relative_path": "affable-babbage-0545/simple.execution.log",
                      "catalog_handler_location": ".catalog",
                      "stage": "put"
                  }
              ]
          },
          "success": {
              "name": "success",
              "internal_name": "success",
              "status": "SUCCESS",
              "step_type": "success",
              "message": "",
              "mock": false,
              "code_identities": [
                  {
                      "code_identifier": "2d1951a9126c213cf9db917a4f17a72b51557181",
                      "code_identifier_type": "git",
                      "code_identifier_dependable": true,
                      "code_identifier_url": "https://github.com/AstraZeneca/magnus-core.git",
                      "code_identifier_message": ""
                  }
              ],
              "attempts": [
                  {
                      "attempt_number": 1,
                      "start_time": "2023-12-29 05:45:26.854451",
                      "end_time": "2023-12-29 05:45:26.854503",
                      "duration": "0:00:00.000052",
                      "status": "SUCCESS",
                      "message": "",
                      "parameters": {}
                  }
              ],
              "user_defined_metrics": {},
              "branches": {},
              "data_catalog": []
          }
      },
      "parameters": {},
      "run_config": {
          "executor": {
              "service_name": "local",
              "service_type": "executor",
              "enable_parallel": false,
              "placeholders": {}
          },
          "run_log_store": {
              "service_name": "buffered",
              "service_type": "run_log_store"
          },
          "secrets_handler": {
              "service_name": "do-nothing",
              "service_type": "secrets"
          },
          "catalog_handler": {
              "service_name": "file-system",
              "service_type": "catalog",
              "compute_data_folder": "data"
          },
          "experiment_tracker": {
              "service_name": "do-nothing",
              "service_type": "experiment_tracker"
          },
          "pipeline_file": "",
          "parameters_file": "",
          "configuration_file": "",
          "tag": "",
          "run_id": "affable-babbage-0545",
          "variables": {},
          "use_cached": false,
          "original_run_id": "",
          "dag": {
              "start_at": "simple",
              "name": "",
              "description": "",
              "internal_branch_name": "",
              "steps": {
                  "simple": {
                      "type": "task",
                      "name": "simple",
                      "internal_name": "simple",
                      "internal_branch_name": "",
                      "is_composite": false
                  },
                  "success": {
                      "type": "success",
                      "name": "success",
                      "internal_name": "success",
                      "internal_branch_name": "",
                      "is_composite": false
                  },
                  "fail": {
                      "type": "fail",
                      "name": "fail",
                      "internal_name": "fail",
                      "internal_branch_name": "",
                      "is_composite": false
                  }
              }
          },
          "dag_hash": "",
          "execution_plan": "chained"
      }
    }
    ```


In the above example,

- ```run_id```: Defined in line #2, is a a unique id generated for every execution of the pipeline.
- ```use_cached```: in line #4, is the execution id of an older run that is being restarted in the current execution.
- ```tag```: A user defined label to be attached to an execution of the pipeline to contextually group executions.
This label can also be used to group experiments of experiment tracking tools like
[mlflow](https://mlflow.org/docs/latest/tracking/tracking-api.html#organizing-runs-in-experiments).
- ```status```: In line #7, defines the global status of the execution. ``` SUCCESS``, ```PROCESSING``` or ```FAILED``
are the three possible states.
- ```steps```: From lines 8- 79, capture the logs of individual steps of the execution. It is a mapping of
name of the step to a [step og](#step_log)
- ```parameters```: In line #80, are the final state of parameters used during the execution.
- ```run_config```: From line #81 to end, capture the configuration used during the execution. It details the
configuration of different services (executor, catalog, secrets handler etc) and also the pipeline definition. This
is the internal representation of the execution.


!!! tip

    The system generated ```run_id``` is always appended with the time of execution. Use this to distinguish
    between execution id's during rapid experimentation.

    In the above example, the ```run_id```, "affable-babbage-0545" is executed at 05:45.


## Step Log

The step log captures the information about the execution of the steps. It is mapping indexed by the name of the step
in the pipeline and is ordered chronologically by the start time of the execution of the step.

### Example

A snippet from the above example:

```json linenums="1"
"steps": {
  "simple": {
      "name": "simple",
      "internal_name": "simple",
      "status": "SUCCESS",
      "step_type": "task",
      "message": "",
      "mock": false,
      "code_identities": [
          {
              "code_identifier": "2d1951a9126c213cf9db917a4f17a72b51557181",
              "code_identifier_type": "git",
              "code_identifier_dependable": true,
              "code_identifier_url": "https://github.com/AstraZeneca/magnus-core.git",
              "code_identifier_message": ""
          }
      ],
      "attempts": [
          {
              "attempt_number": 1,
              "start_time": "2023-12-29 05:45:26.785469",
              "end_time": "2023-12-29 05:45:26.789541",
              "duration": "0:00:00.004072",
              "status": "SUCCESS",
              "message": "",
              "parameters": {}
          }
      ],
      "user_defined_metrics": {},
      "branches": {},
      "data_catalog": [
          {
              "name": "simple.execution.log",
              "data_hash": "03ba204e50d126e4674c005e04d82e84c21366780af1f43bd54a37816b6ab340",
              "catalog_relative_path": "affable-babbage-0545/simple.execution.log",
              "catalog_handler_location": ".catalog",
              "stage": "put"
          }
      ]
  },
  ...

```

- The key is the "name" of the step in steps dictionary.
- ```status```: In line #5 is the status of the step with three possible states, "SUCCESS", "FAILURE" or "PROCESSING".
- ```step_type```: In line #6, is the type of step.



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
