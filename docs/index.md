:runner: Orchestrate python functions, notebooks or scripts on your local machine by just adding
*one file*.

:runner: Move to any cloud by adding *one more file*.

## functions

The below content is assumed to be ```examples/functions.py```

!!! note inline end "pydantic models"

    The functions should use pydantic models as their outputs.

    Pydantic models offer better representations of the input and output, inspired by
    [FastAPI's implementation](https://fastapi.tiangolo.com/features/#pydantic-features).

```python linenums="1"
--8<-- "examples/functions.py"
```


There is nothing special about the functions, they are *plain old python functions*.

!!! tip "Notebooks and Shell scripts"

    You can execute notebooks and shell scripts too!!

    They can be written just as you would want them, *plain old notebooks and scripts*.



## local :runner:


Replace the "driver" function with a *runnable* definition in either ```python sdk```
or ```yaml```.


=== "yaml"

    !!! note inline end "pipeline and steps"

        The pipeline is essentially a representation of the "driver" function.

        The gains by this definition for local executions are clearer by the metadata gathered
        during the exeuction.

    ``` yaml linenums="1"
    --8<-- "examples/python-tasks.yaml"
    ```

    1. Start the pipeline execution at step1
    2. The name of the step.
    3. The path to the python function
    4. Go to step2, if successful
    5. Go to success node, if successful
    6. Mark the execution as success



=== "python"

    !!! note inline end "pipeline and steps"

        The pipeline is essentially a representation of the "driver" function.

        The gains by this definition for local executions are clearer by the metadata gathered
        during the exeuction.

    ```python linenums="1"
    --8<-- "examples/python-tasks.py"
    ```

    1. The name of the step.
    2. The path to the python function
    3. ```terminate_with_success``` indicates that the pipeline is completed successfully. You can also use ```terminate_with_failure``` to indicate the pipeline fail.
    4. There are many ways to define dependencies within nodes, step1 >> step2, step1 << step2 or using depends_on.
    5. Start the pipeline execution at step1
    6. The list of steps to be executed, the order does not matter.
    7. Add ```success``` and ```fail``` nodes to the pipeline.
    8. Returns the metadata captured during the execution.

=== "metadata"

    #### TODO: Change this

    Captures information to understand the execution plan for debugging or
    lineage purposes.

    ```json linenums="1"
    {
      "run_id": "piquant-pasteur-0613", // Unique run identifier
      "dag_hash": "",
      "use_cached": false,
      "tag": "",
      "original_run_id": "",
      "status": "SUCCESS",
      "steps": {
          "step1": {
              "name": "step1", // name of the step
              "internal_name": "step1",
              "status": "SUCCESS",
              "step_type": "task",
              "message": "",
              "mock": false,
              "code_identities": [ // The code status at the time of execution of step1
                  {
                      "code_identifier": "f68561360eed64e2715929d2ddd0736fd277d706",
                      "code_identifier_type": "git",
                      "code_identifier_dependable": true,
                      "code_identifier_url": "https://github.com/vijayvammi/runnable.git",
                      "code_identifier_message": ""
                  }
              ],
              "attempts": [
                  {
                      "attempt_number": 1,
                      "start_time": "2024-02-25 06:13:53.295595",
                      "end_time": "2024-02-25 06:13:53.306082",
                      "duration": "0:00:00.010487",
                      "status": "SUCCESS",
                      "message": "",
                      "parameters": {}
                  }
              ],
              "user_defined_metrics": {},
              "branches": {},
              "data_catalog": [
                  {
                      "name": "step1.execution.log", // THe stdout and stderr of execution
                      "data_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                      "catalog_relative_path": "piquant-pasteur-0613/step1.execution.log",
                      "catalog_handler_location": ".catalog",
                      "stage": "put"
                  }
              ]
          },
          "step2": {
              "name": "step2",
              "internal_name": "step2",
              "status": "SUCCESS",
              "step_type": "task",
              "message": "",
              "mock": false,
              "code_identities": [
                  {
                      "code_identifier": "f68561360eed64e2715929d2ddd0736fd277d706",
                      "code_identifier_type": "git",
                      "code_identifier_dependable": true,
                      "code_identifier_url": "https://github.com/vijayvammi/runnable.git",
                      "code_identifier_message": ""
                  }
              ],
              "attempts": [
                  {
                      "attempt_number": 1,
                      "start_time": "2024-02-25 06:13:53.372941",
                      "end_time": "2024-02-25 06:13:53.378192",
                      "duration": "0:00:00.005251",
                      "status": "SUCCESS",
                      "message": "",
                      "parameters": { // The parameters at the time of execution
                          "x": 1,
                          "y": {
                              "foo": 10,
                              "bar": "hello world"
                          }
                      }
                  }
              ],
              "user_defined_metrics": {},
              "branches": {},
              "data_catalog": [
                  {
                      "name": "step2.execution.log",
                      "data_hash": "612160fd5e1d7d1f3d8b4db1a6e73de63f97ff4c5db616525f856d774a2837b4",
                      "catalog_relative_path": "piquant-pasteur-0613/step2.execution.log",
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
                      "code_identifier": "f68561360eed64e2715929d2ddd0736fd277d706",
                      "code_identifier_type": "git",
                      "code_identifier_dependable": true,
                      "code_identifier_url": "https://github.com/vijayvammi/runnable.git",
                      "code_identifier_message": ""
                  }
              ],
              "attempts": [
                  {
                      "attempt_number": 1,
                      "start_time": "2024-02-25 06:13:53.441232",
                      "end_time": "2024-02-25 06:13:53.441295",
                      "duration": "0:00:00.000063",
                      "status": "SUCCESS",
                      "message": "",
                      "parameters": {
                          "x": 1,
                          "y": {
                              "foo": 10,
                              "bar": "hello world"
                          }
                      }
                  }
              ],
              "user_defined_metrics": {},
              "branches": {},
              "data_catalog": []
          }
      },
      "parameters": { // THe final state of the parameters
          "x": 1,
          "y": {
              "foo": 10,
              "bar": "hello world"
          }
      },
      "run_config": { // The configuration of the execution
          "executor": {
              "service_name": "local",
              "service_type": "executor",
              "enable_parallel": false,
              "overrides": {}
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
              "catalog_location": ".catalog"
          },
          "experiment_tracker": {
              "service_name": "do-nothing",
              "service_type": "experiment_tracker"
          },
          "pipeline_file": "",
          "parameters_file": "",
          "configuration_file": "",
          "tag": "",
          "run_id": "piquant-pasteur-0613",
          "variables": {},
          "use_cached": false,
          "original_run_id": "",
          "dag": { // THe pipeline representation
              "start_at": "step1",
              "name": "",
              "description": "",
              "steps": {
                  "step1": {
                      "type": "task",
                      "name": "step1",
                      "next": "step2",
                      "on_failure": "",
                      "overrides": {},
                      "catalog": null,
                      "max_attempts": 1,
                      "command": "examples.functions.return_parameter",
                      "command_type": "python",
                      "node_name": "step1"
                  },
                  "step2": {
                      "type": "task",
                      "name": "step2",
                      "next": "success",
                      "on_failure": "",
                      "overrides": {},
                      "catalog": null,
                      "max_attempts": 1,
                      "command": "examples.functions.display_parameter",
                      "command_type": "python",
                      "node_name": "step2"
                  },
                  "success": {
                      "type": "success",
                      "name": "success"
                  },
                  "fail": {
                      "type": "fail",
                      "name": "fail"
                  }
              }
          },
          "dag_hash": "",
          "execution_plan": "chained"
      }
    }
    ```

:sparkles: Thats it!! :sparkles:

By adding *one file* you created a pipeline. Your application code
did not change at all.

There is no boilerplate code, no adherence to structure, no intrusion into the
application code.

## cloud :runner:
