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

    ```yaml linenums="1"
    --8<-- "examples/concepts/task_shell_parameters.yaml"
    ```


=== "Run log"

    ```json linenums="1"
    {
      "run_id": "devout-jones-0640",
      "dag_hash": "9070f0b9c661d4ff7a23647cbe0ed2d461b9a26e",
      "use_cached": false,
      "tag": "",
      "original_run_id": "",
      "status": "SUCCESS",
      "steps": {
          "access initial": {
              "name": "access initial",
              "internal_name": "access initial",
              "status": "SUCCESS",
              "step_type": "task",
              "message": "",
              "mock": false,
              "code_identities": [
                  {
                      "code_identifier": "ca4c5fbff4148d3862a4738942d4607a9c4f0d88",
                      "code_identifier_type": "git",
                      "code_identifier_dependable": true,
                      "code_identifier_url": "https://github.com/AstraZeneca/magnus-core.git",
                      "code_identifier_message": ""
                  }
              ],
              "attempts": [
                  {
                      "attempt_number": 1,
                      "start_time": "2023-12-30 06:40:55.188207",
                      "end_time": "2023-12-30 06:40:55.202317",
                      "duration": "0:00:00.014110",
                      "status": "SUCCESS",
                      "message": "",
                      "parameters": {
                          "spam": "Hello",
                          "eggs": {
                              "ham": "Yes, please!!"
                          }
                      }
                  }
              ],
              "user_defined_metrics": {},
              "branches": {},
              "data_catalog": [
                  {
                      "name": "access_initial.execution.log",
                      "data_hash": "8a18b647052b3c85020beb2024f2a25289fe955b1421026008521b12cff4f44c",
                      "catalog_relative_path": "devout-jones-0640/access_initial.execution.log",
                      "catalog_handler_location": ".catalog",
                      "stage": "put"
                  }
              ]
          },
          "modify initial": {
              "name": "modify initial",
              "internal_name": "modify initial",
              "status": "SUCCESS",
              "step_type": "task",
              "message": "",
              "mock": false,
              "code_identities": [
                  {
                      "code_identifier": "ca4c5fbff4148d3862a4738942d4607a9c4f0d88",
                      "code_identifier_type": "git",
                      "code_identifier_dependable": true,
                      "code_identifier_url": "https://github.com/AstraZeneca/magnus-core.git",
                      "code_identifier_message": ""
                  }
              ],
              "attempts": [
                  {
                      "attempt_number": 1,
                      "start_time": "2023-12-30 06:40:55.266858",
                      "end_time": "2023-12-30 06:40:55.281405",
                      "duration": "0:00:00.014547",
                      "status": "SUCCESS",
                      "message": "",
                      "parameters": {
                          "spam": "Hello",
                          "eggs": {
                              "ham": "Yes, please!!"
                          }
                      }
                  }
              ],
              "user_defined_metrics": {},
              "branches": {},
              "data_catalog": [
                  {
                      "name": "modify_initial.execution.log",
                      "data_hash": "9dea22c132992504146374f6ac7cfe2f5510da78ca3bb5cc576abcfde0a4da3c",
                      "catalog_relative_path": "devout-jones-0640/modify_initial.execution.log",
                      "catalog_handler_location": ".catalog",
                      "stage": "put"
                  }
              ]
          },
          "display again": {
              "name": "display again",
              "internal_name": "display again",
              "status": "SUCCESS",
              "step_type": "task",
              "message": "",
              "mock": false,
              "code_identities": [
                  {
                      "code_identifier": "ca4c5fbff4148d3862a4738942d4607a9c4f0d88",
                      "code_identifier_type": "git",
                      "code_identifier_dependable": true,
                      "code_identifier_url": "https://github.com/AstraZeneca/magnus-core.git",
                      "code_identifier_message": ""
                  }
              ],
              "attempts": [
                  {
                      "attempt_number": 1,
                      "start_time": "2023-12-30 06:40:55.354662",
                      "end_time": "2023-12-30 06:40:55.366113",
                      "duration": "0:00:00.011451",
                      "status": "SUCCESS",
                      "message": "",
                      "parameters": {
                          "spam": "World",
                          "eggs": {
                              "ham": "No, Thank you!!"
                          }
                      }
                  }
              ],
              "user_defined_metrics": {},
              "branches": {},
              "data_catalog": [
                  {
                      "name": "display_again.execution.log",
                      "data_hash": "9126727342ebef3d3635db294708ad96b49092bf3680da8f38490ea84844c8d4",
                      "catalog_relative_path": "devout-jones-0640/display_again.execution.log",
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
                      "code_identifier": "ca4c5fbff4148d3862a4738942d4607a9c4f0d88",
                      "code_identifier_type": "git",
                      "code_identifier_dependable": true,
                      "code_identifier_url": "https://github.com/AstraZeneca/magnus-core.git",
                      "code_identifier_message": ""
                  }
              ],
              "attempts": [
                  {
                      "attempt_number": 1,
                      "start_time": "2023-12-30 06:40:55.431271",
                      "end_time": "2023-12-30 06:40:55.431327",
                      "duration": "0:00:00.000056",
                      "status": "SUCCESS",
                      "message": "",
                      "parameters": {
                          "spam": "Universe",
                          "eggs": {
                              "ham": "Maybe, one more.."
                          }
                      }
                  }
              ],
              "user_defined_metrics": {},
              "branches": {},
              "data_catalog": []
          }
      },
      "parameters": {
          "spam": "Universe",
          "eggs": {
              "ham": "Maybe, one more.."
          }
      },
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
          "pipeline_file": "examples/concepts/task_shell_parameters.yaml",
          "parameters_file": "examples/concepts/parameters.yaml",
          "configuration_file": null,
          "tag": "",
          "run_id": "devout-jones-0640",
          "variables": {},
          "use_cached": false,
          "original_run_id": "",
          "dag": {
              "start_at": "access initial",
              "name": "",
              "description": "This is a sample pipeline to show the parameter flow for shell
              types.\n\nThe step \"access initial\" just displays the initial parameters
              defined in examples/concepts/parameters.yaml\nThe step modify_initial updates
              the parameters and sets them back as environment variables.\nThe step
              display_again displays the updated parameters from modify_initial and updates
              them.\n\n
              You can run this pipeline as:\n  magnus execute -f
              examples/concepts/task_shell_parameters.yaml  -p examples/concepts/parameters.
              yaml\n",
              "internal_branch_name": "",
              "steps": {
                  "access initial": {
                      "type": "task",
                      "name": "access initial",
                      "internal_name": "access initial",
                      "internal_branch_name": "",
                      "is_composite": false
                  },
                  "modify initial": {
                      "type": "task",
                      "name": "modify initial",
                      "internal_name": "modify initial",
                      "internal_branch_name": "",
                      "is_composite": false
                  },
                  "display again": {
                      "type": "task",
                      "name": "display again",
                      "internal_name": "display again",
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
          "dag_hash": "9070f0b9c661d4ff7a23647cbe0ed2d461b9a26e",
          "execution_plan": "chained"
      }
    }
    ```


In the above example of ```run log``` tab,

- ```run_id```: Defined in line #2, is a a unique id generated for every execution of the pipeline.
- ```use_cached```: in line #4, is the execution id of an older run that is being restarted in the current execution.
- ```tag```: A user defined label to be attached to an execution of the pipeline to contextually group executions.
This label can also be used to group experiments of experiment tracking tools like
[mlflow](https://mlflow.org/docs/latest/tracking/tracking-api.html#organizing-runs-in-experiments).
- ```status```: In line #7, defines the global status of the execution. ``` SUCCESS``, ```PROCESSING``` or ```FAILED``
are the three possible states.
- ```run_config```: From line #184 to end, capture the configuration used during the
execution. It details the configuration of different services (executor, catalog, secrets
handler etc) and also the pipeline definition. This is the internal representation of the
execution.


!!! tip

    The system generated ```run_id``` is always appended with the time of execution. Use this to distinguish
    between execution id's during rapid experimentation.

    In the above example, the ```run_id```, "affable-babbage-0545" is executed at 05:45.


## parameters

The final state of parameters are captured at the run log level while individual
[step logs](#step_log) show the parameters at the point of execution of the task.

In the above example, lines 178-183 show the final parameters at the end of execution.


## Step Log

The step log captures the information about the execution of the steps. It is mapping indexed by the name of the step
in the pipeline and is ordered chronologically by the start time of the execution of the step.

### Example

A snippet from the above example:

```json linenums="1"
"steps": {
  "access initial": {
      "name": "access initial",
      "internal_name": "access initial",
      "status": "SUCCESS",
      "step_type": "task",
      "message": "",
      "mock": false,
      "code_identities": [
          {
              "code_identifier": "ca4c5fbff4148d3862a4738942d4607a9c4f0d88",
              "code_identifier_type": "git",
              "code_identifier_dependable": true,
              "code_identifier_url": "https://github.com/AstraZeneca/magnus-core.git",
              "code_identifier_message": ""
          }
      ],
      "attempts": [
          {
              "attempt_number": 1,
              "start_time": "2023-12-30 06:40:55.188207",
              "end_time": "2023-12-30 06:40:55.202317",
              "duration": "0:00:00.014110",
              "status": "SUCCESS",
              "message": "",
              "parameters": {
                  "spam": "Hello",
                  "eggs": {
                      "ham": "Yes, please!!"
                  }
              }
          }
      ],
      "user_defined_metrics": {},
      "branches": {},
      "data_catalog": [
          {
              "name": "access_initial.execution.log",
              "data_hash": "8a18b647052b3c85020beb2024f2a25289fe955b1421026008521b12cff4f44c",
              "catalog_relative_path": "devout-jones-0640/access_initial.execution.log",
              "catalog_handler_location": ".catalog",
              "stage": "put"
          }
      ]
  },
  ...

```

- For non-nested steps, the key is the name of the step. For example, the first entry
in the steps mapping is "access initial" which corresponds to the name of the task in
the pipeline. For nested steps, the step log is also nested and shown in more detail for
  [parallel](../parallel), [map](../map) and [dag](../dag).

- ```status```: In line #5 is the status of the step with three possible states, "SUCCESS", "FAILURE" or "PROCESSING".
- ```step_type```: In line #6, is the type of step, in this case is a ```task```.
- ```message```: in line #7, is a short description of the error if the step failed.
This might not always be useful as a step can fail for many complicate reasons.
- ```code_identities```: We capture the unique identifier of the state of the code for
reproducibility purposes.

    * The ```code_identifier``` is the git sha of the code.
    * ```code_identifier_dependable``` indicates if the current branch is clean. Unclean branches makes it hard to
    determine the exact state of the code.
    * ```code_identifier_message```: Captures the names of the files which have uncommitted changes.


  It is easy to extend and customize the metrics being captured here. For example, executors like ```local-container```
  or ```argo``` can add the docker container identities as part of the log.

- ```attempts```: In line #19-34, Is the ordered list of attempts to execute the step. It shows the start time,
end time, duration of the execution and the parameters at the time of execution of the step.

  For example, at the time of executing the step ```access initial```, the parameters are the
  ```json
  "spam": "Hello",
  "eggs": {
      "ham": "Yes, please!!"
  }
  ```
  while for the step, ```display again``` shows the modified parameters:
  ```json
  "spam": "World",
  "eggs": {
      "ham": "No, Thank you!!"
  }
  ```

- ```user_defined_metrics```: are any [experiment tracking metrics](../task/#experiment_tracking)
captured during the execution of the step.

- ```branches```: This only applies to parallel, map or dag steps and shows the logs captured during the
execution of the branch.
- ```data_catalog```: Captures any data flowing through the tasks by the [catalog](../catalog).
By default, the execution logs of the task are put in the catalog for easier debugging purposes.

For example,  the below lines from the snippet specifies one entry into the catalog which is the execution log
of the task ```access initial``` and also the hash of the data.

```json
"data_catalog": [
    {
        "name": "access_initial.execution.log",
        "data_hash": "8a18b647052b3c85020beb2024f2a25289fe955b1421026008521b12cff4f44c",
        "catalog_relative_path": "devout-jones-0640/access_initial.execution.log",
        "catalog_handler_location": ".catalog",
        "stage": "put"
    }
]
```


## Retrying failures


## API

Tasks can access the ```run log``` during the execution of the step
[using the API](../../interactions/#magnus.get_run_log). The run log returned by this method is a deep copy
to prevent any modifications.


Tasks can also access the ```run_id``` of the current execution either by
[using the API](../../interactions/#magnus.get_run_id) or by the environment
variable ```MAGNUS_RUN_ID```.
