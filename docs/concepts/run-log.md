# Run Log

Internally, runnable uses a ```run log``` to keep track of the execution of the pipeline. It
also stores the parameters, experiment tracking metrics and reproducibility information captured during the execution.

It should not be confused with application logs generated during the execution of a ```task``` i.e the stdout and stderr
when running the ```command``` of a task.

## Example

=== "pipeline"

    This is the same example [described in tasks](../concepts/task.md/#shell).

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
                      "code_identifier_url": "https://github.com/AstraZeneca/runnable-core.git",
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
                      "code_identifier_url": "https://github.com/AstraZeneca/runnable-core.git",
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
                      "code_identifier_url": "https://github.com/AstraZeneca/runnable-core.git",
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
                      "code_identifier_url": "https://github.com/AstraZeneca/runnable-core.git",
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
              You can run this pipeline as:\n  runnable execute -f
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
- ```status```: In line #7, defines the global status of the execution. ```SUCCESS```, ```PROCESSING``` or ```FAILED```
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
              "code_identifier_url": "https://github.com/AstraZeneca/runnable-core.git",
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
  [parallel](../concepts/parallel.md), [map](../concepts/map.md).

- ```status```: In line #5 is the status of the step with three possible states,
```SUCCESS```, ```PROCESSING``` or ```FAILED```
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

- ```user_defined_metrics```: are any [experiment tracking metrics](../concepts/task.md/#experiment_tracking)
captured during the execution of the step.

- ```branches```: This only applies to parallel, map or dag steps and shows the logs captured during the
execution of the branch.
- ```data_catalog```: Captures any data flowing through the tasks by the [catalog](../concepts/catalog.md).
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

The structure of the run log remains the same independent of the ```executor``` used to execute.
This enables to debug failures during the execution in complex environments to be easily
reproduced in local environments and fixed.

!!! note "Shortcomings"

    Currently, the support is only available for

    - non-nested, linear pipelines
    - non-chunked run log store

    [mocked executor](../configurations/executors/mocked.md) provides better support in debugging failures.


### Example

=== "Argo configuration"

    The configuration file is assumed to be located at: ```examples/configs/argo-config-catalog.yaml```

    ```yaml linenums="1"
    --8<-- "examples/configs/argo-config-catalog.yaml"
    ```

=== "Faulty pipeline"

    To run the pipeline in argo, change the configuration file from
    ```examples/configs/fs-catalog-run_log.yaml``` to
    ```examples/configs/argo-config-catalog.yaml```

    ```yaml linenums="1"
    --8<-- "examples/retry-fail.yaml"
    ```

=== "Run log in Argo"

    ```json linenums="1"
    {
        "run_id": "toFail",
        "dag_hash": "13f7c1b29ebb07ce058305253171ceae504e1683",
        "use_cached": false,
        "tag": "",
        "original_run_id": "",
        "status": "PROCESSING",
        "steps": {
            "Setup": {
                "name": "Setup",
                "internal_name": "Setup",
                "status": "SUCCESS",
                "step_type": "task",
                "message": "",
                "mock": false,
                "code_identities": [
                    {
                        "code_identifier": "f94e49a4fcecebac4d5eecbb5b691561b08e45c0",
                        "code_identifier_type": "git",
                        "code_identifier_dependable": true,
                        "code_identifier_url": "https://github.com/AstraZeneca/runnable-core.git",
                        "code_identifier_message": ""
                    }
                ],
                "attempts": [
                    {
                        "attempt_number": 1,
                        "start_time": "2024-02-05 22:11:47.213714",
                        "end_time": "2024-02-05 22:11:47.290352",
                        "duration": "0:00:00.076638",
                        "status": "SUCCESS",
                        "message": "",
                        "parameters": {}
                    }
                ],
                "user_defined_metrics": {},
                "branches": {},
                "data_catalog": [
                    {
                        "name": "Setup.execution.log",
                        "data_hash": "b709b710424701bd86be1cca36c5ec18f412b6dbb8d4e7729ec10e44319adbaf",
                        "catalog_relative_path": "toFail/Setup.execution.log",
                        "catalog_handler_location": "/mnt/catalog",
                        "stage": "put"
                    }
                ]
            },
            "Create Content": {
                "name": "Create Content",
                "internal_name": "Create Content",
                "status": "SUCCESS",
                "step_type": "task",
                "message": "",
                "mock": false,
                "code_identities": [
                    {
                        "code_identifier": "f94e49a4fcecebac4d5eecbb5b691561b08e45c0",
                        "code_identifier_type": "git",
                        "code_identifier_dependable": true,
                        "code_identifier_url": "https://github.com/AstraZeneca/runnable-core.git",
                        "code_identifier_message": ""
                    }
                ],
                "attempts": [
                    {
                        "attempt_number": 1,
                        "start_time": "2024-02-05 22:12:14.210011",
                        "end_time": "2024-02-05 22:12:14.225645",
                        "duration": "0:00:00.015634",
                        "status": "SUCCESS",
                        "message": "",
                        "parameters": {}
                    }
                ],
                "user_defined_metrics": {},
                "branches": {},
                "data_catalog": [
                    {
                        "name": "Create_Content.execution.log",
                        "data_hash": "618e515729e00c7811865306b41e91d698c00577078e75b2e4bcf87ec9669d62",
                        "catalog_relative_path": "toFail/Create_Content.execution.log",
                        "catalog_handler_location": "/mnt/catalog",
                        "stage": "put"
                    },
                    {
                        "name": "data/hello.txt",
                        "data_hash": "949a4f1afcea77b4b3f483ebe993e733122fb87b7539a3fc3d6752030be6ad44",
                        "catalog_relative_path": "toFail/data/hello.txt",
                        "catalog_handler_location": "/mnt/catalog",
                        "stage": "put"
                    }
                ]
            },
            "Retrieve Content": {
                "name": "Retrieve Content",
                "internal_name": "Retrieve Content",
                "status": "FAIL",
                "step_type": "task",
                "message": "",
                "mock": false,
                "code_identities": [
                    {
                        "code_identifier": "f94e49a4fcecebac4d5eecbb5b691561b08e45c0",
                        "code_identifier_type": "git",
                        "code_identifier_dependable": true,
                        "code_identifier_url": "https://github.com/AstraZeneca/runnable-core.git",
                        "code_identifier_message": ""
                    }
                ],
                "attempts": [
                    {
                        "attempt_number": 1,
                        "start_time": "2024-02-05 22:12:36.514484",
                        "end_time": "2024-02-05 22:12:36.985694",
                        "duration": "0:00:00.471210",
                        "status": "FAIL",
                        "message": "Command failed",
                        "parameters": {}
                    }
                ],
                "user_defined_metrics": {},
                "branches": {},
                "data_catalog": [
                    {
                        "name": "data/hello.txt",
                        "data_hash": "949a4f1afcea77b4b3f483ebe993e733122fb87b7539a3fc3d6752030be6ad44",
                        "catalog_relative_path": "data/hello.txt",
                        "catalog_handler_location": "/mnt/catalog",
                        "stage": "get"
                    },
                    {
                        "name": "Retrieve_Content.execution.log",
                        "data_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                        "catalog_relative_path": "toFail/Retrieve_Content.execution.log",
                        "catalog_handler_location": "/mnt/catalog",
                        "stage": "put"
                    }
                ]
            }
        },
        "parameters": {},
        "run_config": {
            "executor": {
                "service_name": "argo",
                "service_type": "executor",
                "enable_parallel": false,
                "overrides": {},
                "image": "$argo_docker_image",
                "expose_parameters_as_inputs": true,
                "secrets_from_k8s": [],
                "output_file": "argo-pipeline.yaml",
                "name": "runnable-dag-",
                "annotations": {},
                "labels": {},
                "activeDeadlineSeconds": 172800,
                "nodeSelector": null,
                "parallelism": null,
                "retryStrategy": {
                    "limit": "0",
                    "retryPolicy": "Always",
                    "backoff": {
                        "duration": "120",
                        "factor": 2,
                        "maxDuration": "3600"
                    }
                },
                "max_step_duration_in_seconds": 7200,
                "tolerations": null,
                "image_pull_policy": "",
                "service_account_name": "default-editor",
                "persistent_volumes": [
                    {
                        "name": "runnable-volume",
                        "mount_path": "/mnt"
                    }
                ],
                "step_timeout": 14400
            },
            "run_log_store": {
                "service_name": "file-system",
                "service_type": "run_log_store",
                "log_folder": "/mnt/run_log_store"
            },
            "secrets_handler": {
                "service_name": "do-nothing",
                "service_type": "secrets"
            },
            "catalog_handler": {
                "service_name": "file-system",
                "service_type": "catalog",
                "catalog_location": "/mnt/catalog"
            },
            "experiment_tracker": {
                "service_name": "do-nothing",
                "service_type": "experiment_tracker"
            },
            "pipeline_file": "examples/retry-fail.yaml",
            "parameters_file": null,
            "configuration_file": "examples/configs/argo-config-catalog.yaml",
            "tag": "",
            "run_id": "toFail",
            "variables": {},
            "use_cached": false,
            "original_run_id": "",
            "dag": {
                "start_at": "Setup",
                "name": "",
                "description": "This is a simple pipeline that demonstrates retrying failures.\n\n1. Setup: We setup a data folder, we ignore if it is already present\n2. Create Content: We create a \"hello.txt\" and \"put\" the file in catalog\n3. Retrieve Content: We \"get\" the file \"hello.txt\" from the catalog and show the contents\n5. Cleanup: We remove the data folder. Note that this is stubbed to prevent accidental deletion.\n\n\nYou can run this pipeline by:\n   runnable execute -f examples/catalog.yaml -c examples/configs/fs-catalog.yaml\n",
                "steps": {
                    "Setup": {
                        "type": "task",
                        "name": "Setup",
                        "next": "Create Content",
                        "on_failure": "",
                        "overrides": {},
                        "catalog": null,
                        "max_attempts": 1,
                        "command_type": "shell",
                        "command": "mkdir -p data",
                        "node_name": "Setup"
                    },
                    "Create Content": {
                        "type": "task",
                        "name": "Create Content",
                        "next": "Retrieve Content",
                        "on_failure": "",
                        "overrides": {},
                        "catalog": {
                            "get": [],
                            "put": [
                                "data/hello.txt"
                            ]
                        },
                        "max_attempts": 1,
                        "command_type": "shell",
                        "command": "echo \"Hello from runnable\" >> data/hello.txt\n",
                        "node_name": "Create Content"
                    },
                    "Retrieve Content": {
                        "type": "task",
                        "name": "Retrieve Content",
                        "next": "success",
                        "on_failure": "",
                        "overrides": {},
                        "catalog": {
                            "get": [
                                "data/hello.txt"
                            ],
                            "put": []
                        },
                        "max_attempts": 1,
                        "command_type": "shell",
                        "command": "cat data/hello1.txt",
                        "node_name": "Retrieve Content"
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
            "dag_hash": "13f7c1b29ebb07ce058305253171ceae504e1683",
            "execution_plan": "chained"
        }
    }
    ```


=== "Fixed pipeline in local environment"

    Bring the run log from K8's volumes to local machine for a retry.

    ```yaml linenums="1"
    --8<-- "examples/retry-fixed.yaml"
    ```


=== "Run log in local"


    ```json linenums="1"
    {
        "run_id": "polynomial-bartik-2226",
        "dag_hash": "2beec08fd417134cd3b04599d6684469db4ad176",
        "use_cached": true,
        "tag": "",
        "original_run_id": "toFail",
        "status": "SUCCESS",
        "steps": {
            "Setup": {
                "name": "Setup",
                "internal_name": "Setup",
                "status": "SUCCESS",
                "step_type": "task",
                "message": "",
                "mock": true,
                "code_identities": [
                    {
                        "code_identifier": "f94e49a4fcecebac4d5eecbb5b691561b08e45c0",
                        "code_identifier_type": "git",
                        "code_identifier_dependable": true,
                        "code_identifier_url": "https://github.com/AstraZeneca/runnable-core.git",
                        "code_identifier_message": ""
                    }
                ],
                "attempts": [],
                "user_defined_metrics": {},
                "branches": {},
                "data_catalog": []
            },
            "Create Content": {
                "name": "Create Content",
                "internal_name": "Create Content",
                "status": "SUCCESS",
                "step_type": "task",
                "message": "",
                "mock": true,
                "code_identities": [
                    {
                        "code_identifier": "f94e49a4fcecebac4d5eecbb5b691561b08e45c0",
                        "code_identifier_type": "git",
                        "code_identifier_dependable": true,
                        "code_identifier_url": "https://github.com/AstraZeneca/runnable-core.git",
                        "code_identifier_message": ""
                    }
                ],
                "attempts": [],
                "user_defined_metrics": {},
                "branches": {},
                "data_catalog": []
            },
            "Retrieve Content": {
                "name": "Retrieve Content",
                "internal_name": "Retrieve Content",
                "status": "SUCCESS",
                "step_type": "task",
                "message": "",
                "mock": false,
                "code_identities": [
                    {
                        "code_identifier": "f94e49a4fcecebac4d5eecbb5b691561b08e45c0",
                        "code_identifier_type": "git",
                        "code_identifier_dependable": true,
                        "code_identifier_url": "https://github.com/AstraZeneca/runnable-core.git",
                        "code_identifier_message": ""
                    }
                ],
                "attempts": [
                    {
                        "attempt_number": 1,
                        "start_time": "2024-02-05 22:26:05.366143",
                        "end_time": "2024-02-05 22:26:05.383790",
                        "duration": "0:00:00.017647",
                        "status": "SUCCESS",
                        "message": "",
                        "parameters": {}
                    }
                ],
                "user_defined_metrics": {},
                "branches": {},
                "data_catalog": [
                    {
                        "name": "data/hello.txt",
                        "data_hash": "14e0a818c551fd963f9496f5b9e780f741e3ee020456c7d8b761b902fbfa4cb4",
                        "catalog_relative_path": "data/hello.txt",
                        "catalog_handler_location": ".catalog",
                        "stage": "get"
                    },
                    {
                        "name": "Retrieve_Content.execution.log",
                        "data_hash": "f7911c18bf8be5131e6f61eecbeaf607758b9bf38a84b237e2aad7497ff46211",
                        "catalog_relative_path": "polynomial-bartik-2226/Retrieve_Content.execution.log",
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
                        "code_identifier": "f94e49a4fcecebac4d5eecbb5b691561b08e45c0",
                        "code_identifier_type": "git",
                        "code_identifier_dependable": true,
                        "code_identifier_url": "https://github.com/AstraZeneca/runnable-core.git",
                        "code_identifier_message": ""
                    }
                ],
                "attempts": [
                    {
                        "attempt_number": 1,
                        "start_time": "2024-02-05 22:26:05.465249",
                        "end_time": "2024-02-05 22:26:05.466008",
                        "duration": "0:00:00.000759",
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
                "overrides": {}
            },
            "run_log_store": {
                "service_name": "file-system",
                "service_type": "run_log_store",
                "log_folder": ".run_log_store"
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
            "pipeline_file": "examples/retry-fixed.yaml",
            "parameters_file": null,
            "configuration_file": "examples/configs/fs-catalog-run_log.yaml",
            "tag": "",
            "run_id": "polynomial-bartik-2226",
            "variables": {
                "argo_docker_image": "harbor.csis.astrazeneca.net/mlops/runnable:latest"
            },
            "use_cached": true,
            "original_run_id": "toFail",
            "dag": {
                "start_at": "Setup",
                "name": "",
                "description": "This is a simple pipeline that demonstrates passing data between steps.\n\n1. Setup: We setup a data folder, we ignore if it is already
    present\n2. Create Content: We create a \"hello.txt\" and \"put\" the file in catalog\n3. Clean up to get again: We remove the data folder. Note that this is stubbed
    to prevent\n  accidental deletion of your contents. You can change type to task to make really run.\n4. Retrieve Content: We \"get\" the file \"hello.txt\" from the
    catalog and show the contents\n5. Cleanup: We remove the data folder. Note that this is stubbed to prevent accidental deletion.\n\n\nYou can run this pipeline by:\n
    runnable execute -f examples/catalog.yaml -c examples/configs/fs-catalog.yaml\n",
                "steps": {
                    "Setup": {
                        "type": "task",
                        "name": "Setup",
                        "next": "Create Content",
                        "on_failure": "",
                        "overrides": {},
                        "catalog": null,
                        "max_attempts": 1,
                        "command_type": "shell",
                        "command": "mkdir -p data",
                        "node_name": "Setup"
                    },
                    "Create Content": {
                        "type": "task",
                        "name": "Create Content",
                        "next": "Retrieve Content",
                        "on_failure": "",
                        "overrides": {},
                        "catalog": {
                            "get": [],
                            "put": [
                                "data/hello.txt"
                            ]
                        },
                        "max_attempts": 1,
                        "command_type": "shell",
                        "command": "echo \"Hello from runnable\" >> data/hello.txt\n",
                        "node_name": "Create Content"
                    },
                    "Retrieve Content": {
                        "type": "task",
                        "name": "Retrieve Content",
                        "next": "success",
                        "on_failure": "",
                        "overrides": {},
                        "catalog": {
                            "get": [
                                "data/hello.txt"
                            ],
                            "put": []
                        },
                        "max_attempts": 1,
                        "command_type": "shell",
                        "command": "cat data/hello.txt",
                        "node_name": "Retrieve Content"
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
            "dag_hash": "2beec08fd417134cd3b04599d6684469db4ad176",
            "execution_plan": "chained"
        }
    }

    ```

=== "Diff"

    ```diff
    diff .run_log_store/toFail.json .run_log_store/polynomial-bartik-2226.json
    2,4c2,4
    <     "run_id": "toFail",
    <     "dag_hash": "13f7c1b29ebb07ce058305253171ceae504e1683",
    <     "use_cached": false,
    ---
    >     "run_id": "polynomial-bartik-2226",
    >     "dag_hash": "2beec08fd417134cd3b04599d6684469db4ad176",
    >     "use_cached": true,
    6,7c6,7
    <     "original_run_id": "",
    <     "status": "PROCESSING",
    ---
    >     "original_run_id": "toFail",
    >     "status": "SUCCESS",
    15c15
    <             "mock": false,
    ---
    >             "mock": true,
    25,35c25
    <             "attempts": [
    <                 {
    <                     "attempt_number": 1,
    <                     "start_time": "2024-02-05 22:11:47.213714",
    <                     "end_time": "2024-02-05 22:11:47.290352",
    <                     "duration": "0:00:00.076638",
    <                     "status": "SUCCESS",
    <                     "message": "",
    <                     "parameters": {}
    <                 }
    <             ],
    ---
    >             "attempts": [],
    38,46c28
    <             "data_catalog": [
    <                 {
    <                     "name": "Setup.execution.log",
    <                     "data_hash": "b709b710424701bd86be1cca36c5ec18f412b6dbb8d4e7729ec10e44319adbaf",
    <                     "catalog_relative_path": "toFail/Setup.execution.log",
    <                     "catalog_handler_location": "/mnt/catalog",
    <                     "stage": "put"
    <                 }
    <             ]
    ---
    >             "data_catalog": []
    53a36,56
    >             "mock": true,
    >             "code_identities": [
    >                 {
    >                     "code_identifier": "f94e49a4fcecebac4d5eecbb5b691561b08e45c0",
    >                     "code_identifier_type": "git",
    >                     "code_identifier_dependable": true,
    >                     "code_identifier_url": "https://github.com/AstraZeneca/runnable-core.git",
    >                     "code_identifier_message": ""
    >                 }
    >             ],
    >             "attempts": [],
    >             "user_defined_metrics": {},
    >             "branches": {},
    >             "data_catalog": []
    >         },
    >         "Retrieve Content": {
    >             "name": "Retrieve Content",
    >             "internal_name": "Retrieve Content",
    >             "status": "SUCCESS",
    >             "step_type": "task",
    >             "message": "",
    67,69c70,72
    <                     "start_time": "2024-02-05 22:12:14.210011",
    <                     "end_time": "2024-02-05 22:12:14.225645",
    <                     "duration": "0:00:00.015634",
    ---
    >                     "start_time": "2024-02-05 22:26:05.366143",
    >                     "end_time": "2024-02-05 22:26:05.383790",
    >                     "duration": "0:00:00.017647",
    79,83c82,86
    <                     "name": "Create_Content.execution.log",
    <                     "data_hash": "618e515729e00c7811865306b41e91d698c00577078e75b2e4bcf87ec9669d62",
    <                     "catalog_relative_path": "toFail/Create_Content.execution.log",
    <                     "catalog_handler_location": "/mnt/catalog",
    <                     "stage": "put"
    ---
    >                     "name": "data/hello.txt",
    >                     "data_hash": "14e0a818c551fd963f9496f5b9e780f741e3ee020456c7d8b761b902fbfa4cb4",
    >                     "catalog_relative_path": "data/hello.txt",
    >                     "catalog_handler_location": ".catalog",
    >                     "stage": "get"
    86,89c89,92
    <                     "name": "data/hello.txt",
    <                     "data_hash": "949a4f1afcea77b4b3f483ebe993e733122fb87b7539a3fc3d6752030be6ad44",
    <                     "catalog_relative_path": "toFail/data/hello.txt",
    <                     "catalog_handler_location": "/mnt/catalog",
    ---
    >                     "name": "Retrieve_Content.execution.log",
    >                     "data_hash": "f7911c18bf8be5131e6f61eecbeaf607758b9bf38a84b237e2aad7497ff46211",
    >                     "catalog_relative_path": "polynomial-bartik-2226/Retrieve_Content.execution.log",
    >                     "catalog_handler_location": ".catalog",
    94,98c97,101
    <         "Retrieve Content": {
    <             "name": "Retrieve Content",
    <             "internal_name": "Retrieve Content",
    <             "status": "FAIL",
    <             "step_type": "task",
    ---
    >         "success": {
    >             "name": "success",
    >             "internal_name": "success",
    >             "status": "SUCCESS",
    >             "step_type": "success",
    113,117c116,120
    <                     "start_time": "2024-02-05 22:12:36.514484",
    <                     "end_time": "2024-02-05 22:12:36.985694",
    <                     "duration": "0:00:00.471210",
    <                     "status": "FAIL",
    <                     "message": "Command failed",
    ---
    >                     "start_time": "2024-02-05 22:26:05.465249",
    >                     "end_time": "2024-02-05 22:26:05.466008",
    >                     "duration": "0:00:00.000759",
    >                     "status": "SUCCESS",
    >                     "message": "",
    123,138c126
    <             "data_catalog": [
    <                 {
    <                     "name": "data/hello.txt",
    <                     "data_hash": "949a4f1afcea77b4b3f483ebe993e733122fb87b7539a3fc3d6752030be6ad44",
    <                     "catalog_relative_path": "data/hello.txt",
    <                     "catalog_handler_location": "/mnt/catalog",
    <                     "stage": "get"
    <                 },
    <                 {
    <                     "name": "Retrieve_Content.execution.log",
    <                     "data_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    <                     "catalog_relative_path": "toFail/Retrieve_Content.execution.log",
    <                     "catalog_handler_location": "/mnt/catalog",
    <                     "stage": "put"
    <                 }
    <             ]
    ---
    >             "data_catalog": []
    144c132
    <             "service_name": "argo",
    ---
    >             "service_name": "local",
    147,177c135
    <             "overrides": {},
    <             "image": "$argo_docker_image",
    <             "expose_parameters_as_inputs": true,
    <             "secrets_from_k8s": [],
    <             "output_file": "argo-pipeline.yaml",
    <             "name": "runnable-dag-",
    <             "annotations": {},
    <             "labels": {},
    <             "activeDeadlineSeconds": 172800,
    <             "nodeSelector": null,
    <             "parallelism": null,
    <             "retryStrategy": {
    <                 "limit": "0",
    <                 "retryPolicy": "Always",
    <                 "backoff": {
    <                     "duration": "120",
    <                     "factor": 2,
    <                     "maxDuration": "3600"
    <                 }
    <             },
    <             "max_step_duration_in_seconds": 7200,
    <             "tolerations": null,
    <             "image_pull_policy": "",
    <             "service_account_name": "default-editor",
    <             "persistent_volumes": [
    <                 {
    <                     "name": "runnable-volume",
    <                     "mount_path": "/mnt"
    <                 }
    <             ],
    <             "step_timeout": 14400
    ---
    >             "overrides": {}
    182c140
    <             "log_folder": "/mnt/run_log_store"
    ---
    >             "log_folder": ".run_log_store"
    191c149
    <             "catalog_location": "/mnt/catalog"
    ---
    >             "catalog_location": ".catalog"
    197c155
    <         "pipeline_file": "examples/retry-fail.yaml",
    ---
    >         "pipeline_file": "examples/retry-fixed.yaml",
    199c157
    <         "configuration_file": "examples/configs/argo-config-catalog.yaml",
    ---
    >         "configuration_file": "examples/configs/fs-catalog-run_log.yaml",
    201,204c159,164
    <         "run_id": "toFail",
    <         "variables": {},
    <         "use_cached": false,
    <         "original_run_id": "",
    ---
    >         "run_id": "polynomial-bartik-2226",
    >         "variables": {
    >             "argo_docker_image": "harbor.csis.astrazeneca.net/mlops/runnable:latest"
    >         },
    >         "use_cached": true,
    >         "original_run_id": "toFail",
    208c168
    <             "description": "This is a simple pipeline that demonstrates retrying failures.\n\n1. Setup: We setup a data folder, we ignore if it is already present\n2. Create Content: We create a \"hello.txt\" and \"put\" the file in catalog\n3. Retrieve Content: We \"get\" the file \"hello.txt\" from the catalog and show the contents\n5. Cleanup: We remove the data folder. Note that this is stubbed to prevent accidental deletion.\n\n\nYou can run this pipeline by:\n   runnable execute -f examples/catalog.yaml -c examples/configs/fs-catalog.yaml\n",
    ---
    >             "description": "This is a simple pipeline that demonstrates passing data between steps.\n\n1. Setup: We setup a data folder, we ignore if it is already present\n2. Create Content: We create a \"hello.txt\" and \"put\" the file in catalog\n3. Clean up to get again: We remove the data folder. Note that this is stubbed to prevent\n  accidental deletion of your contents. You can change type to task to make really run.\n4. Retrieve Content: We \"get\" the file \"hello.txt\" from the catalog and show the contents\n5. Cleanup: We remove the data folder. Note that this is stubbed to prevent accidental deletion.\n\n\nYou can run this pipeline by:\n   runnable execute -f examples/catalog.yaml -c examples/configs/fs-catalog.yaml\n",
    253c213
    <                     "command": "cat data/hello1.txt",
    ---
    >                     "command": "cat data/hello.txt",
    266c226
    <         "dag_hash": "13f7c1b29ebb07ce058305253171ceae504e1683",
    ---
    >         "dag_hash": "2beec08fd417134cd3b04599d6684469db4ad176",
    ```


## API

Tasks can access the ```run log``` during the execution of the step
[using the API](../interactions.md/#runnable.get_run_log). The run log returned by this method is a deep copy
to prevent any modifications.


Tasks can also access the ```run_id``` of the current execution either by
[using the API](../interactions.md/#runnable.get_run_id) or by the environment
variable ```runnable_RUN_ID```.
