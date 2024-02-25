Along with tracking the progress and status of the execution of the pipeline, run log
also keeps a track of parameters, experiment tracking metrics, data flowing through
the pipeline and any reproducibility metrics emitted by the tasks of the pipeline.

Please refer here for detailed [information about run log](../concepts/run-log.md).


## buffered

Stores all the run log in-memory. The run log is not persisted and destroyed immediately
after the execution is complete.

!!! warning inline end "Parallel execution"

    ```buffered``` run log stores suffers from race conditions when two tasks
    need to update status concurrently.


### Configuration

```yaml linenums="1"
run_log_store:
  type: buffered
```

<hr style="border:2px dotted orange">

## file-system

Stores the run log as a ```json``` file in the file-system accessible by all the steps
of the pipeline.


!!! warning inline end "Parallel execution"

    ```file-system``` based run log stores suffers from race conditions when two tasks
    need to update status concurrently. Use ```chunked``` version to avoid this behavior
    or disable parallelism.



### Configuration

```yaml linenums="1"
run_log_store:
  type: file-system
  config:
    log_folder: # defaults to  ".run_log_store"
```

### Example

=== "Configuration"

    Assumed to be present at ```examples/configs/fs-run_log.yaml```

    ```yaml linenums="1"
    --8<-- "examples/configs/fs-run_log.yaml"
    ```


=== "sdk pipeline"

    The configuration can be provided dynamically by setting the environment variable
    ```runnable_CONFIGURATION_FILE```.

    Executing the pipeline with:

    ```runnable_CONFIGURATION_FILE=examples/configs/fs-run_log.yaml python examples/concepts/simple.py```

    ```python linenums="1"
    --8<-- "examples/concepts/simple.py"
    ```

=== "Run log"

    The structure of the run log is [detailed in concepts](../concepts/run-log.md).

    ```json linenums="1"
    {
        "run_id": "blocking-shaw-0538",
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
                        "code_identifier": "39cd98770cb2fd6994d8ac08ae4c5506e5ce694a",
                        "code_identifier_type": "git",
                        "code_identifier_dependable": true,
                        "code_identifier_url": "https://github.com/AstraZeneca/runnable-core.git",
                        "code_identifier_message": ""
                    }
                ],
                "attempts": [
                    {
                        "attempt_number": 1,
                        "start_time": "2024-02-02 05:38:07.973392",
                        "end_time": "2024-02-02 05:38:07.977228",
                        "duration": "0:00:00.003836",
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
                        "catalog_relative_path": "blocking-shaw-0538/simple.execution.log",
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
                        "code_identifier": "39cd98770cb2fd6994d8ac08ae4c5506e5ce694a",
                        "code_identifier_type": "git",
                        "code_identifier_dependable": true,
                        "code_identifier_url": "https://github.com/AstraZeneca/runnable-core.git",
                        "code_identifier_message": ""
                    }
                ],
                "attempts": [
                    {
                        "attempt_number": 1,
                        "start_time": "2024-02-02 05:38:08.056864",
                        "end_time": "2024-02-02 05:38:08.057359",
                        "duration": "0:00:00.000495",
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
            "pipeline_file": "",
            "parameters_file": "",
            "configuration_file": "examples/configs/fs-run_log.yaml",
            "tag": "",
            "run_id": "blocking-shaw-0538",
            "use_cached": false,
            "original_run_id": "",
            "dag": {
                "start_at": "simple",
                "name": "",
                "description": "",
                "steps": {
                    "simple": {
                        "type": "task",
                        "name": "simple",
                        "next": "success",
                        "on_failure": "",
                        "overrides": {},
                        "catalog": null,
                        "max_attempts": 1,
                        "command": "examples.concepts.simple.simple_function",
                        "command_type": "python",
                        "node_name": "simple"
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

=== "folder structure"

    All the run logs are stored in .run_log_store with the filename being the ```run_id```.

    ```
    >>> tree .run_log_store
    .run_log_store
    └── blocking-shaw-0538.json

    1 directory, 1 file
    ```

<hr style="border:2px dotted orange">


## chunked-fs

Chunked file system is similar to the ```file-system``` but stores concents of the run log
that have concurrency blocks in separate files.


### Configuration

```yaml linenums="1"
run_log_store:
  type: chunked-fs
  config:
    log_folder: # defaults to  ".run_log_store"
```

=== "Configuration"

    Assumed to be present at ```examples/configs/chunked-fs-run_log.yaml```

    ```yaml linenums="1"
    --8<-- "examples/configs/chunked-fs-run_log.yaml"
    ```


=== "sdk pipeline"

    The configuration can be provided dynamically by setting the environment variable
    ```runnable_CONFIGURATION_FILE```.

    Executing the pipeline with:

    ```runnable_CONFIGURATION_FILE=examples/configs/chunked-fs-run_log.yaml python examples/concepts/simple.py```

    ```python linenums="1"
    --8<-- "examples/concepts/simple.py"
    ```

=== "Run log"

    The structure of the run log is [detailed in concepts](../concepts/run-log.md).

    === "RunLog.json"

        Stores only the metadata of the run log. The contents of this are safe for concurrent
        executions.

        ```json linenums="1"
        {
            "run_id": "pleasant-lamarr-0549",
            "dag_hash": "",
            "use_cached": false,
            "tag": "",
            "original_run_id": "",
            "status": "SUCCESS",
            "steps": {},
            "parameters": {},
            "run_config": {
                "executor": {
                    "service_name": "local",
                    "service_type": "executor",
                    "enable_parallel": false,
                    "overrides": {}
                },
                "run_log_store": {
                    "service_name": "chunked-fs",
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
                "pipeline_file": "",
                "parameters_file": "",
                "configuration_file": "examples/configs/chunked-fs-run_log.yaml",
                "tag": "",
                "run_id": "pleasant-lamarr-0549",
                "use_cached": false,
                "original_run_id": "",
                "dag": {
                    "start_at": "simple",
                    "name": "",
                    "description": "",
                    "steps": {
                        "simple": {
                            "type": "task",
                            "name": "simple",
                            "next": "success",
                            "on_failure": "",
                            "overrides": {},
                            "catalog": null,
                            "max_attempts": 1,
                            "command": "examples.concepts.simple.simple_function",
                            "command_type": "python",
                            "node_name": "simple"
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

    === "StepLog-simple-1706852981689005000.json"

        Contains only the information of the single step ```simple```.
        The name of the file follows the pattern:

        ```StepLog-<Step name>-<timestamp>.json```. The timestamp allows runnable to infer
        the order of execution of the steps.

        ```json linenums="1"
        {
            "name": "simple",
            "internal_name": "simple",
            "status": "SUCCESS",
            "step_type": "task",
            "message": "",
            "mock": false,
            "code_identities": [
                {
                    "code_identifier": "39cd98770cb2fd6994d8ac08ae4c5506e5ce694a",
                    "code_identifier_type": "git",
                    "code_identifier_dependable": true,
                    "code_identifier_url": "https://github.com/AstraZeneca/runnable-core.git",
                    "code_identifier_message": ""
                }
            ],
            "attempts": [
                {
                    "attempt_number": 1,
                    "start_time": "2024-02-02 05:49:41.697142",
                    "end_time": "2024-02-02 05:49:41.702983",
                    "duration": "0:00:00.005841",
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
                    "catalog_relative_path": "pleasant-lamarr-0549/simple.execution.log",
                    "catalog_handler_location": ".catalog",
                    "stage": "put"
                }
            ]
        }
        ```


=== "folder structure"

    All the run logs are stored in .run_log_store with the directory name being the ```run_id```.

    Instead of storing a single ```json``` file, the contents are stored in the folder
    by the name of the ```run_id``.

    ```
    .run_log_store
    └── pleasant-lamarr-0549
        ├── RunLog.json
        ├── StepLog-simple-1706852981689005000.json
        └── StepLog-success-1706852981779002000.json

    2 directories, 3 files
    ```
