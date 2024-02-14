Magnus allows you to [debug and recover](../concepts/run-log.md/#retrying_failures) from a
failure during the execution of pipeline. The pipeline can be
restarted in any suitable environment for debugging.


!!! example annotate

    A pipeline that is transpiled to argo workflows can be re-run on your local compute
    for debugging purposes. The only caveat is that, your local compute should have access to run log of the failed
    execution (1), generated catalog artifacts (2) from the the failed execution.

1. Access to the run log can be as simple as copy the json file to your local compute.
2. Generated catalog artifacts can be sourced from ```file-system``` which is your local folder.



Below is an example of retrying a pipeline that failed.


=== "Failed pipeline"

    !!! note

        You can run this pipeline on your local machine by

        ```magnus execute -f examples/retry-fail.yaml -c examples/configs/fs-catalog-run_log.yaml --run-id wrong-file-name```

        Note that we have specified the ```run_id``` to be something we can use later.
        The execution logs of the steps in the catalog will show the reason of the failure.

    ```yaml title="Pipeline that fails"
    --8<-- "examples/retry-fail.yaml"
    ```

    1. We make a data folder to store content.
    2. Puts a file in the data folder and catalogs it for downstream steps.
    3. It will fail here as there is no file called ```hello1.txt``` in the data folder.
    4. Get the file, ```hello.txt``` generated from previous steps into data folder.


=== "Failed run log"

    Please note the overall status of the pipeline in line #7 to be ```FAIL```.
    The step log of the failed step is also marked with status ```FAIL```.

    ```json linenums="1" hl_lines="7 94-139"
    {
        "run_id": "wrong-file-name",
        "dag_hash": "13f7c1b29ebb07ce058305253171ceae504e1683",
        "use_cached": false,
        "tag": "",
        "original_run_id": "",
        "status": "FAIL",
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
                        "code_identifier_url": "https://github.com/AstraZeneca/magnus-core.git",
                        "code_identifier_message": ""
                    }
                ],
                "attempts": [
                    {
                        "attempt_number": 1,
                        "start_time": "2024-02-07 06:08:45.330918",
                        "end_time": "2024-02-07 06:08:45.348227",
                        "duration": "0:00:00.017309",
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
                        "data_hash": "e1f8eaa5d49d88fae21fd8a34ff9774bcd4136bdbc3aa613f88a986261bac694",
                        "catalog_relative_path": "wrong-file-name/Setup.execution.log",
                        "catalog_handler_location": ".catalog",
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
                        "code_identifier_url": "https://github.com/AstraZeneca/magnus-core.git",
                        "code_identifier_message": ""
                    }
                ],
                "attempts": [
                    {
                        "attempt_number": 1,
                        "start_time": "2024-02-07 06:08:45.422420",
                        "end_time": "2024-02-07 06:08:45.438199",
                        "duration": "0:00:00.015779",
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
                        "data_hash": "e1f8eaa5d49d88fae21fd8a34ff9774bcd4136bdbc3aa613f88a986261bac694",
                        "catalog_relative_path": "wrong-file-name/Create_Content.execution.log",
                        "catalog_handler_location": ".catalog",
                        "stage": "put"
                    },
                    {
                        "name": "data/hello.txt",
                        "data_hash": "108ecead366a67c2bb17e223032e12629bcc21b4ab0fff77cf48a5b784f208c7",
                        "catalog_relative_path": "wrong-file-name/data/hello.txt",
                        "catalog_handler_location": ".catalog",
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
                        "code_identifier_url": "https://github.com/AstraZeneca/magnus-core.git",
                        "code_identifier_message": ""
                    }
                ],
                "attempts": [
                    {
                        "attempt_number": 1,
                        "start_time": "2024-02-07 06:08:45.525924",
                        "end_time": "2024-02-07 06:08:45.605381",
                        "duration": "0:00:00.079457",
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
                        "data_hash": "108ecead366a67c2bb17e223032e12629bcc21b4ab0fff77cf48a5b784f208c7",
                        "catalog_relative_path": "data/hello.txt",
                        "catalog_handler_location": ".catalog",
                        "stage": "get"
                    },
                    {
                        "name": "Retrieve_Content.execution.log",
                        "data_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                        "catalog_relative_path": "wrong-file-name/Retrieve_Content.execution.log",
                        "catalog_handler_location": ".catalog",
                        "stage": "put"
                    }
                ]
            },
            "fail": {
                "name": "fail",
                "internal_name": "fail",
                "status": "SUCCESS",
                "step_type": "fail",
                "message": "",
                "mock": false,
                "code_identities": [
                    {
                        "code_identifier": "f94e49a4fcecebac4d5eecbb5b691561b08e45c0",
                        "code_identifier_type": "git",
                        "code_identifier_dependable": true,
                        "code_identifier_url": "https://github.com/AstraZeneca/magnus-core.git",
                        "code_identifier_message": ""
                    }
                ],
                "attempts": [
                    {
                        "attempt_number": 1,
                        "start_time": "2024-02-07 06:08:45.701371",
                        "end_time": "2024-02-07 06:08:45.701954",
                        "duration": "0:00:00.000583",
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
            "pipeline_file": "examples/retry-fail.yaml",
            "parameters_file": null,
            "configuration_file": "examples/configs/fs-catalog-run_log.yaml",
            "tag": "",
            "run_id": "wrong-file-name",
            "variables": {
                "argo_docker_image": "harbor.csis.astrazeneca.net/mlops/magnus:latest"
            },
            "use_cached": false,
            "original_run_id": "",
            "dag": {
                "start_at": "Setup",
                "name": "",
                "description": "This is a simple pipeline that demonstrates retrying failures.\n\n1. Setup: We setup a data folder, we ignore if it is already present\n2. Create Content: We create a \"hello.txt\" and \"put\" the file in catalog\n3. Retrieve Content: We \"get\" the file \"hello.txt\" from the catalog and show the contents\n5. Cleanup: We remove the data folder. Note that this is stubbed to prevent accidental deletion.\n\n\nYou can run this pipeline by:\n   magnus execute -f examples/catalog.yaml -c examples/configs/fs-catalog.yaml\n",
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
                        "command": "echo \"Hello from magnus\" >> data/hello.txt\n",
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


=== "Fixed pipeline"

    !!! note

        You can run this pipeline on your local machine by

        ```magnus execute -f examples/retry-fixed.yaml -c examples/configs/fs-catalog-run_log.yaml --use-cached wrong-file-name```

        Note that we have specified the run_id of the failed execution to be ```use-cached``` for the new execution.


    ```yaml title="Pipeline that restarts"
    --8<-- "examples/retry-fixed.yaml"
    ```

    1. Though this step is identical to the failed pipeline, this step does not execute in retry.
    2. We mark this step to be stub to demonstrate a re-run using cached does not execute the
    successful task.



=== "Fixed Run log"

    The retry pipeline is executed with success state.

    Note the execution of step ```Setup``` has been marked as ```mock: true```, this step
    has not been executed but passed through.

    The step ```Create Content``` has been modified to ```stub``` to prevent execution in the
    fixed pipeline.

    ```json linenums="1" hl_lines="15 34 51-96"
    {
        "run_id": "naive-wilson-0625",
        "dag_hash": "148de99f96565bb1b276db2baf23eba682615c76",
        "use_cached": true,
        "tag": "",
        "original_run_id": "wrong-file-name",
        "status": "SUCCESS",
        "steps": {
            "Setup": {
                "name": "Setup",
                "internal_name": "Setup",
                "status": "SUCCESS",
                "step_type": "stub",
                "message": "",
                "mock": true,
                "code_identities": [
                    {
                        "code_identifier": "f94e49a4fcecebac4d5eecbb5b691561b08e45c0",
                        "code_identifier_type": "git",
                        "code_identifier_dependable": true,
                        "code_identifier_url": "https://github.com/AstraZeneca/magnus-core.git",
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
                "step_type": "stub",
                "message": "",
                "mock": true,
                "code_identities": [
                    {
                        "code_identifier": "f94e49a4fcecebac4d5eecbb5b691561b08e45c0",
                        "code_identifier_type": "git",
                        "code_identifier_dependable": true,
                        "code_identifier_url": "https://github.com/AstraZeneca/magnus-core.git",
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
                        "code_identifier_url": "https://github.com/AstraZeneca/magnus-core.git",
                        "code_identifier_message": ""
                    }
                ],
                "attempts": [
                    {
                        "attempt_number": 1,
                        "start_time": "2024-02-07 06:25:13.506657",
                        "end_time": "2024-02-07 06:25:13.527603",
                        "duration": "0:00:00.020946",
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
                        "data_hash": "108ecead366a67c2bb17e223032e12629bcc21b4ab0fff77cf48a5b784f208c7",
                        "catalog_relative_path": "data/hello.txt",
                        "catalog_handler_location": ".catalog",
                        "stage": "get"
                    },
                    {
                        "name": "Retrieve_Content.execution.log",
                        "data_hash": "bd8e06cb7432666dc3b1b0db8034966c034397863c7ff629c98ffd13966681d7",
                        "catalog_relative_path": "naive-wilson-0625/Retrieve_Content.execution.log",
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
                        "code_identifier_url": "https://github.com/AstraZeneca/magnus-core.git",
                        "code_identifier_message": ""
                    }
                ],
                "attempts": [
                    {
                        "attempt_number": 1,
                        "start_time": "2024-02-07 06:25:13.597125",
                        "end_time": "2024-02-07 06:25:13.597694",
                        "duration": "0:00:00.000569",
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
            "run_id": "naive-wilson-0625",
            "variables": {
                "argo_docker_image": "harbor.csis.astrazeneca.net/mlops/magnus:latest"
            },
            "use_cached": true,
            "original_run_id": "wrong-file-name",
            "dag": {
                "start_at": "Setup",
                "name": "",
                "description": "This is a simple pipeline that demonstrates passing data between steps.\n\n1. Setup: We setup a data folder, we ignore if it is already
    present\n2. Create Content: We create a \"hello.txt\" and \"put\" the file in catalog\n3. Clean up to get again: We remove the data folder. Note that this is stubbed
    to prevent\n  accidental deletion of your contents. You can change type to task to make really run.\n4. Retrieve Content: We \"get\" the file \"hello.txt\" from the
    catalog and show the contents\n5. Cleanup: We remove the data folder. Note that this is stubbed to prevent accidental deletion.\n\n\nYou can run this pipeline by:\n
    magnus execute -f examples/catalog.yaml -c examples/configs/fs-catalog.yaml\n",
                "steps": {
                    "Setup": {
                        "type": "stub",
                        "name": "Setup",
                        "next": "Create Content",
                        "on_failure": "",
                        "overrides": {},
                        "catalog": null,
                        "max_attempts": 1,
                        "command_type": "shell",
                        "command": "mkdir -p data"
                    },
                    "Create Content": {
                        "type": "stub",
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
                        "command": "echo \"Hello from magnus\" >> data/hello.txt\n"
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
            "dag_hash": "148de99f96565bb1b276db2baf23eba682615c76",
            "execution_plan": "chained"
        }
    }
    ```


Magnus also supports [```mocked``` executor](../configurations/executors/mocked.md) which can
patch and mock tasks to isolate and focus on the failed task. Since python functions and notebooks
are run in the same shell, it is possible to use
[python debugger](https://docs.python.org/3/library/pdb.html) and
[ploomber debugger](https://engine.ploomber.io/en/docs/user-guide/debugging/debuglater.html)
to debug failed tasks.
