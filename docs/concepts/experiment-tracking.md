# Overview

[Run log](../run-log) stores a lot of information about the execution along with the metrics captured
during the execution of the pipeline.


## Example


=== "Using the API"

    The highlighted lines in the below example show how to [use the API](../../interactions/#magnus.track_this)

    Any pydantic model as a value would be dumped as a dict, respecting the alias, before tracking it.

    ```python linenums="1" hl_lines="3 17-19"
    --8<-- "examples/concepts/experiment_tracking_api.py"
    ```


=== "Using environment variables"

    The highlighted lines in the below example show how to use environment variables to track metrics.

    Only string values are allowed to be environment variables. Numeric values sent in as strings are converted
    to int/float before storing them as metrics.

    There is no support for boolean values in environment variables.

    ```yaml linenums="1" hl_lines="16-18"
    --8<-- "examples/concepts/experiment_tracking_env.yaml"
    ```

=== "Run log entry"

    Any experiment tracking metrics found during the execution of the task are stored in
    ```user_defined_metrics``` field of the step log.

    For example, below is the content for the shell execution.

    ```json linenums="1" hl_lines="36-42"
    {
        "run_id": "blazing-colden-0544",
        "dag_hash": "4494aeb907ef950934fbcc34b226f72134d06687",
        "use_cached": false,
        "tag": "",
        "original_run_id": "",
        "status": "SUCCESS",
        "steps": {
            "shell": {
                "name": "shell",
                "internal_name": "shell",
                "status": "SUCCESS",
                "step_type": "task",
                "message": "",
                "mock": false,
                "code_identities": [
                    {
                        "code_identifier": "793b052b8b603760ff1eb843597361219832b61c",
                        "code_identifier_type": "git",
                        "code_identifier_dependable": true,
                        "code_identifier_url": "https://github.com/AstraZeneca/magnus-core.git",
                        "code_identifier_message": ""
                    }
                ],
                "attempts": [
                    {
                        "attempt_number": 1,
                        "start_time": "2024-01-09 05:44:42.841295",
                        "end_time": "2024-01-09 05:44:42.849938",
                        "duration": "0:00:00.008643",
                        "status": "SUCCESS",
                        "message": "",
                        "parameters": {}
                    }
                ],
                "user_defined_metrics": {
                    "eggs": {
                        "ham": "world"
                    },
                    "answer": 42.0,
                    "spam": "hello"
                },
                "branches": {},
                "data_catalog": [
                    {
                        "name": "shell.execution.log",
                        "data_hash": "07723e6188e7893ac79e8f07b7cc15dd1a62d2974335f173a0b5a6e58a3735d6",
                        "catalog_relative_path": "blazing-colden-0544/shell.execution.log",
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
                        "code_identifier": "793b052b8b603760ff1eb843597361219832b61c",
                        "code_identifier_type": "git",
                        "code_identifier_dependable": true,
                        "code_identifier_url": "https://github.com/AstraZeneca/magnus-core.git",
                        "code_identifier_message": ""
                    }
                ],
                "attempts": [
                    {
                        "attempt_number": 1,
                        "start_time": "2024-01-09 05:44:42.913905",
                        "end_time": "2024-01-09 05:44:42.913963",
                        "duration": "0:00:00.000058",
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
                "service_type": "catalog"
            },
            "experiment_tracker": {
                "service_name": "do-nothing",
                "service_type": "experiment_tracker"
            },
            "pipeline_file": "examples/concepts/experiment_tracking_env.yaml",
            "parameters_file": null,
            "configuration_file": null,
            "tag": "",
            "run_id": "blazing-colden-0544",
            "variables": {},
            "use_cached": false,
            "original_run_id": "",
            "dag": {
                "start_at": "shell",
                "name": "",
                "description": "An example pipeline to demonstrate setting experiment tracking metrics\nusing environment variables. Any environment variable with
                prefix\n'MAGNUS_TRACK_' will be recorded as a metric captured during the step.\n\nYou can run this pipeline as:\n  magnus execute -f
                examples/concepts/experiment_tracking_env.yaml\n",
                "internal_branch_name": "",
                "steps": {
                    "shell": {
                        "type": "task",
                        "name": "shell",
                        "internal_name": "shell",
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
            "dag_hash": "4494aeb907ef950934fbcc34b226f72134d06687",
            "execution_plan": "chained"
        }
    }
    ```


## Incremental tracking

It is possible to track metrics over time within a task. To do so, use the ```step``` parameter in the API
or post-fixing ```_STEP_``` and the increment when using environment variables.

The step is defaulted to be 0.

### Example

=== "Using the API"

    The highlighted lines in the below example show how to [use the API](../../interactions/#magnus.track_this) with
    the step parameter.


    ```python linenums="1" hl_lines="3 17-20"
    --8<-- "examples/concepts/experiment_tracking_step.py"
    ```

=== "Using environment variables"

    The highlighted lines in the below example show how to use environment variables to track metrics.

    ```yaml linenums="1" hl_lines="16-20"
    --8<-- "examples/concepts/experiment_tracking_env_step.yaml"
    ```

=== "Run log entry"

    ```json
    {
        "run_id": "syrupy-jang-1528",
        "dag_hash": "",
        "use_cached": false,
        "tag": "",
        "original_run_id": "",
        "status": "SUCCESS",
        "steps": {
            "Emit Metrics": {
                "name": "Emit Metrics",
                "internal_name": "Emit Metrics",
                "status": "SUCCESS",
                "step_type": "task",
                "message": "",
                "mock": false,
                "code_identities": [
                    {
                        "code_identifier": "d80b86c0f9d053301868410908a1ed7c8e741d87",
                        "code_identifier_type": "git",
                        "code_identifier_dependable": true,
                        "code_identifier_url": "https://github.com/AstraZeneca/magnus-core.git",
                        "code_identifier_message": ""
                    }
                ],
                "attempts": [
                    {
                        "attempt_number": 1,
                        "start_time": "2024-01-09 15:28:17.625129",
                        "end_time": "2024-01-09 15:28:17.628064",
                        "duration": "0:00:00.002935",
                        "status": "SUCCESS",
                        "message": "",
                        "parameters": {}
                    }
                ],
                "user_defined_metrics": {
                    "spam": {
                        "0": "hello",
                        "1": "hello"
                    },
                    "eggs": {
                        "0": {
                            "ham": "world"
                        },
                        "1": {
                            "ham": "universe"
                        }
                    },
                    "answer": 42.0,
                    "is_it_true": false
                },
                "branches": {},
                "data_catalog": [
                    {
                        "name": "Emit_Metrics.execution.log",
                        "data_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                        "catalog_relative_path": "syrupy-jang-1528/Emit_Metrics.execution.log",
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
                        "code_identifier": "d80b86c0f9d053301868410908a1ed7c8e741d87",
                        "code_identifier_type": "git",
                        "code_identifier_dependable": true,
                        "code_identifier_url": "https://github.com/AstraZeneca/magnus-core.git",
                        "code_identifier_message": ""
                    }
                ],
                "attempts": [
                    {
                        "attempt_number": 1,
                        "start_time": "2024-01-09 15:28:17.816447",
                        "end_time": "2024-01-09 15:28:17.816528",
                        "duration": "0:00:00.000081",
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
                "service_type": "catalog"
            },
            "experiment_tracker": {
                "service_name": "do-nothing",
                "service_type": "experiment_tracker"
            },
            "pipeline_file": "",
            "parameters_file": "",
            "configuration_file": "",
            "tag": "",
            "run_id": "syrupy-jang-1528",
            "variables": {},
            "use_cached": false,
            "original_run_id": "",
            "dag": {
                "start_at": "Emit Metrics",
                "name": "",
                "description": "",
                "internal_branch_name": "",
                "steps": {
                    "Emit Metrics": {
                        "type": "task",
                        "name": "Emit Metrics",
                        "internal_name": "Emit Metrics",
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


## Client context


### Example


## Grouping experiments

## Experiment tracking tools
