Metrics in data science projects summarize important information about the execution and performance of the
experiment.

Magnus captures [this information as part of the run log](/concepts/experiment-tracking) and also provides
an [interface to experiment tracking tools](/concepts/experiment-tracking/#experiment_tracking_tools)
like [mlflow](https://mlflow.org/docs/latest/tracking.html) or
[Weights and Biases](https://wandb.ai/site/experiment-tracking).


### Example


=== "python"

    ```python linenums="1"
    --8<-- "examples/experiment_tracking_api.py"
    ```

    1. Nested metrics are possible as pydantic models.
    2. Using mlflow as experiment tracking tool.

=== "yaml"

    ```yaml linenums="1"
    --8<-- "examples/experiment_tracking_env.yaml"
    ```

=== "configuration"

    Assumed to be present in ```examples/configs/mlflow-config.yaml```

    ```yaml linenums="1"
    --8<-- "examples/configs/mlflow-config.yaml"
    ```

=== "Run log"

    The captured metrics as part of the run log are highlighted.

    ```json linenums="1" hl_lines="36-43"
    {
        "run_id": "clean-ride-1048",
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
                        "code_identifier": "0b62e4c661a4b4a2187afdf44a7c64520374202d",
                        "code_identifier_type": "git",
                        "code_identifier_dependable": true,
                        "code_identifier_url": "https://github.com/AstraZeneca/magnus-core.git",
                        "code_identifier_message": ""
                    }
                ],
                "attempts": [
                    {
                        "attempt_number": 1,
                        "start_time": "2024-01-10 10:48:10.089266",
                        "end_time": "2024-01-10 10:48:10.092541",
                        "duration": "0:00:00.003275",
                        "status": "SUCCESS",
                        "message": "",
                        "parameters": {}
                    }
                ],
                "user_defined_metrics": {
                    "spam": "hello",
                    "eggs": {
                        "ham": "world"
                    },
                    "answer": 42.0,
                    "is_it_true": false
                },
                "branches": {},
                "data_catalog": [
                    {
                        "name": "Emit_Metrics.execution.log",
                        "data_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                        "catalog_relative_path": "clean-ride-1048/Emit_Metrics.execution.log",
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
                        "code_identifier": "0b62e4c661a4b4a2187afdf44a7c64520374202d",
                        "code_identifier_type": "git",
                        "code_identifier_dependable": true,
                        "code_identifier_url": "https://github.com/AstraZeneca/magnus-core.git",
                        "code_identifier_message": ""
                    }
                ],
                "attempts": [
                    {
                        "attempt_number": 1,
                        "start_time": "2024-01-10 10:48:10.585832",
                        "end_time": "2024-01-10 10:48:10.585937",
                        "duration": "0:00:00.000105",
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
                "service_name": "mlflow",
                "service_type": "experiment_tracker"
            },
            "pipeline_file": "",
            "parameters_file": "",
            "configuration_file": "examples/configs/mlflow-config.yaml",
            "tag": "",
            "run_id": "clean-ride-1048",
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


=== "mlflow"

    The metrics are also sent to mlflow.

    <figure markdown>
        ![Image](/assets/screenshots/mlflow_example.png){ width="800" height="600"}
        <figcaption>mlflow UI for the execution. The run_id remains the same as the run_id of magnus</figcaption>
    </figure>
