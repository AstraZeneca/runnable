Parallel nodes in magnus allows you to run multiple pipelines in parallel and use your compute resources efficiently.

## Example

!!! note "Only stubs?"

    All the steps in the below example are ```stubbed``` for convenience. The functionality is similar
    even if the steps are execution units like ```tasks``` or any other nodes.

    We support deeply [nested steps](/concepts/nesting). For example, a step in the parallel branch can be a ```map``` which internally
    loops over a ```dag``` and so on. Though this functionality is useful, it can be difficult to debug and
    understand in large code bases.

Below is a stubbed out example of a pipeline that trains two models in parallel and create an ensemble model to
do the inference. The models XGBoost and Random Forest (RF model) are trained in parallel and training of the
ensemble model happens only after both models are (successfully) trained.

=== "Visualisation"

    In the below visualisation, the green lined steps happen in sequence and wait for the previous step to
    successfully complete.

    The branches lined in yellow run in parallel to each other but sequential within the branch.



    ```mermaid
    flowchart TD

        getFeatures([Get Features]):::green
        trainStep(Train Models):::green
        ensembleModel([Ensemble Modelling]):::green
        inference([Run Inference]):::green
        success([Success]):::green

        prepareXG([Prepare for XGBoost]):::yellow
        trainXG([Train XGBoost]):::yellow
        successXG([XGBoost success]):::yellow
        prepareXG --> trainXG --> successXG

        trainRF([Train RF model]):::yellow
        successRF([RF Model success]):::yellow
        trainRF --> successRF


        getFeatures --> trainStep
        trainStep --> prepareXG
        trainStep --> trainRF
        successXG --> ensembleModel
        successRF --> ensembleModel
        ensembleModel --> inference
        inference --> success


        classDef yellow stroke:#FFFF00
        classDef green stroke:#0f0


    ```

=== "Pipeline in yaml"

    ```yaml linenums="1"
    --8<-- "examples/concepts/parallel.yaml"
    ```

=== "python sdk"

    You can run this example by: ```python examples/concepts/parallel.py```

    ```python linenums="1"
    --8<-- "examples/concepts/parallel.py"
    ```

=== "Run log"

    The step log for the parallel branch ```Train models``` has branches which have similar
    structure to a run log.

    ```json linenums="1"
    {
        "run_id": "savory-pike-0201",
        "dag_hash": "",
        "use_cached": false,
        "tag": "",
        "original_run_id": "",
        "status": "SUCCESS",
        "steps": {
            "Get Features": {
                "name": "Get Features",
                "internal_name": "Get Features",
                "status": "SUCCESS",
                "step_type": "stub",
                "message": "",
                "mock": false,
                "code_identities": [
                    {
                        "code_identifier": "f0a2719001de9be30c27069933e4b4a64a065e2b",
                        "code_identifier_type": "git",
                        "code_identifier_dependable": true,
                        "code_identifier_url": "https://github.com/AstraZeneca/magnus-core.git",
                        "code_identifier_message": ""
                    }
                ],
                "attempts": [
                    {
                        "attempt_number": 1,
                        "start_time": "2024-01-18 02:01:10.978646",
                        "end_time": "2024-01-18 02:01:10.978665",
                        "duration": "0:00:00.000019",
                        "status": "SUCCESS",
                        "message": "",
                        "parameters": {}
                    }
                ],
                "user_defined_metrics": {},
                "branches": {},
                "data_catalog": []
            },
            "Train Models": {
                "name": "Train Models",
                "internal_name": "Train Models",
                "status": "SUCCESS",
                "step_type": "parallel",
                "message": "",
                "mock": false,
                "code_identities": [
                    {
                        "code_identifier": "f0a2719001de9be30c27069933e4b4a64a065e2b",
                        "code_identifier_type": "git",
                        "code_identifier_dependable": true,
                        "code_identifier_url": "https://github.com/AstraZeneca/magnus-core.git",
                        "code_identifier_message": ""
                    }
                ],
                "attempts": [],
                "user_defined_metrics": {},
                "branches": {
                    "Train Models.XGBoost": {
                        "internal_name": "Train Models.XGBoost",
                        "status": "SUCCESS",
                        "steps": {
                            "Train Models.XGBoost.Prepare for XGBoost": {
                                "name": "Prepare for XGBoost",
                                "internal_name": "Train Models.XGBoost.Prepare for XGBoost",
                                "status": "SUCCESS",
                                "step_type": "stub",
                                "message": "",
                                "mock": false,
                                "code_identities": [
                                    {
                                        "code_identifier": "f0a2719001de9be30c27069933e4b4a64a065e2b",
                                        "code_identifier_type": "git",
                                        "code_identifier_dependable": true,
                                        "code_identifier_url": "https://github.com/AstraZeneca/magnus-core.git",
                                        "code_identifier_message": ""
                                    }
                                ],
                                "attempts": [
                                    {
                                        "attempt_number": 1,
                                        "start_time": "2024-01-18 02:01:11.132822",
                                        "end_time": "2024-01-18 02:01:11.132840",
                                        "duration": "0:00:00.000018",
                                        "status": "SUCCESS",
                                        "message": "",
                                        "parameters": {}
                                    }
                                ],
                                "user_defined_metrics": {},
                                "branches": {},
                                "data_catalog": []
                            },
                            "Train Models.XGBoost.Train XGBoost": {
                                "name": "Train XGBoost",
                                "internal_name": "Train Models.XGBoost.Train XGBoost",
                                "status": "SUCCESS",
                                "step_type": "stub",
                                "message": "",
                                "mock": false,
                                "code_identities": [
                                    {
                                        "code_identifier": "f0a2719001de9be30c27069933e4b4a64a065e2b",
                                        "code_identifier_type": "git",
                                        "code_identifier_dependable": true,
                                        "code_identifier_url": "https://github.com/AstraZeneca/magnus-core.git",
                                        "code_identifier_message": ""
                                    }
                                ],
                                "attempts": [
                                    {
                                        "attempt_number": 1,
                                        "start_time": "2024-01-18 02:01:11.216418",
                                        "end_time": "2024-01-18 02:01:11.216430",
                                        "duration": "0:00:00.000012",
                                        "status": "SUCCESS",
                                        "message": "",
                                        "parameters": {}
                                    }
                                ],
                                "user_defined_metrics": {},
                                "branches": {},
                                "data_catalog": []
                            },
                            "Train Models.XGBoost.success": {
                                "name": "success",
                                "internal_name": "Train Models.XGBoost.success",
                                "status": "SUCCESS",
                                "step_type": "success",
                                "message": "",
                                "mock": false,
                                "code_identities": [
                                    {
                                        "code_identifier": "f0a2719001de9be30c27069933e4b4a64a065e2b",
                                        "code_identifier_type": "git",
                                        "code_identifier_dependable": true,
                                        "code_identifier_url": "https://github.com/AstraZeneca/magnus-core.git",
                                        "code_identifier_message": ""
                                    }
                                ],
                                "attempts": [
                                    {
                                        "attempt_number": 1,
                                        "start_time": "2024-01-18 02:01:11.291222",
                                        "end_time": "2024-01-18 02:01:11.292140",
                                        "duration": "0:00:00.000918",
                                        "status": "SUCCESS",
                                        "message": "",
                                        "parameters": {}
                                    }
                                ],
                                "user_defined_metrics": {},
                                "branches": {},
                                "data_catalog": []
                            }
                        }
                    },
                    "Train Models.RF Model": {
                        "internal_name": "Train Models.RF Model",
                        "status": "SUCCESS",
                        "steps": {
                            "Train Models.RF Model.Train RF": {
                                "name": "Train RF",
                                "internal_name": "Train Models.RF Model.Train RF",
                                "status": "SUCCESS",
                                "step_type": "stub",
                                "message": "",
                                "mock": false,
                                "code_identities": [
                                    {
                                        "code_identifier": "f0a2719001de9be30c27069933e4b4a64a065e2b",
                                        "code_identifier_type": "git",
                                        "code_identifier_dependable": true,
                                        "code_identifier_url": "https://github.com/AstraZeneca/magnus-core.git",
                                        "code_identifier_message": ""
                                    }
                                ],
                                "attempts": [
                                    {
                                        "attempt_number": 1,
                                        "start_time": "2024-01-18 02:01:11.379438",
                                        "end_time": "2024-01-18 02:01:11.379453",
                                        "duration": "0:00:00.000015",
                                        "status": "SUCCESS",
                                        "message": "",
                                        "parameters": {}
                                    }
                                ],
                                "user_defined_metrics": {},
                                "branches": {},
                                "data_catalog": []
                            },
                            "Train Models.RF Model.success": {
                                "name": "success",
                                "internal_name": "Train Models.RF Model.success",
                                "status": "SUCCESS",
                                "step_type": "success",
                                "message": "",
                                "mock": false,
                                "code_identities": [
                                    {
                                        "code_identifier": "f0a2719001de9be30c27069933e4b4a64a065e2b",
                                        "code_identifier_type": "git",
                                        "code_identifier_dependable": true,
                                        "code_identifier_url": "https://github.com/AstraZeneca/magnus-core.git",
                                        "code_identifier_message": ""
                                    }
                                ],
                                "attempts": [
                                    {
                                        "attempt_number": 1,
                                        "start_time": "2024-01-18 02:01:11.458716",
                                        "end_time": "2024-01-18 02:01:11.459695",
                                        "duration": "0:00:00.000979",
                                        "status": "SUCCESS",
                                        "message": "",
                                        "parameters": {}
                                    }
                                ],
                                "user_defined_metrics": {},
                                "branches": {},
                                "data_catalog": []
                            }
                        }
                    }
                },
                "data_catalog": []
            },
            "Ensemble Modelling": {
                "name": "Ensemble Modelling",
                "internal_name": "Ensemble Modelling",
                "status": "SUCCESS",
                "step_type": "stub",
                "message": "",
                "mock": false,
                "code_identities": [
                    {
                        "code_identifier": "f0a2719001de9be30c27069933e4b4a64a065e2b",
                        "code_identifier_type": "git",
                        "code_identifier_dependable": true,
                        "code_identifier_url": "https://github.com/AstraZeneca/magnus-core.git",
                        "code_identifier_message": ""
                    }
                ],
                "attempts": [
                    {
                        "attempt_number": 1,
                        "start_time": "2024-01-18 02:01:11.568072",
                        "end_time": "2024-01-18 02:01:11.568085",
                        "duration": "0:00:00.000013",
                        "status": "SUCCESS",
                        "message": "",
                        "parameters": {}
                    }
                ],
                "user_defined_metrics": {},
                "branches": {},
                "data_catalog": []
            },
            "Run Inference": {
                "name": "Run Inference",
                "internal_name": "Run Inference",
                "status": "SUCCESS",
                "step_type": "stub",
                "message": "",
                "mock": false,
                "code_identities": [
                    {
                        "code_identifier": "f0a2719001de9be30c27069933e4b4a64a065e2b",
                        "code_identifier_type": "git",
                        "code_identifier_dependable": true,
                        "code_identifier_url": "https://github.com/AstraZeneca/magnus-core.git",
                        "code_identifier_message": ""
                    }
                ],
                "attempts": [
                    {
                        "attempt_number": 1,
                        "start_time": "2024-01-18 02:01:11.650023",
                        "end_time": "2024-01-18 02:01:11.650037",
                        "duration": "0:00:00.000014",
                        "status": "SUCCESS",
                        "message": "",
                        "parameters": {}
                    }
                ],
                "user_defined_metrics": {},
                "branches": {},
                "data_catalog": []
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
                        "code_identifier": "f0a2719001de9be30c27069933e4b4a64a065e2b",
                        "code_identifier_type": "git",
                        "code_identifier_dependable": true,
                        "code_identifier_url": "https://github.com/AstraZeneca/magnus-core.git",
                        "code_identifier_message": ""
                    }
                ],
                "attempts": [
                    {
                        "attempt_number": 1,
                        "start_time": "2024-01-18 02:01:11.727802",
                        "end_time": "2024-01-18 02:01:11.728651",
                        "duration": "0:00:00.000849",
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
                "service_name": "file-system",
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
            "configuration_file": "examples/configs/fs-catalog-run_log.yaml",
            "tag": "",
            "run_id": "savory-pike-0201",
            "variables": {},
            "use_cached": false,
            "original_run_id": "",
            "dag": {
                "start_at": "Get Features",
                "name": "",
                "description": "",
                "steps": {
                    "Get Features": {
                        "type": "stub",
                        "name": "Get Features",
                        "next": "Train Models",
                        "on_failure": "",
                        "executor_config": {},
                        "catalog": null,
                        "max_attempts": 1
                    },
                    "Train Models": {
                        "type": "parallel",
                        "name": "Train Models",
                        "next": "Ensemble Modelling",
                        "on_failure": "",
                        "executor_config": {},
                        "branches": {
                            "XGBoost": {
                                "start_at": "Prepare for XGBoost",
                                "name": "",
                                "description": "",
                                "steps": {
                                    "Prepare for XGBoost": {
                                        "type": "stub",
                                        "name": "Prepare for XGBoost",
                                        "next": "Train XGBoost",
                                        "on_failure": "",
                                        "executor_config": {},
                                        "catalog": null,
                                        "max_attempts": 1
                                    },
                                    "Train XGBoost": {
                                        "type": "stub",
                                        "name": "Train XGBoost",
                                        "next": "success",
                                        "on_failure": "",
                                        "executor_config": {},
                                        "catalog": null,
                                        "max_attempts": 1
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
                            "RF Model": {
                                "start_at": "Train RF",
                                "name": "",
                                "description": "",
                                "steps": {
                                    "Train RF": {
                                        "type": "stub",
                                        "name": "Train RF",
                                        "next": "success",
                                        "on_failure": "",
                                        "executor_config": {},
                                        "catalog": null,
                                        "max_attempts": 1
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
                            }
                        }
                    },
                    "Ensemble Modelling": {
                        "type": "stub",
                        "name": "Ensemble Modelling",
                        "next": "Run Inference",
                        "on_failure": "",
                        "executor_config": {},
                        "catalog": null,
                        "max_attempts": 1
                    },
                    "Run Inference": {
                        "type": "stub",
                        "name": "Run Inference",
                        "next": "success",
                        "on_failure": "",
                        "executor_config": {},
                        "catalog": null,
                        "max_attempts": 1
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



All pipelines, nested or parent, have the same structure as defined in
[pipeline definition](/concepts/pipeline).

The parent pipeline defines a step ```Train models``` which is a parallel step.
The branches, XGBoost and RF model, are pipelines themselves.

## Traversal

A branch of a parallel step is considered success only if the ```success``` step is reached at the end.
The steps of the pipeline can fail and be handled by [on failure](/concepts/pipeline/#on_failure) and
redirected to ```success``` if that is the desired behavior.

The parallel step is considered successful only if all the branches of the step have terminated successfully.


## Parameters

All the tasks defined in the branches of the parallel pipeline can
[access to parameters and data as usual](/concepts/task).


!!! warning

    The parameters can be updated by all the tasks and the last task to execute overwrites
    the previous changes.

    Since the order of execution is not guaranteed, its best to avoid mutating the same parameters in
    the steps belonging to parallel step.
