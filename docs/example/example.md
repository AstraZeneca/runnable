

runnable revolves around the concept of [pipelines or workflows](../concepts/pipeline.md).
Pipelines defined in runnable are translated into
other workflow engine definitions like [Argo workflows](https://argoproj.github.io/workflows/) or
[AWS step functions](https://aws.amazon.com/step-functions/).

## Example Pipeline definition

A contrived example of data science workflow without any implementation.

!!! info annotate inline end "Simple pipeline"

    In this extremely reduced example, we acquire data from different sources, clean it and shape it for analysis.
    Features are then engineered from the clean data to run data science modelling.


``` mermaid
%%{ init: { 'flowchart': { 'curve': 'linear' } } }%%
flowchart TD

    step1:::green
    step1([Acquire data]) --> step2:::green
    step2([Prepare data]) --> step3:::green
    step3([Extract features]) --> step4:::green
    step4([Model]) --> suc([success]):::green

    classDef green stroke:#0f0

```


This pipeline can be represented in **runnable** as below:


=== "yaml"

    ``` yaml linenums="1"
    --8<-- "examples/contrived.yaml"
    ```

    1. ```stub``` nodes are mock nodes and always succeed.
    2. Execute the ```next``` node if it succeeds.
    3. This marks the pipeline to be be successfully completed.
    4. Any failure in the execution of the node will, by default, reach this step.

=== "python"

    ``` python linenums="1"
    --8<-- "examples/contrived.py"
    ```

    1. You can specify dependencies by using the ```next``` while creating the node or defer it for later.
    2. ```terminate_with_success``` indicates the pipeline to be successfully complete.
    3. Alternative ways to define dependencies, ```>>```, ```<<```, ```depends_on```. Choose the style that you
    prefer.
    4. ```add_terminal_nodes``` adds success and fail states to the pipeline.
    5. A very rich run log that captures different properties of the run for maximum reproducibility.


=== "Run log"

    Please see [Run log](../concepts/run-log.md) for more detailed information about the structure.

    ```json linenums="1"
    {
        "run_id": "vain-hopper-0731", // (1)
        "dag_hash": "",
        "use_cached": false,
        "tag": "",
        "original_run_id": "",
        "status": "SUCCESS", / (2)
        "steps": {
            "Acquire Data": {
                "name": "Acquire Data", // (3)
                "internal_name": "Acquire Data",
                "status": "SUCCESS",
                "step_type": "stub",
                "message": "",
                "mock": false,
                "code_identities": [
                    {
                        "code_identifier": "399b0d42f4f28aaeeb2e062bb0b938d50ff1595c", // (4)
                        "code_identifier_type": "git",
                        "code_identifier_dependable": true,
                        "code_identifier_url": "https://github.com/AstraZeneca/runnable-core.git",
                        "code_identifier_message": ""
                    }
                ],
                "attempts": [
                    {
                        "attempt_number": 1,
                        "start_time": "2023-11-16 07:31:39.929797",
                        "end_time": "2023-11-16 07:31:39.929815",
                        "duration": "0:00:00.000018",
                        "status": "SUCCESS",
                        "message": "",
                        "parameters": {}
                    }
                ],
                "user_defined_metrics": {}, // (5)
                "branches": {},
                "data_catalog": [] // (6)
            },
            "Prepare Data": {
                "name": "Prepare Data",
                "internal_name": "Prepare Data",
                "status": "SUCCESS",
                "step_type": "stub",
                "message": "",
                "mock": false,
                "code_identities": [
                    {
                        "code_identifier": "399b0d42f4f28aaeeb2e062bb0b938d50ff1595c",
                        "code_identifier_type": "git",
                        "code_identifier_dependable": true,
                        "code_identifier_url": "https://github.com/AstraZeneca/runnable-core.git",
                        "code_identifier_message": ""
                    }
                ],
                "attempts": [
                    {
                        "attempt_number": 1,
                        "start_time": "2023-11-16 07:31:39.993807",
                        "end_time": "2023-11-16 07:31:39.993828",
                        "duration": "0:00:00.000021",
                        "status": "SUCCESS",
                        "message": "",
                        "parameters": {}
                    }
                ],
                "user_defined_metrics": {},
                "branches": {},
                "data_catalog": []
            },
            "Extract Features": {
                "name": "Extract Features",
                "internal_name": "Extract Features",
                "status": "SUCCESS",
                "step_type": "stub",
                "message": "",
                "mock": false,
                "code_identities": [
                    {
                        "code_identifier": "399b0d42f4f28aaeeb2e062bb0b938d50ff1595c",
                        "code_identifier_type": "git",
                        "code_identifier_dependable": true,
                        "code_identifier_url": "https://github.com/AstraZeneca/runnable-core.git",
                        "code_identifier_message": ""
                    }
                ],
                "attempts": [
                    {
                        "attempt_number": 1,
                        "start_time": "2023-11-16 07:31:40.056403",
                        "end_time": "2023-11-16 07:31:40.056420",
                        "duration": "0:00:00.000017",
                        "status": "SUCCESS",
                        "message": "",
                        "parameters": {}
                    }
                ],
                "user_defined_metrics": {},
                "branches": {},
                "data_catalog": []
            },
            "Model": {
                "name": "Model",
                "internal_name": "Model",
                "status": "SUCCESS",
                "step_type": "stub",
                "message": "",
                "mock": false,
                "code_identities": [
                    {
                        "code_identifier": "399b0d42f4f28aaeeb2e062bb0b938d50ff1595c",
                        "code_identifier_type": "git",
                        "code_identifier_dependable": true,
                        "code_identifier_url": "https://github.com/AstraZeneca/runnable-core.git",
                        "code_identifier_message": ""
                    }
                ],
                "attempts": [
                    {
                        "attempt_number": 1,
                        "start_time": "2023-11-16 07:31:40.118268",
                        "end_time": "2023-11-16 07:31:40.118285",
                        "duration": "0:00:00.000017",
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
                        "code_identifier": "399b0d42f4f28aaeeb2e062bb0b938d50ff1595c",
                        "code_identifier_type": "git",
                        "code_identifier_dependable": true,
                        "code_identifier_url": "https://github.com/AstraZeneca/runnable-core.git",
                        "code_identifier_message": ""
                    }
                ],
                "attempts": [
                    {
                        "attempt_number": 1,
                        "start_time": "2023-11-16 07:31:40.176718",
                        "end_time": "2023-11-16 07:31:40.176774",
                        "duration": "0:00:00.000056",
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
        "run_config": { // (7)
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
            "run_id": "vain-hopper-0731",
            "variables": {},
            "use_cached": false,
            "original_run_id": "",
            "dag": { // (8)
                "start_at": "Acquire Data",
                "name": "",
                "description": "",
                "max_time": 86400,
                "internal_branch_name": "",
                "steps": {
                    "Acquire Data": {
                        "type": "stub",
                        "name": "Acquire Data",
                        "internal_name": "Acquire Data",
                        "internal_branch_name": "",
                        "is_composite": false
                    },
                    "Prepare Data": {
                        "type": "stub",
                        "name": "Prepare Data",
                        "internal_name": "Prepare Data",
                        "internal_branch_name": "",
                        "is_composite": false
                    },
                    "Extract Features": {
                        "type": "stub",
                        "name": "Extract Features",
                        "internal_name": "Extract Features",
                        "internal_branch_name": "",
                        "is_composite": false
                    },
                    "Model": {
                        "type": "stub",
                        "name": "Model",
                        "internal_name": "Model",
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

    1. Unique execution id or run id for every run of the pipeline.
    2. The status of the execution, one of success, fail or processing.
    3. Steps as defined in the pipeline configuration.
    4. git hash of the code that was used to run the pipeline.
    5. Optional user defined metrics during the step execution. These are also made available to the experiment tracking
       tool, if they are configured.
    6. Data files that are ```get``` or ```put``` into a central storage during execution of the step.
    7. The configuration used to run the pipeline.
    8. The pipeline definition.


Independent of the platform it is run on,


- [x] The [pipeline definition](..//concepts/pipeline.md) remains the same from an author point of view.
The data scientists are always part of the process and contribute to the development even in production environments.

- [x] The [run log](../concepts/run-log.md) remains the same except for the execution configuration enabling users
to debug the pipeline execution in lower environments for failed executions or to validate the
expectation of the execution.




## Example configuration

To run the pipeline in different environments, we just provide the
[required configuration](../configurations/overview.md).

=== "Default Configuration"

    ``` yaml linenums="1"
    --8<-- "examples/configs/default.yaml"
    ```

    1. Run the pipeline in local environment.
    2. Use the buffer as run log, this will not persist the run log to disk.
    3. Do not move any files to central storage.
    4. Do not use any secrets manager.
    5. Do not integrate with any experiment tracking tools

=== "Argo Configuration"

    To render the pipeline in [argo specification](../configurations/executors/argo.md), mention the
    configuration during execution.

    yaml:

    ```runnable execute -f examples/contrived.yaml -c examples/configs/argo-config.yaml```


    python:

    Please refer to [containerised environments](../configurations/executors/container-environments.md) for more information.

    runnable_CONFIGURATION_FILE=examples/configs/argo-config.yaml python examples/contrived.py && runnable execute -f runnable-pipeline.yaml -c examples/configs/argo-config.yaml

    ``` yaml linenums="1" title="Argo Configuration"
    --8<-- "examples/configs/argo-config.yaml"
    ```

    1. Use argo workflows as the execution engine to run the pipeline.
    2. Run this docker image for every step of the pipeline. Please refer to
    [containerised environments](../configurations/executors/container-environments.md) for more details.
    3. Mount the volume from Kubernetes persistent volumes (runnable-volume) to /mnt directory.
    4. Resource constraints for the container runtime.
    5. Since every step runs in a container, the run log should be persisted. Here we are using the file-system as our
    run log store.
    6. Kubernetes PVC is mounted to every container as ```/mnt```, use that to surface the run log to every step.


=== "Transpiled Workflow"

    The below is the same workflow definition in argo specification.

    ```yaml linenums="1"
    --8<-- "examples/generated-argo-pipeline.yaml"
    ```
