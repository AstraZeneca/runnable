Magnus stores a variety of information about the current execution in [run log](../../concepts/run-log).
The run log is internally used
for keeping track of the execution (status of different steps, parameters, etc) but also has rich information
for reproducing the state at the time of pipeline execution.


The following are "invisibly" captured as part of the run log:

- Code: The ```git``` commit hash of the code used to run a pipeline is stored as part of the run log against
every step.
- Data hash: The data hash of the file passing through the catalog is stored as part of the run log. Since the
catalog itself is indexed against the execution id, it is easy to recreate the exact state of the data used
in the pipeline execution.
- Configuration: The configuration of the pipeline (dag definition, execution configuration) is also stored
as part of the run log.



!!! info annotate "Invisible?"

    Reproducibility should not be a "nice to have" but is a must in data science projects. We believe that
    it should not be left to the data scientist to be conscious of it but should be done without any active
    intervention.


Below we show an example pipeline and the different layers of the run log.


=== "Example pipeline"

    !!! info annotate "Example"

        This example pipeline is the same as the data flow pipeline showcasing flow of files.
        The create content step creates writes a new file which is stored in the catalog and the retrieve content
        gets it from the catalog.


    ```python title="simple data passing pipeline" linenums="1"
    --8<-- "examples/catalog_api.py"
    ```
=== "General run log attributes"

    !!! info annotate

        This section of the run log is about the over all status of the execution. It has information
        about the run_id, the execution status, re-run indicators and the final state of the parameters.


    ```json linenums="1"
    {
        "run_id": "greedy-yonath-1608", // (1)
        "dag_hash": "",
        "use_cached": false,
        "tag": "",
        "original_run_id": "",
        "status": "SUCCESS",
        ...
        "parameters": {}, // (2)
    }
    ```

    1. The unique run_id of the execution.
    2. The parameters at the end of the pipeline.


=== "Logs captured against a step"

    !!! info annotate

        The information stored against an execution of a step. We capture the git commit id's, data hashes,
        parameters at the point of execution. The execution logs are also stored in the catalog against the
        run id.


    ```json linenums="1"
    "create_content": { // (1)
        "name": "create_content",
        "internal_name": "create_content",
        "status": "SUCCESS", // (2)
        "step_type": "task",
        "message": "",
        "mock": false,
        "code_identities": [
            {
                "code_identifier": "ff60e7fa379c38adaa03755977057cd10acc4baa",  // (3)
                "code_identifier_type": "git",
                "code_identifier_dependable": true, // (4)
                "code_identifier_url": "https://github.com/AstraZeneca/magnus-core.git",
                "code_identifier_message": ""
            }
        ],
        "attempts": [
            {
                "attempt_number": 1,
                "start_time": "2023-12-15 16:08:51.869129",
                "end_time": "2023-12-15 16:08:51.878428",
                "duration": "0:00:00.009299",
                "status": "SUCCESS",
                "message": "",
                "parameters": {} // (5)
            }
        ],
        "user_defined_metrics": {},
        "branches": {},
        "data_catalog": [
            {
                "name": "data/hello.txt",  // (6)
                "data_hash": "c2e6b3d23c045731bf40a036aa6f558c9448da247e0cbb4ee3fcf10d3660ef18", // (7)
                "catalog_relative_path": "greedy-yonath-1608/data/hello.txt",
                "catalog_handler_location": ".catalog",
                "stage": "put"
            },
            {
                "name": "create_content",  // (8)
                "data_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                "catalog_relative_path": "greedy-yonath-1608/create_content",
                "catalog_handler_location": ".catalog",
                "stage": "put"
            }
        ]
    },
    ```

    1. The name of step.
    2. The status of the execution of the step.
    3. The git sha of the code at the point of execution of the pipeline.
    4. is True if the branch is clean, false otherwise.
    5. The parameters at the point of execution of the step.
    6. The name of the file that was "put" in the catalog by the step.
    7. The hash of the dataset put in the catalog.
    8. The execution logs of the step put in the catalog.


=== "Captured configuration"

    !!! info annotate

        The information about the configuration used to run the pipeline. It includes the configuration of the
        different ```services``` used, the pipeline definition and state of variables used at the time of
        execution of the pipeline.


    ```json linenums="1"
    "run_config": {
        "executor": { // (1)
            "service_name": "local",
            "service_type": "executor",
            "enable_parallel": false,
            "placeholders": {}
        },
        "run_log_store": { // (2)
            "service_name": "buffered",
            "service_type": "run_log_store"
        },
        "secrets_handler": { // (3)
            "service_name": "do-nothing",
            "service_type": "secrets"
        },
        "catalog_handler": { // (4)
            "service_name": "file-system",
            "service_type": "catalog",
            "compute_data_folder": "."
        },
        "experiment_tracker": { // (5)
            "service_name": "do-nothing",
            "service_type": "experiment_tracker"
        },
        "pipeline_file": "",    //  (6
        "parameters_file": "", // (7)
        "configuration_file": "examples/configs/fs-catalog.yaml", // (8)
        "tag": "",
        "run_id": "greedy-yonath-1608",
        "variables": {},
        "use_cached": false,
        "original_run_id": "",
        "dag": { // (9)
            "start_at": "create_content",
            "name": "",
            "description": "",
            "max_time": 86400,
            "internal_branch_name": "",
            "steps": {
                "create_content": {
                    "type": "task",
                    "name": "create_content",
                    "internal_name": "create_content",
                    "internal_branch_name": "",
                    "is_composite": false
                },
                "retrieve_content": {
                    "type": "task",
                    "name": "retrieve_content",
                    "internal_name": "retrieve_content",
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
    ```

    1. The configuration of the ```executor```
    2. The configuration of ```run log store```. The location where these logs are stored.
    3. The configuration of the secrets manager.
    4. The configuration of the catalog manager.
    5. The configuration of experiment tracker.
    6. The pipeline definition file, empty in this case as we use the SDK.
    7. The initial parameters file used for the execution.
    8. The configuration file used for the execution.
    9. The definition of the DAG being executed.



This structure of the run log is the same independent of where the pipeline was executed. This enables you
to reproduce a failed execution in complex environments on local environments for easier debugging.
