Mocked executors provide a way to control the behavior of ```task``` node types to be either
pass through or execute a alternate command with modified configurations.

- [x] Runs the pipeline only in local environment.
- [x] Enables unit testing of the pipeline in both yaml and SDK definitions.
- [x] Isolates specific node(s) from the execution for further analysis.
- [ ] Not meant to be used for production deployments

### Options

```yaml
executor: mocked
config:
  patches:
    name of the name:
      command_configuration:
```

By default, all the ```task``` steps are passed through without an execution.
By providing ```patches```, indexed by the name of the node, gives control on the command
to run and the configuration of the command.

#### Command configuration for notebook nodes

```python``` and ```shell``` based tasks have no configuration options apart from the ```command```.
Notebook nodes have additional configuration options [detailed in concepts](../../concepts/task.md/#notebook).
Ploomber engine provides [rich options](https://engine.ploomber.io/en/docs/user-guide/debugging/debuglater.html) in debugging failed notebooks.


## Example

### Mocking nodes

The following example shows the simple case of mocking all the steps of the pipeline.

=== "pipeline in yaml"

    You can execute the mocked pipeline by:
    ```runnable execute -f examples/concepts/simple.yaml -c examples/configs/mocked-config-simple.yaml```

    ```yaml linenums="1"
    --8<-- "examples/concepts/simple.yaml"
    ```

=== "python sdk"

    You can execute the mocked pipeline by:

    ```runnable_CONFIGURATION_FILE=examples/configs/mocked-config-simple.yaml python examples/concepts/simple.py```

    ```python linenums="1"
    --8<-- "examples/concepts/simple.py"
    ```

=== "Mocked configuration"

    ```yaml linenums="1"
    --8<-- "examples/configs/mocked-config-simple.yaml"
    ```

=== "Run log"

    The flag ```mock``` is set to be ```true``` for the execution of node simple which
    denotes that the task was mocked.

    ```json linenums="1" hl_lines="15"
    {
        "run_id": "minty-goodall-0528",
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
                "mock": true,
                "code_identities": [
                    {
                        "code_identifier": "d76cf865af2f8e03b6c1205403351cbe42e6cdc4",
                        "code_identifier_type": "git",
                        "code_identifier_dependable": true,
                        "code_identifier_url": "https://github.com/AstraZeneca/runnable-core.git",
                        "code_identifier_message": ""
                    }
                ],
                "attempts": [
                    {
                        "attempt_number": 1,
                        "start_time": "2024-02-11 05:28:40.812597",
                        "end_time": "2024-02-11 05:28:40.812627",
                        "duration": "0:00:00.000030",
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
                        "code_identifier": "d76cf865af2f8e03b6c1205403351cbe42e6cdc4",
                        "code_identifier_type": "git",
                        "code_identifier_dependable": true,
                        "code_identifier_url": "https://github.com/AstraZeneca/runnable-core.git",
                        "code_identifier_message": ""
                    }
                ],
                "attempts": [
                    {
                        "attempt_number": 1,
                        "start_time": "2024-02-11 05:28:40.883909",
                        "end_time": "2024-02-11 05:28:40.884310",
                        "duration": "0:00:00.000401",
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
                "service_name": "mocked",
                "service_type": "executor",
                "enable_parallel": false,
                "overrides": {},
                "patches": {}
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
            "configuration_file": "examples/configs/mocked-config-simple.yaml",
            "tag": "",
            "run_id": "minty-goodall-0528",
            "variables": {},
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


### Patching nodes for unit testing

Pipelines are themselves code and should be testable. In the below example, we
take an example pipeline to test the behavior of the traversal.


The below pipeline is designed to follow: ```step 1 >> step 2 >> step 3``` in case of no failures
and ```step 1 >> step3``` in case of failure. The traversal is
[shown in concepts](../../concepts/pipeline.md/#on_failure).

!!! tip "Asserting Run log"

    The run log is a simple json file that can be parsed and validated against designed
    behaviors. You can also create the ```RunLog``` object by deserializing
    ```runnable.datastore.RunLog``` from the json.

    This can be handy when validating complex pipelines.

=== "pipeline in yaml"

    ```yaml linenums="1"
    --8<-- "examples/on-failure.yaml"
    ```


=== "python sdk"

    ```python linenums="1"
    --8<-- "examples/on_failure.py"
    ```

=== "Run log with no mocking"

    The ```run log``` has only ```step 1``` and ```step 3``` as part of the steps (as designed)
     showing the behavior of the pipeline in case of failure. The status of ```step 1``` is
     captured as ```FAIL``` due to ```exit 1``` command in the pipeline definition.

    ```json linenums="1" hl_lines="9 48 31"
    {
        "run_id": "selfish-pasteur-0559",
        "dag_hash": "",
        "use_cached": false,
        "tag": "",
        "original_run_id": "",
        "status": "SUCCESS",
        "steps": {
            "step 1": {
                "name": "step 1",
                "internal_name": "step 1",
                "status": "FAIL",
                "step_type": "task",
                "message": "",
                "mock": false,
                "code_identities": [
                    {
                        "code_identifier": "d76cf865af2f8e03b6c1205403351cbe42e6cdc4",
                        "code_identifier_type": "git",
                        "code_identifier_dependable": true,
                        "code_identifier_url": "https://github.com/AstraZeneca/runnable-core.git",
                        "code_identifier_message": ""
                    }
                ],
                "attempts": [
                    {
                        "attempt_number": 1,
                        "start_time": "2024-02-11 05:59:08.382587",
                        "end_time": "2024-02-11 05:59:08.446642",
                        "duration": "0:00:00.064055",
                        "status": "FAIL",
                        "message": "Command failed",
                        "parameters": {}
                    }
                ],
                "user_defined_metrics": {},
                "branches": {},
                "data_catalog": [
                    {
                        "name": "step_1.execution.log",
                        "data_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                        "catalog_relative_path": "selfish-pasteur-0559/step_1.execution.log",
                        "catalog_handler_location": ".catalog",
                        "stage": "put"
                    }
                ]
            },
            "step 3": {
                "name": "step 3",
                "internal_name": "step 3",
                "status": "SUCCESS",
                "step_type": "stub",
                "message": "",
                "mock": false,
                "code_identities": [
                    {
                        "code_identifier": "d76cf865af2f8e03b6c1205403351cbe42e6cdc4",
                        "code_identifier_type": "git",
                        "code_identifier_dependable": true,
                        "code_identifier_url": "https://github.com/AstraZeneca/runnable-core.git",
                        "code_identifier_message": ""
                    }
                ],
                "attempts": [
                    {
                        "attempt_number": 1,
                        "start_time": "2024-02-11 05:59:08.516318",
                        "end_time": "2024-02-11 05:59:08.516333",
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
            "success": {
                "name": "success",
                "internal_name": "success",
                "status": "SUCCESS",
                "step_type": "success",
                "message": "",
                "mock": false,
                "code_identities": [
                    {
                        "code_identifier": "d76cf865af2f8e03b6c1205403351cbe42e6cdc4",
                        "code_identifier_type": "git",
                        "code_identifier_dependable": true,
                        "code_identifier_url": "https://github.com/AstraZeneca/runnable-core.git",
                        "code_identifier_message": ""
                    }
                ],
                "attempts": [
                    {
                        "attempt_number": 1,
                        "start_time": "2024-02-11 05:59:08.580478",
                        "end_time": "2024-02-11 05:59:08.580555",
                        "duration": "0:00:00.000077",
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
            "run_id": "selfish-pasteur-0559",
            "variables": {},
            "use_cached": false,
            "original_run_id": "",
            "dag": {
                "start_at": "step 1",
                "name": "",
                "description": "",
                "steps": {
                    "step 1": {
                        "type": "task",
                        "name": "step 1",
                        "next": "step 2",
                        "on_failure": "step 3",
                        "overrides": {},
                        "catalog": null,
                        "max_attempts": 1,
                        "command": "exit 1",
                        "command_type": "shell",
                        "node_name": "step 1"
                    },
                    "step 2": {
                        "type": "stub",
                        "name": "step 2",
                        "next": "step 3",
                        "on_failure": "",
                        "overrides": {},
                        "catalog": null,
                        "max_attempts": 1
                    },
                    "step 3": {
                        "type": "stub",
                        "name": "step 3",
                        "next": "success",
                        "on_failure": "",
                        "overrides": {},
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


=== "Mocked configuration"

    We can patch the command of step 1 to be successful to test the behavior of traversal in case
    of no failures.

    Running the pipeline with mocked configuration:

    for yaml: ```runnable execute -f examples/on-failure.yaml -c examples/configs/mocked-config-unittest.yaml```

    for python: ```runnable_CONFIGURATION_FILE=examples/configs/mocked-config-unittest.yaml python examples/on_failure.py```

    ```yaml linenums="1"
    --8<-- "examples/configs/mocked-config-unittest.yaml"
    ```


=== "Run log with mocking"

    As seen in the ```run log```, the steps have ```step 1```, ```step 2```, ```step 3``` as
    executed and successful steps. And the status of ```step 1``` is ```SUCCESS```.

    ```json linenums="1" hl_lines="9 12 48 79"
    {
        "run_id": "syrupy-aryabhata-0552",
        "dag_hash": "026b36dd2b3507fe586f1f85ba308f817745c465",
        "use_cached": false,
        "tag": "",
        "original_run_id": "",
        "status": "SUCCESS",
        "steps": {
            "step 1": {
                "name": "step 1",
                "internal_name": "step 1",
                "status": "SUCCESS",
                "step_type": "task",
                "message": "",
                "mock": false,
                "code_identities": [
                    {
                        "code_identifier": "d76cf865af2f8e03b6c1205403351cbe42e6cdc4",
                        "code_identifier_type": "git",
                        "code_identifier_dependable": true,
                        "code_identifier_url": "https://github.com/AstraZeneca/runnable-core.git",
                        "code_identifier_message": ""
                    }
                ],
                "attempts": [
                    {
                        "attempt_number": 1,
                        "start_time": "2024-02-11 05:52:19.421358",
                        "end_time": "2024-02-11 05:52:19.426678",
                        "duration": "0:00:00.005320",
                        "status": "SUCCESS",
                        "message": "",
                        "parameters": {}
                    }
                ],
                "user_defined_metrics": {},
                "branches": {},
                "data_catalog": [
                    {
                        "name": "step_1.execution.log",
                        "data_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                        "catalog_relative_path": "syrupy-aryabhata-0552/step_1.execution.log",
                        "catalog_handler_location": ".catalog",
                        "stage": "put"
                    }
                ]
            },
            "step 2": {
                "name": "step 2",
                "internal_name": "step 2",
                "status": "SUCCESS",
                "step_type": "stub",
                "message": "",
                "mock": true,
                "code_identities": [
                    {
                        "code_identifier": "d76cf865af2f8e03b6c1205403351cbe42e6cdc4",
                        "code_identifier_type": "git",
                        "code_identifier_dependable": true,
                        "code_identifier_url": "https://github.com/AstraZeneca/runnable-core.git",
                        "code_identifier_message": ""
                    }
                ],
                "attempts": [
                    {
                        "attempt_number": 1,
                        "start_time": "2024-02-11 05:52:19.500544",
                        "end_time": "2024-02-11 05:52:19.500559",
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
            "step 3": {
                "name": "step 3",
                "internal_name": "step 3",
                "status": "SUCCESS",
                "step_type": "stub",
                "message": "",
                "mock": true,
                "code_identities": [
                    {
                        "code_identifier": "d76cf865af2f8e03b6c1205403351cbe42e6cdc4",
                        "code_identifier_type": "git",
                        "code_identifier_dependable": true,
                        "code_identifier_url": "https://github.com/AstraZeneca/runnable-core.git",
                        "code_identifier_message": ""
                    }
                ],
                "attempts": [
                    {
                        "attempt_number": 1,
                        "start_time": "2024-02-11 05:52:19.577734",
                        "end_time": "2024-02-11 05:52:19.577749",
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
            "success": {
                "name": "success",
                "internal_name": "success",
                "status": "SUCCESS",
                "step_type": "success",
                "message": "",
                "mock": false,
                "code_identities": [
                    {
                        "code_identifier": "d76cf865af2f8e03b6c1205403351cbe42e6cdc4",
                        "code_identifier_type": "git",
                        "code_identifier_dependable": true,
                        "code_identifier_url": "https://github.com/AstraZeneca/runnable-core.git",
                        "code_identifier_message": ""
                    }
                ],
                "attempts": [
                    {
                        "attempt_number": 1,
                        "start_time": "2024-02-11 05:52:19.649764",
                        "end_time": "2024-02-11 05:52:19.650318",
                        "duration": "0:00:00.000554",
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
                "service_name": "mocked",
                "service_type": "executor",
                "enable_parallel": false,
                "overrides": {},
                "patches": {
                    "step 1": {
                        "command": "exit 0"
                    }
                }
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
            "pipeline_file": "examples/on-failure.yaml",
            "parameters_file": null,
            "configuration_file": "examples/configs/mocked-config-unittest.yaml",
            "tag": "",
            "run_id": "syrupy-aryabhata-0552",
            "variables": {},
            "use_cached": false,
            "original_run_id": "",
            "dag": {
                "start_at": "step 1",
                "name": "",
                "description": "This is a simple pipeline to demonstrate failure in a step.\n\nThe default behavior is to traverse to step type fail and mark the run as
    failed.\nBut you can control it by providing on_failure.\n\nIn this example: step 1 fails and moves to step 3 skipping step 2. The pipeline status\nis considered to be
    success.\n\nstep 1 (FAIL) >> step 3 >> success\n\nYou can run this pipeline by runnable execute -f examples/on-failure.yaml\n",
                "steps": {
                    "step 1": {
                        "type": "task",
                        "name": "step 1",
                        "next": "step 2",
                        "on_failure": "step 3",
                        "overrides": {},
                        "catalog": null,
                        "max_attempts": 1,
                        "command_type": "shell",
                        "command": "exit 1",
                        "node_name": "step 1"
                    },
                    "step 2": {
                        "type": "stub",
                        "name": "step 2",
                        "next": "step 3",
                        "on_failure": "",
                        "overrides": {},
                        "catalog": null,
                        "max_attempts": 1
                    },
                    "step 3": {
                        "type": "stub",
                        "name": "step 3",
                        "next": "success",
                        "on_failure": "",
                        "overrides": {},
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
            "dag_hash": "026b36dd2b3507fe586f1f85ba308f817745c465",
            "execution_plan": "chained"
        }
    }
    ```





### Debugging failed executions

!!! tip "Using debuggers"

    For pipelines defined by the python SDK, you can create breakpoints at the
    python function being executed and use [debuggers](https://docs.python.org/3/library/pdb.html).

    For ```notebook``` based tasks,
    refer to [ploomber engine documentation](https://engine.ploomber.io/en/docs/user-guide/debugging/debuglater.html) for rich debuggers.

    Shell commands can be run in isolation by providing the parameters as environment variables
    and catalog artifacts present in the ```compute_data_folder``` location.

To debug a failed execution, we can use the mocked executor to mock all the steps except
for the failed step and providing the parameters and data exposed to the step during the
failure which are captured by the ```run log``` and ```catalog```.

=== "Faulty pipeline"

    ```yaml linenums="1"
    --8<-- "examples/retry-fail.yaml"
    ```

=== "Faulty run log"

    ```json linenums="1"
    {
        "run_id": "wrong-file-name",
        "dag_hash": "7b12d64874eff2072c9dd97912a17149f2c32ed2",
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
                        "code_identifier": "d76cf865af2f8e03b6c1205403351cbe42e6cdc4",
                        "code_identifier_type": "git",
                        "code_identifier_dependable": true,
                        "code_identifier_url": "https://github.com/AstraZeneca/runnable-core.git",
                        "code_identifier_message": ""
                    }
                ],
                "attempts": [
                    {
                        "attempt_number": 1,
                        "start_time": "2024-02-11 23:03:00.417889",
                        "end_time": "2024-02-11 23:03:00.429579",
                        "duration": "0:00:00.011690",
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
                        "data_hash": "d2dd9105fa3c62c35d89182c44fbd1ec992d8d408e38f0350d582fa29ed88074",
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
                        "code_identifier": "d76cf865af2f8e03b6c1205403351cbe42e6cdc4",
                        "code_identifier_type": "git",
                        "code_identifier_dependable": true,
                        "code_identifier_url": "https://github.com/AstraZeneca/runnable-core.git",
                        "code_identifier_message": ""
                    }
                ],
                "attempts": [
                    {
                        "attempt_number": 1,
                        "start_time": "2024-02-11 23:03:00.507067",
                        "end_time": "2024-02-11 23:03:00.514757",
                        "duration": "0:00:00.007690",
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
                        "data_hash": "d2dd9105fa3c62c35d89182c44fbd1ec992d8d408e38f0350d582fa29ed88074",
                        "catalog_relative_path": "wrong-file-name/Create_Content.execution.log",
                        "catalog_handler_location": ".catalog",
                        "stage": "put"
                    },
                    {
                        "name": "data/hello.txt",
                        "data_hash": "2ac8edfe4eb5d0d9392cb070664c31c45eecca78c43cb99d2d9c6f5a8c813932",
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
                        "code_identifier": "d76cf865af2f8e03b6c1205403351cbe42e6cdc4",
                        "code_identifier_type": "git",
                        "code_identifier_dependable": true,
                        "code_identifier_url": "https://github.com/AstraZeneca/runnable-core.git",
                        "code_identifier_message": ""
                    }
                ],
                "attempts": [
                    {
                        "attempt_number": 1,
                        "start_time": "2024-02-11 23:03:00.595992",
                        "end_time": "2024-02-11 23:03:00.645752",
                        "duration": "0:00:00.049760",
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
                        "data_hash": "2ac8edfe4eb5d0d9392cb070664c31c45eecca78c43cb99d2d9c6f5a8c813932",
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
                        "code_identifier": "d76cf865af2f8e03b6c1205403351cbe42e6cdc4",
                        "code_identifier_type": "git",
                        "code_identifier_dependable": true,
                        "code_identifier_url": "https://github.com/AstraZeneca/runnable-core.git",
                        "code_identifier_message": ""
                    }
                ],
                "attempts": [
                    {
                        "attempt_number": 1,
                        "start_time": "2024-02-11 23:03:00.727316",
                        "end_time": "2024-02-11 23:03:00.727911",
                        "duration": "0:00:00.000595",
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
            "variables": {},
            "use_cached": false,
            "original_run_id": "",
            "dag": {
                "start_at": "Setup",
                "name": "",
                "description": "This is a simple pipeline that demonstrates retrying failures.\n\n1. Setup: We setup a data folder, we ignore if it is already present\n2. Create Content: We create a \"hello.txt\" and \"put\" the file in catalog\n3. Retrieve Content: We \"get\" the file \"hello.txt\" from the catalog and show the contents\n5. Cleanup: We remove the data folder. Note that this is stubbed to prevent accidental deletion.\n\n\nYou can run this pipeline by:\n   runnable execute -f examples/retry-fail.yaml -c examples/configs/fs-catalog-run_log.yaml \\\n    --run-id wrong-file-name\n",
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
            "dag_hash": "7b12d64874eff2072c9dd97912a17149f2c32ed2",
            "execution_plan": "chained"
        }
    }
    ```



=== "mocked configuration"

    ```yaml linenums="1"
    --8<-- "examples/configs/mocked-config-debug.yaml"
    ```

=== "Debugging failed executions"

    Copy the catalog during the failed execution to the debugging execution and
    retry the step. We give it a run_id ```debug-pipeline```

    cp .catalog/wrong-file-name debug-pipeline

    and retry with the fix:

    ```runnable execute -f examples/retry-fail.yaml -c examples/configs/mocked-config-debug.yaml
    --run-id debug-pipeline```
