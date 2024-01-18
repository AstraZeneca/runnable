```map``` nodes in magnus allows you to execute a sequence of nodes (i.e a pipeline) for all the items in a list. This is similar to
[Map state of AWS Step functions](https://docs.aws.amazon.com/step-functions/latest/dg/amazon-states-language-map-state.html) or
[loops in Argo workflows](https://argo-workflows.readthedocs.io/en/latest/walk-through/loops/).

Conceptually, map node can be represented in python like below.

```python
for i in iterable_parameter:
    # a pipeline of steps
    execute_first_step(i)
    execute_second_step(i)
    ...
```

You can control the parallelism by configuration of the executor.

## Example

Below is an example of processing a inventory of files (50) in parallel batches of 10 files per batch.
The ```stride``` parameter controls the chunk size and every batch is given the start index
of the files to process.

=== "visualization"

    The step "chunk files" identifies the number of files to process and computes the start index of every
    batch of files to process for a chunk size of 10, the stride.

    "Process Chunk" pipelines are then triggered in parallel to process the chunk of files between ```start index```
    and ```start index + stride```

    ```mermaid
    flowchart TD
    chunkify([Chunk files]):::green
    success([Success]):::green

    subgraph one[Process Chunk]
        process_chunk1([Process Chunk]):::yellow
        success_chunk1([Success]):::yellow

        process_chunk1 --> success_chunk1
    end

    subgraph two[Process Chunk]
        process_chunk2([Process Chunk]):::yellow
        success_chunk2([Success]):::yellow

        process_chunk2 --> success_chunk2
    end

    subgraph three[Process Chunk]
        process_chunk3([Process Chunk]):::yellow
        success_chunk3([Success]):::yellow

        process_chunk3 --> success_chunk3
    end

    subgraph four[Process Chunk]
        process_chunk4([Process Chunk]):::yellow
        success_chunk4([Success]):::yellow

        process_chunk4 --> success_chunk4
    end

    subgraph five[Process Chunk]
        process_chunk5([Process Chunk]):::yellow
        success_chunk5([Success]):::yellow

        process_chunk5 --> success_chunk5
    end



    chunkify -- (stride=10, start_index=0)--> one --> success
    chunkify -- (stride=10, start_index=10)--> two --> success
    chunkify -- (stride=10, start_index=20)--> three --> success
    chunkify -- (stride=10, start_index=30)--> four --> success
    chunkify -- (stride=10, start_index=40)--> five --> success

    classDef yellow stroke:#FFFF00
    classDef green stroke:#0f0
    ```

=== "python sdk"

    The ```start_index``` argument for the function ```process_chunk``` is dynamically set by iterating
    over ```chunks```.

    If the argument ```start_index``` is not provided, you can still access the current
    value by ```MAGNUS_MAP_VARIABLE``` environment variable.
    The environment variable ```MAGNUS_MAP_VARIABLE``` is a dictionary with keys as
    ```iterate_as```

    This instruction is set while defining the map node.

    ```python linenums="1" hl_lines="21 52-58"
    --8<-- "examples/concepts/map.py"
    ```


=== "pipeline in yaml"

    The ```start_index``` argument for the function ```process_chunk``` is dynamically set by iterating
    over ```chunks```.

    This instruction is set while defining the map node.
    Note that the ```branch``` of the map node has a similar schema of the pipeline.

    ```yaml linenums="1" hl_lines="22-23 25-36"
    --8<-- "examples/concepts/map.yaml"
    ```

=== "pipeline with shell tasks"

    The task ```chunk files``` sets the parameters ```stride``` and ```chunks``` similar to the python
    functions.

    The map branch "iterate and execute" iterates over chunks and exposes the current start_index of
    as environment variable ```MAGNUS_MAP_VARIABLE```.

    The environment variable ```MAGNUS_MAP_VARIABLE``` is a json string with keys of the ```iterate_as```.

    ```yaml linenums="1" hl_lines="23-24 38-40"
    --8<-- "examples/concepts/map_shell.yaml"
    ```

=== "Run log"

    The step log of the ```iterate and execute``` has branches for every dynamically executed branch
    of the format ```iterate and execute.<iterate_as value>```.

    ```json
    {
        "run_id": "simple-turing-0153",
        "dag_hash": "",
        "use_cached": false,
        "tag": "",
        "original_run_id": "",
        "status": "SUCCESS",
        "steps": {
            "chunk files": {
                "name": "chunk files",
                "internal_name": "chunk files",
                "status": "SUCCESS",
                "step_type": "task",
                "message": "",
                "mock": false,
                "code_identities": [
                    {
                        "code_identifier": "30ca73bb01ac45db08b1ca75460029da142b53fa",
                        "code_identifier_type": "git",
                        "code_identifier_dependable": true,
                        "code_identifier_url": "https://github.com/AstraZeneca/magnus-core.git",
                        "code_identifier_message": ""
                    }
                ],
                "attempts": [
                    {
                        "attempt_number": 1,
                        "start_time": "2024-01-18 01:54:00.038461",
                        "end_time": "2024-01-18 01:54:00.045343",
                        "duration": "0:00:00.006882",
                        "status": "SUCCESS",
                        "message": "",
                        "parameters": {}
                    }
                ],
                "user_defined_metrics": {},
                "branches": {},
                "data_catalog": [
                    {
                        "name": "chunk_files.execution.log",
                        "data_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                        "catalog_relative_path": "simple-turing-0153/chunk_files.execution.log",
                        "catalog_handler_location": ".catalog",
                        "stage": "put"
                    }
                ]
            },
            "iterate and execute": {
                "name": "iterate and execute",
                "internal_name": "iterate and execute",
                "status": "SUCCESS",
                "step_type": "map",
                "message": "",
                "mock": false,
                "code_identities": [
                    {
                        "code_identifier": "30ca73bb01ac45db08b1ca75460029da142b53fa",
                        "code_identifier_type": "git",
                        "code_identifier_dependable": true,
                        "code_identifier_url": "https://github.com/AstraZeneca/magnus-core.git",
                        "code_identifier_message": ""
                    }
                ],
                "attempts": [],
                "user_defined_metrics": {},
                "branches": {
                    "iterate and execute.0": {
                        "internal_name": "iterate and execute.0",
                        "status": "SUCCESS",
                        "steps": {
                            "iterate and execute.0.execute": {
                                "name": "execute",
                                "internal_name": "iterate and execute.0.execute",
                                "status": "SUCCESS",
                                "step_type": "task",
                                "message": "",
                                "mock": false,
                                "code_identities": [
                                    {
                                        "code_identifier": "30ca73bb01ac45db08b1ca75460029da142b53fa",
                                        "code_identifier_type": "git",
                                        "code_identifier_dependable": true,
                                        "code_identifier_url": "https://github.com/AstraZeneca/magnus-core.git",
                                        "code_identifier_message": ""
                                    }
                                ],
                                "attempts": [
                                    {
                                        "attempt_number": 1,
                                        "start_time": "2024-01-18 01:54:00.221240",
                                        "end_time": "2024-01-18 01:54:00.222560",
                                        "duration": "0:00:00.001320",
                                        "status": "SUCCESS",
                                        "message": "",
                                        "parameters": {
                                            "chunks": [
                                                0,
                                                10,
                                                20,
                                                30,
                                                40
                                            ],
                                            "stride": 10
                                        }
                                    }
                                ],
                                "user_defined_metrics": {},
                                "branches": {},
                                "data_catalog": [
                                    {
                                        "name": "execute.execution.log_0",
                                        "data_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                                        "catalog_relative_path": "simple-turing-0153/execute.execution.log_0",
                                        "catalog_handler_location": ".catalog",
                                        "stage": "put"
                                    }
                                ]
                            },
                            "iterate and execute.0.success": {
                                "name": "success",
                                "internal_name": "iterate and execute.0.success",
                                "status": "SUCCESS",
                                "step_type": "success",
                                "message": "",
                                "mock": false,
                                "code_identities": [
                                    {
                                        "code_identifier": "30ca73bb01ac45db08b1ca75460029da142b53fa",
                                        "code_identifier_type": "git",
                                        "code_identifier_dependable": true,
                                        "code_identifier_url": "https://github.com/AstraZeneca/magnus-core.git",
                                        "code_identifier_message": ""
                                    }
                                ],
                                "attempts": [
                                    {
                                        "attempt_number": 1,
                                        "start_time": "2024-01-18 01:54:00.301335",
                                        "end_time": "2024-01-18 01:54:00.302161",
                                        "duration": "0:00:00.000826",
                                        "status": "SUCCESS",
                                        "message": "",
                                        "parameters": {
                                            "chunks": [
                                                0,
                                                10,
                                                20,
                                                30,
                                                40
                                            ],
                                            "stride": 10
                                        }
                                    }
                                ],
                                "user_defined_metrics": {},
                                "branches": {},
                                "data_catalog": []
                            }
                        }
                    },
                    "iterate and execute.10": {
                        "internal_name": "iterate and execute.10",
                        "status": "SUCCESS",
                        "steps": {
                            "iterate and execute.10.execute": {
                                "name": "execute",
                                "internal_name": "iterate and execute.10.execute",
                                "status": "SUCCESS",
                                "step_type": "task",
                                "message": "",
                                "mock": false,
                                "code_identities": [
                                    {
                                        "code_identifier": "30ca73bb01ac45db08b1ca75460029da142b53fa",
                                        "code_identifier_type": "git",
                                        "code_identifier_dependable": true,
                                        "code_identifier_url": "https://github.com/AstraZeneca/magnus-core.git",
                                        "code_identifier_message": ""
                                    }
                                ],
                                "attempts": [
                                    {
                                        "attempt_number": 1,
                                        "start_time": "2024-01-18 01:54:00.396194",
                                        "end_time": "2024-01-18 01:54:00.397462",
                                        "duration": "0:00:00.001268",
                                        "status": "SUCCESS",
                                        "message": "",
                                        "parameters": {
                                            "chunks": [
                                                0,
                                                10,
                                                20,
                                                30,
                                                40
                                            ],
                                            "stride": 10
                                        }
                                    }
                                ],
                                "user_defined_metrics": {},
                                "branches": {},
                                "data_catalog": [
                                    {
                                        "name": "execute.execution.log_10",
                                        "data_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                                        "catalog_relative_path": "simple-turing-0153/execute.execution.log_10",
                                        "catalog_handler_location": ".catalog",
                                        "stage": "put"
                                    }
                                ]
                            },
                            "iterate and execute.10.success": {
                                "name": "success",
                                "internal_name": "iterate and execute.10.success",
                                "status": "SUCCESS",
                                "step_type": "success",
                                "message": "",
                                "mock": false,
                                "code_identities": [
                                    {
                                        "code_identifier": "30ca73bb01ac45db08b1ca75460029da142b53fa",
                                        "code_identifier_type": "git",
                                        "code_identifier_dependable": true,
                                        "code_identifier_url": "https://github.com/AstraZeneca/magnus-core.git",
                                        "code_identifier_message": ""
                                    }
                                ],
                                "attempts": [
                                    {
                                        "attempt_number": 1,
                                        "start_time": "2024-01-18 01:54:00.469211",
                                        "end_time": "2024-01-18 01:54:00.470266",
                                        "duration": "0:00:00.001055",
                                        "status": "SUCCESS",
                                        "message": "",
                                        "parameters": {
                                            "chunks": [
                                                0,
                                                10,
                                                20,
                                                30,
                                                40
                                            ],
                                            "stride": 10
                                        }
                                    }
                                ],
                                "user_defined_metrics": {},
                                "branches": {},
                                "data_catalog": []
                            }
                        }
                    },
                    "iterate and execute.20": {
                        "internal_name": "iterate and execute.20",
                        "status": "SUCCESS",
                        "steps": {
                            "iterate and execute.20.execute": {
                                "name": "execute",
                                "internal_name": "iterate and execute.20.execute",
                                "status": "SUCCESS",
                                "step_type": "task",
                                "message": "",
                                "mock": false,
                                "code_identities": [
                                    {
                                        "code_identifier": "30ca73bb01ac45db08b1ca75460029da142b53fa",
                                        "code_identifier_type": "git",
                                        "code_identifier_dependable": true,
                                        "code_identifier_url": "https://github.com/AstraZeneca/magnus-core.git",
                                        "code_identifier_message": ""
                                    }
                                ],
                                "attempts": [
                                    {
                                        "attempt_number": 1,
                                        "start_time": "2024-01-18 01:54:00.558053",
                                        "end_time": "2024-01-18 01:54:00.561472",
                                        "duration": "0:00:00.003419",
                                        "status": "SUCCESS",
                                        "message": "",
                                        "parameters": {
                                            "chunks": [
                                                0,
                                                10,
                                                20,
                                                30,
                                                40
                                            ],
                                            "stride": 10
                                        }
                                    }
                                ],
                                "user_defined_metrics": {},
                                "branches": {},
                                "data_catalog": [
                                    {
                                        "name": "execute.execution.log_20",
                                        "data_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                                        "catalog_relative_path": "simple-turing-0153/execute.execution.log_20",
                                        "catalog_handler_location": ".catalog",
                                        "stage": "put"
                                    }
                                ]
                            },
                            "iterate and execute.20.success": {
                                "name": "success",
                                "internal_name": "iterate and execute.20.success",
                                "status": "SUCCESS",
                                "step_type": "success",
                                "message": "",
                                "mock": false,
                                "code_identities": [
                                    {
                                        "code_identifier": "30ca73bb01ac45db08b1ca75460029da142b53fa",
                                        "code_identifier_type": "git",
                                        "code_identifier_dependable": true,
                                        "code_identifier_url": "https://github.com/AstraZeneca/magnus-core.git",
                                        "code_identifier_message": ""
                                    }
                                ],
                                "attempts": [
                                    {
                                        "attempt_number": 1,
                                        "start_time": "2024-01-18 01:54:00.660092",
                                        "end_time": "2024-01-18 01:54:00.661215",
                                        "duration": "0:00:00.001123",
                                        "status": "SUCCESS",
                                        "message": "",
                                        "parameters": {
                                            "chunks": [
                                                0,
                                                10,
                                                20,
                                                30,
                                                40
                                            ],
                                            "stride": 10
                                        }
                                    }
                                ],
                                "user_defined_metrics": {},
                                "branches": {},
                                "data_catalog": []
                            }
                        }
                    },
                    "iterate and execute.30": {
                        "internal_name": "iterate and execute.30",
                        "status": "SUCCESS",
                        "steps": {
                            "iterate and execute.30.execute": {
                                "name": "execute",
                                "internal_name": "iterate and execute.30.execute",
                                "status": "SUCCESS",
                                "step_type": "task",
                                "message": "",
                                "mock": false,
                                "code_identities": [
                                    {
                                        "code_identifier": "30ca73bb01ac45db08b1ca75460029da142b53fa",
                                        "code_identifier_type": "git",
                                        "code_identifier_dependable": true,
                                        "code_identifier_url": "https://github.com/AstraZeneca/magnus-core.git",
                                        "code_identifier_message": ""
                                    }
                                ],
                                "attempts": [
                                    {
                                        "attempt_number": 1,
                                        "start_time": "2024-01-18 01:54:00.765689",
                                        "end_time": "2024-01-18 01:54:00.766705",
                                        "duration": "0:00:00.001016",
                                        "status": "SUCCESS",
                                        "message": "",
                                        "parameters": {
                                            "chunks": [
                                                0,
                                                10,
                                                20,
                                                30,
                                                40
                                            ],
                                            "stride": 10
                                        }
                                    }
                                ],
                                "user_defined_metrics": {},
                                "branches": {},
                                "data_catalog": [
                                    {
                                        "name": "execute.execution.log_30",
                                        "data_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                                        "catalog_relative_path": "simple-turing-0153/execute.execution.log_30",
                                        "catalog_handler_location": ".catalog",
                                        "stage": "put"
                                    }
                                ]
                            },
                            "iterate and execute.30.success": {
                                "name": "success",
                                "internal_name": "iterate and execute.30.success",
                                "status": "SUCCESS",
                                "step_type": "success",
                                "message": "",
                                "mock": false,
                                "code_identities": [
                                    {
                                        "code_identifier": "30ca73bb01ac45db08b1ca75460029da142b53fa",
                                        "code_identifier_type": "git",
                                        "code_identifier_dependable": true,
                                        "code_identifier_url": "https://github.com/AstraZeneca/magnus-core.git",
                                        "code_identifier_message": ""
                                    }
                                ],
                                "attempts": [
                                    {
                                        "attempt_number": 1,
                                        "start_time": "2024-01-18 01:54:00.851112",
                                        "end_time": "2024-01-18 01:54:00.852454",
                                        "duration": "0:00:00.001342",
                                        "status": "SUCCESS",
                                        "message": "",
                                        "parameters": {
                                            "chunks": [
                                                0,
                                                10,
                                                20,
                                                30,
                                                40
                                            ],
                                            "stride": 10
                                        }
                                    }
                                ],
                                "user_defined_metrics": {},
                                "branches": {},
                                "data_catalog": []
                            }
                        }
                    },
                    "iterate and execute.40": {
                        "internal_name": "iterate and execute.40",
                        "status": "SUCCESS",
                        "steps": {
                            "iterate and execute.40.execute": {
                                "name": "execute",
                                "internal_name": "iterate and execute.40.execute",
                                "status": "SUCCESS",
                                "step_type": "task",
                                "message": "",
                                "mock": false,
                                "code_identities": [
                                    {
                                        "code_identifier": "30ca73bb01ac45db08b1ca75460029da142b53fa",
                                        "code_identifier_type": "git",
                                        "code_identifier_dependable": true,
                                        "code_identifier_url": "https://github.com/AstraZeneca/magnus-core.git",
                                        "code_identifier_message": ""
                                    }
                                ],
                                "attempts": [
                                    {
                                        "attempt_number": 1,
                                        "start_time": "2024-01-18 01:54:00.950911",
                                        "end_time": "2024-01-18 01:54:00.952000",
                                        "duration": "0:00:00.001089",
                                        "status": "SUCCESS",
                                        "message": "",
                                        "parameters": {
                                            "chunks": [
                                                0,
                                                10,
                                                20,
                                                30,
                                                40
                                            ],
                                            "stride": 10
                                        }
                                    }
                                ],
                                "user_defined_metrics": {},
                                "branches": {},
                                "data_catalog": [
                                    {
                                        "name": "execute.execution.log_40",
                                        "data_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                                        "catalog_relative_path": "simple-turing-0153/execute.execution.log_40",
                                        "catalog_handler_location": ".catalog",
                                        "stage": "put"
                                    }
                                ]
                            },
                            "iterate and execute.40.success": {
                                "name": "success",
                                "internal_name": "iterate and execute.40.success",
                                "status": "SUCCESS",
                                "step_type": "success",
                                "message": "",
                                "mock": false,
                                "code_identities": [
                                    {
                                        "code_identifier": "30ca73bb01ac45db08b1ca75460029da142b53fa",
                                        "code_identifier_type": "git",
                                        "code_identifier_dependable": true,
                                        "code_identifier_url": "https://github.com/AstraZeneca/magnus-core.git",
                                        "code_identifier_message": ""
                                    }
                                ],
                                "attempts": [
                                    {
                                        "attempt_number": 1,
                                        "start_time": "2024-01-18 01:54:01.032790",
                                        "end_time": "2024-01-18 01:54:01.034254",
                                        "duration": "0:00:00.001464",
                                        "status": "SUCCESS",
                                        "message": "",
                                        "parameters": {
                                            "chunks": [
                                                0,
                                                10,
                                                20,
                                                30,
                                                40
                                            ],
                                            "stride": 10
                                        }
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
            "success": {
                "name": "success",
                "internal_name": "success",
                "status": "SUCCESS",
                "step_type": "success",
                "message": "",
                "mock": false,
                "code_identities": [
                    {
                        "code_identifier": "30ca73bb01ac45db08b1ca75460029da142b53fa",
                        "code_identifier_type": "git",
                        "code_identifier_dependable": true,
                        "code_identifier_url": "https://github.com/AstraZeneca/magnus-core.git",
                        "code_identifier_message": ""
                    }
                ],
                "attempts": [
                    {
                        "attempt_number": 1,
                        "start_time": "2024-01-18 01:54:01.141928",
                        "end_time": "2024-01-18 01:54:01.142928",
                        "duration": "0:00:00.001000",
                        "status": "SUCCESS",
                        "message": "",
                        "parameters": {
                            "chunks": [
                                0,
                                10,
                                20,
                                30,
                                40
                            ],
                            "stride": 10
                        }
                    }
                ],
                "user_defined_metrics": {},
                "branches": {},
                "data_catalog": []
            }
        },
        "parameters": {
            "chunks": [
                0,
                10,
                20,
                30,
                40
            ],
            "stride": 10
        },
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
            "run_id": "simple-turing-0153",
            "variables": {},
            "use_cached": false,
            "original_run_id": "",
            "dag": {
                "start_at": "chunk files",
                "name": "",
                "description": "",
                "steps": {
                    "chunk files": {
                        "type": "task",
                        "name": "chunk files",
                        "next": "iterate and execute",
                        "on_failure": "",
                        "executor_config": {},
                        "catalog": null,
                        "max_attempts": 1,
                        "command": "examples.concepts.map.chunk_files",
                        "node_name": "chunk files"
                    },
                    "iterate and execute": {
                        "type": "map",
                        "name": "iterate and execute",
                        "is_composite": true,
                        "next": "success",
                        "on_failure": "",
                        "executor_config": {},
                        "iterate_on": "chunks",
                        "iterate_as": "start_index",
                        "branch": {
                            "start_at": "execute",
                            "name": "",
                            "description": "",
                            "steps": {
                                "execute": {
                                    "type": "task",
                                    "name": "execute",
                                    "next": "success",
                                    "on_failure": "",
                                    "executor_config": {},
                                    "catalog": null,
                                    "max_attempts": 1,
                                    "command": "examples.concepts.map.process_chunk",
                                    "node_name": "execute"
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


## Traversal

A branch of a map step is considered success only if the ```success``` step is reached at the end.
The steps of the pipeline can fail and be handled by [on failure](../concepts/ppiline/on_failure) and
redirected to ```success``` if that is the desired behavior.

The map step is considered successful only if all the branches of the step have terminated successfully.


## Parameters

All the tasks defined in the branches of the map pipeline can
[access to parameters and data as usual](../task).


!!! warning

    The parameters can be updated by all the tasks and the last task to execute overwrites
    the previous changes.

    Since the order of execution is not guaranteed, its best to avoid mutating the same parameters in
    the steps belonging to map step.
