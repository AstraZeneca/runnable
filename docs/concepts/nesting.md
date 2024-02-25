As seen from the definitions of [parallel](../concepts/parallel.md) or
[map](../concepts/map.md), the branches are pipelines
themselves. This allows for deeply nested workflows in **runnable**.

Technically there is no limit in the depth of nesting but there are some practical considerations.


- Not all workflow engines that runnable can transpile the workflow to support deeply nested workflows.
AWS Step functions and Argo workflows support them.

- Deeply nested workflows are complex to understand and debug during errors.


## Example


=== "python sdk"


    You can run this pipeline by ```python examples/concepts/nesting.py```

    ```python linenums="1"
    --8<-- "examples/concepts/nesting.py"
    ```

=== "yaml"

    You can run this pipeline by ```runnable execute examples/concepts/nesting.yaml```

    ```yaml linenums="1"
    --8<-- "examples/concepts/nesting.yaml"
    ```

=== "Run log"

    <details>
    <summary>Click to expand!</summary>

    ```json
    {
        "run_id": "bipartite-neumann-1913",
        "dag_hash": "",
        "use_cached": false,
        "tag": "",
        "original_run_id": "",
        "status": "SUCCESS",
        "steps": {
            "generate list": {
                "name": "generate list",
                "internal_name": "generate list",
                "status": "SUCCESS",
                "step_type": "task",
                "message": "",
                "mock": false,
                "code_identities": [
                    {
                        "code_identifier": "99139c3507898c60932ad5d35c08b395399a19f6",
                        "code_identifier_type": "git",
                        "code_identifier_dependable": true,
                        "code_identifier_url": "https://github.com/AstraZeneca/runnable-core.git",
                        "code_identifier_message": ""
                    }
                ],
                "attempts": [
                    {
                        "attempt_number": 1,
                        "start_time": "2024-01-18 19:13:49.748656",
                        "end_time": "2024-01-18 19:13:49.756826",
                        "duration": "0:00:00.008170",
                        "status": "SUCCESS",
                        "message": "",
                        "parameters": {}
                    }
                ],
                "user_defined_metrics": {},
                "branches": {},
                "data_catalog": [
                    {
                        "name": "generate_list.execution.log",
                        "data_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                        "catalog_relative_path": "bipartite-neumann-1913/generate_list.execution.log",
                        "catalog_handler_location": ".catalog",
                        "stage": "put"
                    }
                ]
            },
            "outer most": {
                "name": "outer most",
                "internal_name": "outer most",
                "status": "SUCCESS",
                "step_type": "map",
                "message": "",
                "mock": false,
                "code_identities": [
                    {
                        "code_identifier": "99139c3507898c60932ad5d35c08b395399a19f6",
                        "code_identifier_type": "git",
                        "code_identifier_dependable": true,
                        "code_identifier_url": "https://github.com/AstraZeneca/runnable-core.git",
                        "code_identifier_message": ""
                    }
                ],
                "attempts": [],
                "user_defined_metrics": {},
                "branches": {
                    "outer most.0": {
                        "internal_name": "outer most.0",
                        "status": "SUCCESS",
                        "steps": {
                            "outer most.0.nested parallel": {
                                "name": "nested parallel",
                                "internal_name": "outer most.0.nested parallel",
                                "status": "SUCCESS",
                                "step_type": "parallel",
                                "message": "",
                                "mock": false,
                                "code_identities": [
                                    {
                                        "code_identifier": "99139c3507898c60932ad5d35c08b395399a19f6",
                                        "code_identifier_type": "git",
                                        "code_identifier_dependable": true,
                                        "code_identifier_url": "https://github.com/AstraZeneca/runnable-core.git",
                                        "code_identifier_message": ""
                                    }
                                ],
                                "attempts": [],
                                "user_defined_metrics": {},
                                "branches": {
                                    "outer most.0.nested parallel.a": {
                                        "internal_name": "outer most.0.nested parallel.a",
                                        "status": "SUCCESS",
                                        "steps": {
                                            "outer most.0.nested parallel.a.inner most": {
                                                "name": "inner most",
                                                "internal_name": "outer most.0.nested parallel.a.inner most",
                                                "status": "SUCCESS",
                                                "step_type": "map",
                                                "message": "",
                                                "mock": false,
                                                "code_identities": [
                                                    {
                                                        "code_identifier": "99139c3507898c60932ad5d35c08b395399a19f6",
                                                        "code_identifier_type": "git",
                                                        "code_identifier_dependable": true,
                                                        "code_identifier_url": "https://github.com/AstraZeneca/runnable-core.git",
                                                        "code_identifier_message": ""
                                                    }
                                                ],
                                                "attempts": [],
                                                "user_defined_metrics": {},
                                                "branches": {
                                                    "outer most.0.nested parallel.a.inner most.0": {
                                                        "internal_name": "outer most.0.nested parallel.a.inner most.0",
                                                        "status": "SUCCESS",
                                                        "steps": {
                                                            "outer most.0.nested parallel.a.inner most.0.executable": {
                                                                "name": "executable",
                                                                "internal_name": "outer most.0.nested parallel.a.inner most.0.executable",
                                                                "status": "SUCCESS",
                                                                "step_type": "stub",
                                                                "message": "",
                                                                "mock": false,
                                                                "code_identities": [
                                                                    {
                                                                        "code_identifier": "99139c3507898c60932ad5d35c08b395399a19f6",
                                                                        "code_identifier_type": "git",
                                                                        "code_identifier_dependable": true,
                                                                        "code_identifier_url": "https://github.com/AstraZeneca/runnable-core.git",
                                                                        "code_identifier_message": ""
                                                                    }
                                                                ],
                                                                "attempts": [
                                                                    {
                                                                        "attempt_number": 1,
                                                                        "start_time": "2024-01-18 19:13:49.997158",
                                                                        "end_time": "2024-01-18 19:13:49.997172",
                                                                        "duration": "0:00:00.000014",
                                                                        "status": "SUCCESS",
                                                                        "message": "",
                                                                        "parameters": {
                                                                            "array": [
                                                                                0,
                                                                                1
                                                                            ]
                                                                        }
                                                                    }
                                                                ],
                                                                "user_defined_metrics": {},
                                                                "branches": {},
                                                                "data_catalog": []
                                                            },
                                                            "outer most.0.nested parallel.a.inner most.0.success": {
                                                                "name": "success",
                                                                "internal_name": "outer most.0.nested parallel.a.inner most.0.success",
                                                                "status": "SUCCESS",
                                                                "step_type": "success",
                                                                "message": "",
                                                                "mock": false,
                                                                "code_identities": [
                                                                    {
                                                                        "code_identifier": "99139c3507898c60932ad5d35c08b395399a19f6",
                                                                        "code_identifier_type": "git",
                                                                        "code_identifier_dependable": true,
                                                                        "code_identifier_url": "https://github.com/AstraZeneca/runnable-core.git",
                                                                        "code_identifier_message": ""
                                                                    }
                                                                ],
                                                                "attempts": [
                                                                    {
                                                                        "attempt_number": 1,
                                                                        "start_time": "2024-01-18 19:13:50.060734",
                                                                        "end_time": "2024-01-18 19:13:50.061345",
                                                                        "duration": "0:00:00.000611",
                                                                        "status": "SUCCESS",
                                                                        "message": "",
                                                                        "parameters": {
                                                                            "array": [
                                                                                0,
                                                                                1
                                                                            ]
                                                                        }
                                                                    }
                                                                ],
                                                                "user_defined_metrics": {},
                                                                "branches": {},
                                                                "data_catalog": []
                                                            }
                                                        }
                                                    },
                                                    "outer most.0.nested parallel.a.inner most.1": {
                                                        "internal_name": "outer most.0.nested parallel.a.inner most.1",
                                                        "status": "SUCCESS",
                                                        "steps": {
                                                            "outer most.0.nested parallel.a.inner most.1.executable": {
                                                                "name": "executable",
                                                                "internal_name": "outer most.0.nested parallel.a.inner most.1.executable",
                                                                "status": "SUCCESS",
                                                                "step_type": "stub",
                                                                "message": "",
                                                                "mock": false,
                                                                "code_identities": [
                                                                    {
                                                                        "code_identifier": "99139c3507898c60932ad5d35c08b395399a19f6",
                                                                        "code_identifier_type": "git",
                                                                        "code_identifier_dependable": true,
                                                                        "code_identifier_url": "https://github.com/AstraZeneca/runnable-core.git",
                                                                        "code_identifier_message": ""
                                                                    }
                                                                ],
                                                                "attempts": [
                                                                    {
                                                                        "attempt_number": 1,
                                                                        "start_time": "2024-01-18 19:13:50.131067",
                                                                        "end_time": "2024-01-18 19:13:50.131078",
                                                                        "duration": "0:00:00.000011",
                                                                        "status": "SUCCESS",
                                                                        "message": "",
                                                                        "parameters": {
                                                                            "array": [
                                                                                0,
                                                                                1
                                                                            ]
                                                                        }
                                                                    }
                                                                ],
                                                                "user_defined_metrics": {},
                                                                "branches": {},
                                                                "data_catalog": []
                                                            },
                                                            "outer most.0.nested parallel.a.inner most.1.success": {
                                                                "name": "success",
                                                                "internal_name": "outer most.0.nested parallel.a.inner most.1.success",
                                                                "status": "SUCCESS",
                                                                "step_type": "success",
                                                                "message": "",
                                                                "mock": false,
                                                                "code_identities": [
                                                                    {
                                                                        "code_identifier": "99139c3507898c60932ad5d35c08b395399a19f6",
                                                                        "code_identifier_type": "git",
                                                                        "code_identifier_dependable": true,
                                                                        "code_identifier_url": "https://github.com/AstraZeneca/runnable-core.git",
                                                                        "code_identifier_message": ""
                                                                    }
                                                                ],
                                                                "attempts": [
                                                                    {
                                                                        "attempt_number": 1,
                                                                        "start_time": "2024-01-18 19:13:50.194038",
                                                                        "end_time": "2024-01-18 19:13:50.194978",
                                                                        "duration": "0:00:00.000940",
                                                                        "status": "SUCCESS",
                                                                        "message": "",
                                                                        "parameters": {
                                                                            "array": [
                                                                                0,
                                                                                1
                                                                            ]
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
                                            "outer most.0.nested parallel.a.success": {
                                                "name": "success",
                                                "internal_name": "outer most.0.nested parallel.a.success",
                                                "status": "SUCCESS",
                                                "step_type": "success",
                                                "message": "",
                                                "mock": false,
                                                "code_identities": [
                                                    {
                                                        "code_identifier": "99139c3507898c60932ad5d35c08b395399a19f6",
                                                        "code_identifier_type": "git",
                                                        "code_identifier_dependable": true,
                                                        "code_identifier_url": "https://github.com/AstraZeneca/runnable-core.git",
                                                        "code_identifier_message": ""
                                                    }
                                                ],
                                                "attempts": [
                                                    {
                                                        "attempt_number": 1,
                                                        "start_time": "2024-01-18 19:13:50.263302",
                                                        "end_time": "2024-01-18 19:13:50.264215",
                                                        "duration": "0:00:00.000913",
                                                        "status": "SUCCESS",
                                                        "message": "",
                                                        "parameters": {
                                                            "array": [
                                                                0,
                                                                1
                                                            ]
                                                        }
                                                    }
                                                ],
                                                "user_defined_metrics": {},
                                                "branches": {},
                                                "data_catalog": []
                                            }
                                        }
                                    },
                                    "outer most.0.nested parallel.b": {
                                        "internal_name": "outer most.0.nested parallel.b",
                                        "status": "SUCCESS",
                                        "steps": {
                                            "outer most.0.nested parallel.b.inner most": {
                                                "name": "inner most",
                                                "internal_name": "outer most.0.nested parallel.b.inner most",
                                                "status": "SUCCESS",
                                                "step_type": "map",
                                                "message": "",
                                                "mock": false,
                                                "code_identities": [
                                                    {
                                                        "code_identifier": "99139c3507898c60932ad5d35c08b395399a19f6",
                                                        "code_identifier_type": "git",
                                                        "code_identifier_dependable": true,
                                                        "code_identifier_url": "https://github.com/AstraZeneca/runnable-core.git",
                                                        "code_identifier_message": ""
                                                    }
                                                ],
                                                "attempts": [],
                                                "user_defined_metrics": {},
                                                "branches": {
                                                    "outer most.0.nested parallel.b.inner most.0": {
                                                        "internal_name": "outer most.0.nested parallel.b.inner most.0",
                                                        "status": "SUCCESS",
                                                        "steps": {
                                                            "outer most.0.nested parallel.b.inner most.0.executable": {
                                                                "name": "executable",
                                                                "internal_name": "outer most.0.nested parallel.b.inner most.0.executable",
                                                                "status": "SUCCESS",
                                                                "step_type": "stub",
                                                                "message": "",
                                                                "mock": false,
                                                                "code_identities": [
                                                                    {
                                                                        "code_identifier": "99139c3507898c60932ad5d35c08b395399a19f6",
                                                                        "code_identifier_type": "git",
                                                                        "code_identifier_dependable": true,
                                                                        "code_identifier_url": "https://github.com/AstraZeneca/runnable-core.git",
                                                                        "code_identifier_message": ""
                                                                    }
                                                                ],
                                                                "attempts": [
                                                                    {
                                                                        "attempt_number": 1,
                                                                        "start_time": "2024-01-18 19:13:50.402511",
                                                                        "end_time": "2024-01-18 19:13:50.402525",
                                                                        "duration": "0:00:00.000014",
                                                                        "status": "SUCCESS",
                                                                        "message": "",
                                                                        "parameters": {
                                                                            "array": [
                                                                                0,
                                                                                1
                                                                            ]
                                                                        }
                                                                    }
                                                                ],
                                                                "user_defined_metrics": {},
                                                                "branches": {},
                                                                "data_catalog": []
                                                            },
                                                            "outer most.0.nested parallel.b.inner most.0.success": {
                                                                "name": "success",
                                                                "internal_name": "outer most.0.nested parallel.b.inner most.0.success",
                                                                "status": "SUCCESS",
                                                                "step_type": "success",
                                                                "message": "",
                                                                "mock": false,
                                                                "code_identities": [
                                                                    {
                                                                        "code_identifier": "99139c3507898c60932ad5d35c08b395399a19f6",
                                                                        "code_identifier_type": "git",
                                                                        "code_identifier_dependable": true,
                                                                        "code_identifier_url": "https://github.com/AstraZeneca/runnable-core.git",
                                                                        "code_identifier_message": ""
                                                                    }
                                                                ],
                                                                "attempts": [
                                                                    {
                                                                        "attempt_number": 1,
                                                                        "start_time": "2024-01-18 19:13:50.468196",
                                                                        "end_time": "2024-01-18 19:13:50.469218",
                                                                        "duration": "0:00:00.001022",
                                                                        "status": "SUCCESS",
                                                                        "message": "",
                                                                        "parameters": {
                                                                            "array": [
                                                                                0,
                                                                                1
                                                                            ]
                                                                        }
                                                                    }
                                                                ],
                                                                "user_defined_metrics": {},
                                                                "branches": {},
                                                                "data_catalog": []
                                                            }
                                                        }
                                                    },
                                                    "outer most.0.nested parallel.b.inner most.1": {
                                                        "internal_name": "outer most.0.nested parallel.b.inner most.1",
                                                        "status": "SUCCESS",
                                                        "steps": {
                                                            "outer most.0.nested parallel.b.inner most.1.executable": {
                                                                "name": "executable",
                                                                "internal_name": "outer most.0.nested parallel.b.inner most.1.executable",
                                                                "status": "SUCCESS",
                                                                "step_type": "stub",
                                                                "message": "",
                                                                "mock": false,
                                                                "code_identities": [
                                                                    {
                                                                        "code_identifier": "99139c3507898c60932ad5d35c08b395399a19f6",
                                                                        "code_identifier_type": "git",
                                                                        "code_identifier_dependable": true,
                                                                        "code_identifier_url": "https://github.com/AstraZeneca/runnable-core.git",
                                                                        "code_identifier_message": ""
                                                                    }
                                                                ],
                                                                "attempts": [
                                                                    {
                                                                        "attempt_number": 1,
                                                                        "start_time": "2024-01-18 19:13:50.543884",
                                                                        "end_time": "2024-01-18 19:13:50.543896",
                                                                        "duration": "0:00:00.000012",
                                                                        "status": "SUCCESS",
                                                                        "message": "",
                                                                        "parameters": {
                                                                            "array": [
                                                                                0,
                                                                                1
                                                                            ]
                                                                        }
                                                                    }
                                                                ],
                                                                "user_defined_metrics": {},
                                                                "branches": {},
                                                                "data_catalog": []
                                                            },
                                                            "outer most.0.nested parallel.b.inner most.1.success": {
                                                                "name": "success",
                                                                "internal_name": "outer most.0.nested parallel.b.inner most.1.success",
                                                                "status": "SUCCESS",
                                                                "step_type": "success",
                                                                "message": "",
                                                                "mock": false,
                                                                "code_identities": [
                                                                    {
                                                                        "code_identifier": "99139c3507898c60932ad5d35c08b395399a19f6",
                                                                        "code_identifier_type": "git",
                                                                        "code_identifier_dependable": true,
                                                                        "code_identifier_url": "https://github.com/AstraZeneca/runnable-core.git",
                                                                        "code_identifier_message": ""
                                                                    }
                                                                ],
                                                                "attempts": [
                                                                    {
                                                                        "attempt_number": 1,
                                                                        "start_time": "2024-01-18 19:13:50.610499",
                                                                        "end_time": "2024-01-18 19:13:50.611839",
                                                                        "duration": "0:00:00.001340",
                                                                        "status": "SUCCESS",
                                                                        "message": "",
                                                                        "parameters": {
                                                                            "array": [
                                                                                0,
                                                                                1
                                                                            ]
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
                                            "outer most.0.nested parallel.b.success": {
                                                "name": "success",
                                                "internal_name": "outer most.0.nested parallel.b.success",
                                                "status": "SUCCESS",
                                                "step_type": "success",
                                                "message": "",
                                                "mock": false,
                                                "code_identities": [
                                                    {
                                                        "code_identifier": "99139c3507898c60932ad5d35c08b395399a19f6",
                                                        "code_identifier_type": "git",
                                                        "code_identifier_dependable": true,
                                                        "code_identifier_url": "https://github.com/AstraZeneca/runnable-core.git",
                                                        "code_identifier_message": ""
                                                    }
                                                ],
                                                "attempts": [
                                                    {
                                                        "attempt_number": 1,
                                                        "start_time": "2024-01-18 19:13:50.682749",
                                                        "end_time": "2024-01-18 19:13:50.684374",
                                                        "duration": "0:00:00.001625",
                                                        "status": "SUCCESS",
                                                        "message": "",
                                                        "parameters": {
                                                            "array": [
                                                                0,
                                                                1
                                                            ]
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
                            "outer most.0.success": {
                                "name": "success",
                                "internal_name": "outer most.0.success",
                                "status": "SUCCESS",
                                "step_type": "success",
                                "message": "",
                                "mock": false,
                                "code_identities": [
                                    {
                                        "code_identifier": "99139c3507898c60932ad5d35c08b395399a19f6",
                                        "code_identifier_type": "git",
                                        "code_identifier_dependable": true,
                                        "code_identifier_url": "https://github.com/AstraZeneca/runnable-core.git",
                                        "code_identifier_message": ""
                                    }
                                ],
                                "attempts": [
                                    {
                                        "attempt_number": 1,
                                        "start_time": "2024-01-18 19:13:50.763079",
                                        "end_time": "2024-01-18 19:13:50.763895",
                                        "duration": "0:00:00.000816",
                                        "status": "SUCCESS",
                                        "message": "",
                                        "parameters": {
                                            "array": [
                                                0,
                                                1
                                            ]
                                        }
                                    }
                                ],
                                "user_defined_metrics": {},
                                "branches": {},
                                "data_catalog": []
                            }
                        }
                    },
                    "outer most.1": {
                        "internal_name": "outer most.1",
                        "status": "SUCCESS",
                        "steps": {
                            "outer most.1.nested parallel": {
                                "name": "nested parallel",
                                "internal_name": "outer most.1.nested parallel",
                                "status": "SUCCESS",
                                "step_type": "parallel",
                                "message": "",
                                "mock": false,
                                "code_identities": [
                                    {
                                        "code_identifier": "99139c3507898c60932ad5d35c08b395399a19f6",
                                        "code_identifier_type": "git",
                                        "code_identifier_dependable": true,
                                        "code_identifier_url": "https://github.com/AstraZeneca/runnable-core.git",
                                        "code_identifier_message": ""
                                    }
                                ],
                                "attempts": [],
                                "user_defined_metrics": {},
                                "branches": {
                                    "outer most.1.nested parallel.a": {
                                        "internal_name": "outer most.1.nested parallel.a",
                                        "status": "SUCCESS",
                                        "steps": {
                                            "outer most.1.nested parallel.a.inner most": {
                                                "name": "inner most",
                                                "internal_name": "outer most.1.nested parallel.a.inner most",
                                                "status": "SUCCESS",
                                                "step_type": "map",
                                                "message": "",
                                                "mock": false,
                                                "code_identities": [
                                                    {
                                                        "code_identifier": "99139c3507898c60932ad5d35c08b395399a19f6",
                                                        "code_identifier_type": "git",
                                                        "code_identifier_dependable": true,
                                                        "code_identifier_url": "https://github.com/AstraZeneca/runnable-core.git",
                                                        "code_identifier_message": ""
                                                    }
                                                ],
                                                "attempts": [],
                                                "user_defined_metrics": {},
                                                "branches": {
                                                    "outer most.1.nested parallel.a.inner most.0": {
                                                        "internal_name": "outer most.1.nested parallel.a.inner most.0",
                                                        "status": "SUCCESS",
                                                        "steps": {
                                                            "outer most.1.nested parallel.a.inner most.0.executable": {
                                                                "name": "executable",
                                                                "internal_name": "outer most.1.nested parallel.a.inner most.0.executable",
                                                                "status": "SUCCESS",
                                                                "step_type": "stub",
                                                                "message": "",
                                                                "mock": false,
                                                                "code_identities": [
                                                                    {
                                                                        "code_identifier": "99139c3507898c60932ad5d35c08b395399a19f6",
                                                                        "code_identifier_type": "git",
                                                                        "code_identifier_dependable": true,
                                                                        "code_identifier_url": "https://github.com/AstraZeneca/runnable-core.git",
                                                                        "code_identifier_message": ""
                                                                    }
                                                                ],
                                                                "attempts": [
                                                                    {
                                                                        "attempt_number": 1,
                                                                        "start_time": "2024-01-18 19:13:50.981456",
                                                                        "end_time": "2024-01-18 19:13:50.981467",
                                                                        "duration": "0:00:00.000011",
                                                                        "status": "SUCCESS",
                                                                        "message": "",
                                                                        "parameters": {
                                                                            "array": [
                                                                                0,
                                                                                1
                                                                            ]
                                                                        }
                                                                    }
                                                                ],
                                                                "user_defined_metrics": {},
                                                                "branches": {},
                                                                "data_catalog": []
                                                            },
                                                            "outer most.1.nested parallel.a.inner most.0.success": {
                                                                "name": "success",
                                                                "internal_name": "outer most.1.nested parallel.a.inner most.0.success",
                                                                "status": "SUCCESS",
                                                                "step_type": "success",
                                                                "message": "",
                                                                "mock": false,
                                                                "code_identities": [
                                                                    {
                                                                        "code_identifier": "99139c3507898c60932ad5d35c08b395399a19f6",
                                                                        "code_identifier_type": "git",
                                                                        "code_identifier_dependable": true,
                                                                        "code_identifier_url": "https://github.com/AstraZeneca/runnable-core.git",
                                                                        "code_identifier_message": ""
                                                                    }
                                                                ],
                                                                "attempts": [
                                                                    {
                                                                        "attempt_number": 1,
                                                                        "start_time": "2024-01-18 19:13:51.045547",
                                                                        "end_time": "2024-01-18 19:13:51.046526",
                                                                        "duration": "0:00:00.000979",
                                                                        "status": "SUCCESS",
                                                                        "message": "",
                                                                        "parameters": {
                                                                            "array": [
                                                                                0,
                                                                                1
                                                                            ]
                                                                        }
                                                                    }
                                                                ],
                                                                "user_defined_metrics": {},
                                                                "branches": {},
                                                                "data_catalog": []
                                                            }
                                                        }
                                                    },
                                                    "outer most.1.nested parallel.a.inner most.1": {
                                                        "internal_name": "outer most.1.nested parallel.a.inner most.1",
                                                        "status": "SUCCESS",
                                                        "steps": {
                                                            "outer most.1.nested parallel.a.inner most.1.executable": {
                                                                "name": "executable",
                                                                "internal_name": "outer most.1.nested parallel.a.inner most.1.executable",
                                                                "status": "SUCCESS",
                                                                "step_type": "stub",
                                                                "message": "",
                                                                "mock": false,
                                                                "code_identities": [
                                                                    {
                                                                        "code_identifier": "99139c3507898c60932ad5d35c08b395399a19f6",
                                                                        "code_identifier_type": "git",
                                                                        "code_identifier_dependable": true,
                                                                        "code_identifier_url": "https://github.com/AstraZeneca/runnable-core.git",
                                                                        "code_identifier_message": ""
                                                                    }
                                                                ],
                                                                "attempts": [
                                                                    {
                                                                        "attempt_number": 1,
                                                                        "start_time": "2024-01-18 19:13:51.116489",
                                                                        "end_time": "2024-01-18 19:13:51.116501",
                                                                        "duration": "0:00:00.000012",
                                                                        "status": "SUCCESS",
                                                                        "message": "",
                                                                        "parameters": {
                                                                            "array": [
                                                                                0,
                                                                                1
                                                                            ]
                                                                        }
                                                                    }
                                                                ],
                                                                "user_defined_metrics": {},
                                                                "branches": {},
                                                                "data_catalog": []
                                                            },
                                                            "outer most.1.nested parallel.a.inner most.1.success": {
                                                                "name": "success",
                                                                "internal_name": "outer most.1.nested parallel.a.inner most.1.success",
                                                                "status": "SUCCESS",
                                                                "step_type": "success",
                                                                "message": "",
                                                                "mock": false,
                                                                "code_identities": [
                                                                    {
                                                                        "code_identifier": "99139c3507898c60932ad5d35c08b395399a19f6",
                                                                        "code_identifier_type": "git",
                                                                        "code_identifier_dependable": true,
                                                                        "code_identifier_url": "https://github.com/AstraZeneca/runnable-core.git",
                                                                        "code_identifier_message": ""
                                                                    }
                                                                ],
                                                                "attempts": [
                                                                    {
                                                                        "attempt_number": 1,
                                                                        "start_time": "2024-01-18 19:13:51.180471",
                                                                        "end_time": "2024-01-18 19:13:51.181726",
                                                                        "duration": "0:00:00.001255",
                                                                        "status": "SUCCESS",
                                                                        "message": "",
                                                                        "parameters": {
                                                                            "array": [
                                                                                0,
                                                                                1
                                                                            ]
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
                                            "outer most.1.nested parallel.a.success": {
                                                "name": "success",
                                                "internal_name": "outer most.1.nested parallel.a.success",
                                                "status": "SUCCESS",
                                                "step_type": "success",
                                                "message": "",
                                                "mock": false,
                                                "code_identities": [
                                                    {
                                                        "code_identifier": "99139c3507898c60932ad5d35c08b395399a19f6",
                                                        "code_identifier_type": "git",
                                                        "code_identifier_dependable": true,
                                                        "code_identifier_url": "https://github.com/AstraZeneca/runnable-core.git",
                                                        "code_identifier_message": ""
                                                    }
                                                ],
                                                "attempts": [
                                                    {
                                                        "attempt_number": 1,
                                                        "start_time": "2024-01-18 19:13:51.253035",
                                                        "end_time": "2024-01-18 19:13:51.254294",
                                                        "duration": "0:00:00.001259",
                                                        "status": "SUCCESS",
                                                        "message": "",
                                                        "parameters": {
                                                            "array": [
                                                                0,
                                                                1
                                                            ]
                                                        }
                                                    }
                                                ],
                                                "user_defined_metrics": {},
                                                "branches": {},
                                                "data_catalog": []
                                            }
                                        }
                                    },
                                    "outer most.1.nested parallel.b": {
                                        "internal_name": "outer most.1.nested parallel.b",
                                        "status": "SUCCESS",
                                        "steps": {
                                            "outer most.1.nested parallel.b.inner most": {
                                                "name": "inner most",
                                                "internal_name": "outer most.1.nested parallel.b.inner most",
                                                "status": "SUCCESS",
                                                "step_type": "map",
                                                "message": "",
                                                "mock": false,
                                                "code_identities": [
                                                    {
                                                        "code_identifier": "99139c3507898c60932ad5d35c08b395399a19f6",
                                                        "code_identifier_type": "git",
                                                        "code_identifier_dependable": true,
                                                        "code_identifier_url": "https://github.com/AstraZeneca/runnable-core.git",
                                                        "code_identifier_message": ""
                                                    }
                                                ],
                                                "attempts": [],
                                                "user_defined_metrics": {},
                                                "branches": {
                                                    "outer most.1.nested parallel.b.inner most.0": {
                                                        "internal_name": "outer most.1.nested parallel.b.inner most.0",
                                                        "status": "SUCCESS",
                                                        "steps": {
                                                            "outer most.1.nested parallel.b.inner most.0.executable": {
                                                                "name": "executable",
                                                                "internal_name": "outer most.1.nested parallel.b.inner most.0.executable",
                                                                "status": "SUCCESS",
                                                                "step_type": "stub",
                                                                "message": "",
                                                                "mock": false,
                                                                "code_identities": [
                                                                    {
                                                                        "code_identifier": "99139c3507898c60932ad5d35c08b395399a19f6",
                                                                        "code_identifier_type": "git",
                                                                        "code_identifier_dependable": true,
                                                                        "code_identifier_url": "https://github.com/AstraZeneca/runnable-core.git",
                                                                        "code_identifier_message": ""
                                                                    }
                                                                ],
                                                                "attempts": [
                                                                    {
                                                                        "attempt_number": 1,
                                                                        "start_time": "2024-01-18 19:13:51.399358",
                                                                        "end_time": "2024-01-18 19:13:51.399368",
                                                                        "duration": "0:00:00.000010",
                                                                        "status": "SUCCESS",
                                                                        "message": "",
                                                                        "parameters": {
                                                                            "array": [
                                                                                0,
                                                                                1
                                                                            ]
                                                                        }
                                                                    }
                                                                ],
                                                                "user_defined_metrics": {},
                                                                "branches": {},
                                                                "data_catalog": []
                                                            },
                                                            "outer most.1.nested parallel.b.inner most.0.success": {
                                                                "name": "success",
                                                                "internal_name": "outer most.1.nested parallel.b.inner most.0.success",
                                                                "status": "SUCCESS",
                                                                "step_type": "success",
                                                                "message": "",
                                                                "mock": false,
                                                                "code_identities": [
                                                                    {
                                                                        "code_identifier": "99139c3507898c60932ad5d35c08b395399a19f6",
                                                                        "code_identifier_type": "git",
                                                                        "code_identifier_dependable": true,
                                                                        "code_identifier_url": "https://github.com/AstraZeneca/runnable-core.git",
                                                                        "code_identifier_message": ""
                                                                    }
                                                                ],
                                                                "attempts": [
                                                                    {
                                                                        "attempt_number": 1,
                                                                        "start_time": "2024-01-18 19:13:51.465371",
                                                                        "end_time": "2024-01-18 19:13:51.466805",
                                                                        "duration": "0:00:00.001434",
                                                                        "status": "SUCCESS",
                                                                        "message": "",
                                                                        "parameters": {
                                                                            "array": [
                                                                                0,
                                                                                1
                                                                            ]
                                                                        }
                                                                    }
                                                                ],
                                                                "user_defined_metrics": {},
                                                                "branches": {},
                                                                "data_catalog": []
                                                            }
                                                        }
                                                    },
                                                    "outer most.1.nested parallel.b.inner most.1": {
                                                        "internal_name": "outer most.1.nested parallel.b.inner most.1",
                                                        "status": "SUCCESS",
                                                        "steps": {
                                                            "outer most.1.nested parallel.b.inner most.1.executable": {
                                                                "name": "executable",
                                                                "internal_name": "outer most.1.nested parallel.b.inner most.1.executable",
                                                                "status": "SUCCESS",
                                                                "step_type": "stub",
                                                                "message": "",
                                                                "mock": false,
                                                                "code_identities": [
                                                                    {
                                                                        "code_identifier": "99139c3507898c60932ad5d35c08b395399a19f6",
                                                                        "code_identifier_type": "git",
                                                                        "code_identifier_dependable": true,
                                                                        "code_identifier_url": "https://github.com/AstraZeneca/runnable-core.git",
                                                                        "code_identifier_message": ""
                                                                    }
                                                                ],
                                                                "attempts": [
                                                                    {
                                                                        "attempt_number": 1,
                                                                        "start_time": "2024-01-18 19:13:51.536944",
                                                                        "end_time": "2024-01-18 19:13:51.536959",
                                                                        "duration": "0:00:00.000015",
                                                                        "status": "SUCCESS",
                                                                        "message": "",
                                                                        "parameters": {
                                                                            "array": [
                                                                                0,
                                                                                1
                                                                            ]
                                                                        }
                                                                    }
                                                                ],
                                                                "user_defined_metrics": {},
                                                                "branches": {},
                                                                "data_catalog": []
                                                            },
                                                            "outer most.1.nested parallel.b.inner most.1.success": {
                                                                "name": "success",
                                                                "internal_name": "outer most.1.nested parallel.b.inner most.1.success",
                                                                "status": "SUCCESS",
                                                                "step_type": "success",
                                                                "message": "",
                                                                "mock": false,
                                                                "code_identities": [
                                                                    {
                                                                        "code_identifier": "99139c3507898c60932ad5d35c08b395399a19f6",
                                                                        "code_identifier_type": "git",
                                                                        "code_identifier_dependable": true,
                                                                        "code_identifier_url": "https://github.com/AstraZeneca/runnable-core.git",
                                                                        "code_identifier_message": ""
                                                                    }
                                                                ],
                                                                "attempts": [
                                                                    {
                                                                        "attempt_number": 1,
                                                                        "start_time": "2024-01-18 19:13:51.602562",
                                                                        "end_time": "2024-01-18 19:13:51.604264",
                                                                        "duration": "0:00:00.001702",
                                                                        "status": "SUCCESS",
                                                                        "message": "",
                                                                        "parameters": {
                                                                            "array": [
                                                                                0,
                                                                                1
                                                                            ]
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
                                            "outer most.1.nested parallel.b.success": {
                                                "name": "success",
                                                "internal_name": "outer most.1.nested parallel.b.success",
                                                "status": "SUCCESS",
                                                "step_type": "success",
                                                "message": "",
                                                "mock": false,
                                                "code_identities": [
                                                    {
                                                        "code_identifier": "99139c3507898c60932ad5d35c08b395399a19f6",
                                                        "code_identifier_type": "git",
                                                        "code_identifier_dependable": true,
                                                        "code_identifier_url": "https://github.com/AstraZeneca/runnable-core.git",
                                                        "code_identifier_message": ""
                                                    }
                                                ],
                                                "attempts": [
                                                    {
                                                        "attempt_number": 1,
                                                        "start_time": "2024-01-18 19:13:51.676208",
                                                        "end_time": "2024-01-18 19:13:51.678050",
                                                        "duration": "0:00:00.001842",
                                                        "status": "SUCCESS",
                                                        "message": "",
                                                        "parameters": {
                                                            "array": [
                                                                0,
                                                                1
                                                            ]
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
                            "outer most.1.success": {
                                "name": "success",
                                "internal_name": "outer most.1.success",
                                "status": "SUCCESS",
                                "step_type": "success",
                                "message": "",
                                "mock": false,
                                "code_identities": [
                                    {
                                        "code_identifier": "99139c3507898c60932ad5d35c08b395399a19f6",
                                        "code_identifier_type": "git",
                                        "code_identifier_dependable": true,
                                        "code_identifier_url": "https://github.com/AstraZeneca/runnable-core.git",
                                        "code_identifier_message": ""
                                    }
                                ],
                                "attempts": [
                                    {
                                        "attempt_number": 1,
                                        "start_time": "2024-01-18 19:13:51.760988",
                                        "end_time": "2024-01-18 19:13:51.762012",
                                        "duration": "0:00:00.001024",
                                        "status": "SUCCESS",
                                        "message": "",
                                        "parameters": {
                                            "array": [
                                                0,
                                                1
                                            ]
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
                        "code_identifier": "99139c3507898c60932ad5d35c08b395399a19f6",
                        "code_identifier_type": "git",
                        "code_identifier_dependable": true,
                        "code_identifier_url": "https://github.com/AstraZeneca/runnable-core.git",
                        "code_identifier_message": ""
                    }
                ],
                "attempts": [
                    {
                        "attempt_number": 1,
                        "start_time": "2024-01-18 19:13:51.863908",
                        "end_time": "2024-01-18 19:13:51.863975",
                        "duration": "0:00:00.000067",
                        "status": "SUCCESS",
                        "message": "",
                        "parameters": {
                            "array": [
                                0,
                                1
                            ]
                        }
                    }
                ],
                "user_defined_metrics": {},
                "branches": {},
                "data_catalog": []
            }
        },
        "parameters": {
            "array": [
                0,
                1
            ]
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
            "run_id": "bipartite-neumann-1913",
            "variables": {},
            "use_cached": false,
            "original_run_id": "",
            "dag": {
                "start_at": "generate list",
                "name": "",
                "description": "",
                "steps": {
                    "generate list": {
                        "type": "task",
                        "name": "generate list",
                        "next": "outer most",
                        "on_failure": "",
                        "executor_config": {},
                        "catalog": null,
                        "max_attempts": 1,
                        "command": "examples.concepts.nesting.generate_list",
                        "node_name": "generate list"
                    },
                    "outer most": {
                        "type": "map",
                        "name": "outer most",
                        "is_composite": true,
                        "next": "success",
                        "on_failure": "",
                        "executor_config": {},
                        "iterate_on": "array",
                        "iterate_as": "x",
                        "branch": {
                            "start_at": "nested parallel",
                            "name": "",
                            "description": "",
                            "steps": {
                                "nested parallel": {
                                    "type": "parallel",
                                    "name": "nested parallel",
                                    "next": "success",
                                    "on_failure": "",
                                    "executor_config": {},
                                    "branches": {
                                        "a": {
                                            "start_at": "inner most",
                                            "name": "",
                                            "description": "",
                                            "steps": {
                                                "inner most": {
                                                    "type": "map",
                                                    "name": "inner most",
                                                    "is_composite": true,
                                                    "next": "success",
                                                    "on_failure": "",
                                                    "executor_config": {},
                                                    "iterate_on": "array",
                                                    "iterate_as": "y",
                                                    "branch": {
                                                        "start_at": "executable",
                                                        "name": "",
                                                        "description": "",
                                                        "steps": {
                                                            "executable": {
                                                                "type": "stub",
                                                                "name": "executable",
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
                                        "b": {
                                            "start_at": "inner most",
                                            "name": "",
                                            "description": "",
                                            "steps": {
                                                "inner most": {
                                                    "type": "map",
                                                    "name": "inner most",
                                                    "is_composite": true,
                                                    "next": "success",
                                                    "on_failure": "",
                                                    "executor_config": {},
                                                    "iterate_on": "array",
                                                    "iterate_as": "y",
                                                    "branch": {
                                                        "start_at": "executable",
                                                        "name": "",
                                                        "description": "",
                                                        "steps": {
                                                            "executable": {
                                                                "type": "stub",
                                                                "name": "executable",
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
    </details>
