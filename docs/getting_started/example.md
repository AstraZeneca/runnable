# Example Run

---
!!! Note

    It is assumed that you have gone through installation and magnus command line works.

---

To give you a flavour of how magnus works, lets create a simple pipeline.

Copy the contents of this yaml into getting-started.yaml or alternatively in a python file if you are using the SDK.

---
!!! Note

    The below execution would create a folder called 'data' in the current working directory.
    The command as given should work in linux/macOS but for windows, please change accordingly.
---

``` yaml
dag:
  description: Getting started
  start_at: step parameters
  steps:
    step parameters:
      type: task
      command_type: python-lambda
      command: "lambda x: {'x': int(x) + 1}"
      next: step shell
    step shell:
      type: task
      command_type: shell
      command: mkdir data ; env >> data/data.txt # For Linux/macOS
      next: success
      catalog:
        put:
          - "*"
    success:
      type: success
    fail:
      type: fail
```

The same could also be defined via a Python SDK.

```python

#in pipeline.py
from magnus import Pipeline, Task

def pipeline():
    first = Task(name='step parameters', command="lambda x: {'x': int(x) + 1}", command_type='python-lambda',
                next_node='step shell')
    second = Task(name='step shell', command='mkdir data ; env >> data/data.txt',
                  command_type='shell', catalog={'put': '*'})

    pipeline = Pipeline(start_at=first, name='getting_started')
    pipeline.construct([first, second])
    pipeline.execute(parameters_file='parameters.yaml')

if __name__ == '__main__':
    pipeline()

```

Since the pipeline expects a parameter ```x```, lets provide that using ```parameters.yaml```

```yaml
x: 3
```


And let's run the pipeline using:
``` shell
 magnus execute --file getting-started.yaml --parameters-file parameters.yaml
```

If you are using the python SDK:

```
poetry run python pipeline.py
```

You should see a list of warnings but your terminal output should look something similar to this:

``` json
{
    "run_id": "20230131195647",
    "dag_hash": "",
    "use_cached": false,
    "tag": "",
    "original_run_id": "",
    "status": "SUCCESS",
    "steps": {
        "step parameters": {
            "name": "step parameters",
            "internal_name": "step parameters",
            "status": "SUCCESS",
            "step_type": "task",
            "message": "",
            "mock": false,
            "code_identities": [
                {
                    "code_identifier": "e15d1374aac217f649972d11fe772e61b5a2478d",
                    "code_identifier_type": "git",
                    "code_identifier_dependable": true,
                    "code_identifier_url": "INTENTIONALLY REMOVED",
                    "code_identifier_message": ""
                }
            ],
            "attempts": [
                {
                    "attempt_number": 0,
                    "start_time": "2023-01-31 19:56:55.007931",
                    "end_time": "2023-01-31 19:56:55.009273",
                    "duration": "0:00:00.001342",
                    "status": "SUCCESS",
                    "message": ""
                }
            ],
            "user_defined_metrics": {},
            "branches": {},
            "data_catalog": []
        },
        "step shell": {
            "name": "step shell",
            "internal_name": "step shell",
            "status": "SUCCESS",
            "step_type": "task",
            "message": "",
            "mock": false,
            "code_identities": [
                {
                    "code_identifier": "e15d1374aac217f649972d11fe772e61b5a2478d",
                    "code_identifier_type": "git",
                    "code_identifier_dependable": true,
                    "code_identifier_url": "INTENTIONALLY REMOVED",
                    "code_identifier_message": ""
                }
            ],
            "attempts": [
                {
                    "attempt_number": 0,
                    "start_time": "2023-01-31 19:56:55.128697",
                    "end_time": "2023-01-31 19:56:55.150878",
                    "duration": "0:00:00.022181",
                    "status": "SUCCESS",
                    "message": ""
                }
            ],
            "user_defined_metrics": {},
            "branches": {},
            "data_catalog": [
                {
                    "name": "data/data.txt",
                    "data_hash": "7e91b0a9ff8841a3b5bf2c711f58bcc0cbb6a7f85b9bc92aa65e78cdda59a96e",
                    "catalog_relative_path": "20230131195647/data/data.txt",
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
                    "code_identifier": "e15d1374aac217f649972d11fe772e61b5a2478d",
                    "code_identifier_type": "git",
                    "code_identifier_dependable": true,
                    "code_identifier_url": "INTENTIONALLY REMOVED",
                    "code_identifier_message": ""
                }
            ],
            "attempts": [
                {
                    "attempt_number": 0,
                    "start_time": "2023-01-31 19:56:55.239877",
                    "end_time": "2023-01-31 19:56:55.240116",
                    "duration": "0:00:00.000239",
                    "status": "SUCCESS",
                    "message": ""
                }
            ],
            "user_defined_metrics": {},
            "branches": {},
            "data_catalog": []
        }
    },
    "parameters": {
        "x": 4
    },
    "run_config": {
        "executor": {
            "type": "local",
            "config": {
                "enable_parallel": false,
                "placeholders": {}
            }
        },
        "run_log_store": {
            "type": "buffered",
            "config": {}
        },
        "catalog": {
            "type": "file-system",
            "config": {
                "compute_data_folder": "data",
                "catalog_location": ".catalog"
            }
        },
        "secrets": {
            "type": "do-nothing",
            "config": {}
        },
        "experiment_tracker": {
            "type": "do-nothing",
            "config": {}
        },
        "variables": {},
        "pipeline": {
            "start_at": "step parameters",
            "name": "getting_started",
            "description": "",
            "max_time": 86400,
            "steps": {
                "step parameters": {
                    "mode_config": {},
                    "next_node": "step shell",
                    "command": "lambda x: {'x': int(x) + 1}",
                    "command_type": "python-lambda",
                    "command_config": {},
                    "catalog": {},
                    "retry": 1,
                    "on_failure": "",
                    "type": "task"
                },
                "step shell": {
                    "mode_config": {},
                    "next_node": "success",
                    "command": "mkdir data ; env >> data/data.txt",
                    "command_type": "shell",
                    "command_config": {},
                    "catalog": {
                        "put": "*"
                    },
                    "retry": 1,
                    "on_failure": "",
                    "type": "task"
                },
                "success": {
                    "mode_config": {},
                    "type": "success"
                },
                "fail": {
                    "mode_config": {},
                    "type": "fail"
                }
            }
        }
    }
}
```

You should see that ```data``` folder being created with a file called ```data.txt``` in it.
This is according to the command in ```step shell```.

You should also see a folder ```.catalog``` being created with a single folder corresponding to the run_id of this run.

Let's take a closer look at the input and output in the next sections.
