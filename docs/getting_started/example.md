# Example Run

---
!!! Note

    It is assumed that you have gone through installation and magnus command line works.

---

To give you a flavour of how magnus works, lets create a simple pipeline.

Copy the contents of this yaml into getting-started.yaml.

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

Since the pipeline expects a parameter ```x```, lets provide that using ```parameters.yaml```

```yaml
x: 3
```


And let's run the pipeline using:
``` shell
 magnus execute --file getting-started.yaml --parameters-file parameters.yaml
```

You should see a list of warnings but your terminal output should look something similar to this:

``` json
{
    "run_id": "20220118114608",
    "dag_hash": "ce0676d63e99c34848484f2df1744bab8d45e33a",
    "use_cached": false,
    "tag": null,
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
                    "code_identifier": "c5d2f4aa8dd354740d1b2f94b6ee5c904da5e63c",
                    "code_identifier_type": "git",
                    "code_identifier_dependable": false,
                    "code_identifier_url": "<INTENTIONALLY REMOVED>",
                    "code_identifier_message": "<INTENTIONALLY REMOVED>"
                }
            ],
            "attempts": [
                {
                    "attempt_number": 0,
                    "start_time": "2022-01-18 11:46:08.530138",
                    "end_time": "2022-01-18 11:46:08.530561",
                    "duration": "0:00:00.000423",
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
                    "code_identifier": "c5d2f4aa8dd354740d1b2f94b6ee5c904da5e63c",
                    "code_identifier_type": "git",
                    "code_identifier_dependable": false,
                    "code_identifier_url": "<INTENTIONALLY REMOVED>",
                    "code_identifier_message": "<INTENTIONALLY REMOVED>"
                }
            ],
            "attempts": [
                {
                    "attempt_number": 0,
                    "start_time": "2022-01-18 11:46:08.576522",
                    "end_time": "2022-01-18 11:46:08.588158",
                    "duration": "0:00:00.011636",
                    "status": "SUCCESS",
                    "message": ""
                }
            ],
            "user_defined_metrics": {},
            "branches": {},
            "data_catalog": [
                {
                    "name": "data.txt",
                    "data_hash": "8f25ba24e56f182c5125b9ede73cab6c16bf193e3ad36b75ba5145ff1b5db583",
                    "catalog_relative_path": "20220118114608/data.txt",
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
                    "code_identifier": "c5d2f4aa8dd354740d1b2f94b6ee5c904da5e63c",
                    "code_identifier_type": "git",
                    "code_identifier_dependable": false,
                    "code_identifier_url": "<INTENTIONALLY REMOVED>",
                    "code_identifier_message": "<INTENTIONALLY REMOVED>"
                }
            ],
            "attempts": [
                {
                    "attempt_number": 0,
                    "start_time": "2022-01-18 11:46:08.639563",
                    "end_time": "2022-01-18 11:46:08.639680",
                    "duration": "0:00:00.000117",
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
            "config": {}
        },
        "run_log_store": {
            "type": "buffered",
            "config": {}
        },
        "catalog": {
            "type": "file-system",
            "config": {}
        },
        "secrets": {
            "type": "do-nothing",
            "config": {}
        }
    }
}
```

You should see that ```data``` folder being created with a file called ```data.txt``` in it.
This is according to the command in ```step shell```.

You should also see a folder ```.catalog``` being created with a single folder corresponding to the run_id of this run.

Let's take a closer look at the input and output in the next sections.
