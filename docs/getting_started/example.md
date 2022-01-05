# Example Run

---
!!! Note

    It is assumed that you have gone through installation and magnus command line works.

---

To give you a flavour of how magnus works, lets create a simple pipeline.

Copy the contents of this yaml into getting-started.yaml.

``` yaml
dag:
  description: Getting started
  start_at: step1 
  steps:
    step1:
      type: as-is
      next: step2
    step2:
      type: success
    step3:
      type: fail
```

And let's run the pipeline using:
``` shell
magnus execute --file getting-started.yaml
```

You should see a lot of logs along with a [*Run Log*](../../concepts/run-log) in your terminal similar to this:

``` json
{
    "dag_hash": "b2f3284a59b0097184f6f95d55b8f0be94694319",
    "original_run_id": null,
    "parameters": {},
    "run_id": "20210424123209_717c16",
    "status": "SUCCESS",
    "steps": {
        "step1": {
            "attempts": [
                {
                    "attempt_numner": 0,
                    "duration": "0:00:00.000018",
                    "end_time": "2021-04-24 12:32:09.787580",
                    "message": "",
                    "start_time": "2021-04-24 12:32:09.787541",
                    "status": "SUCCESS"
                }
            ],
            "branches": {},
            "code_identities": [
                {
                    "code_identifer_dependable": false,
                    "code_identifier": "c223668de72afe98253bd6895b6b3389bc8099e5",
                    "code_identifier_message": "changes found in <Intentially removed>",
                    "code_identifier_type": "git",
                    "code_identifier_url": "INTENTIONALLY REMOVED, POINTS TO GIT REMOTE"
                }
            ],
            "data_catalog": [],
            "internal_name": "step1",
            "message": null,
            "mock": false,
            "name": "step1",
            "status": "SUCCESS",
            "step_type": "as-is",
            "user_defined_metrics": {}
        },
        "step2": {
            "attempts": [
                {
                    "attempt_numner": 0,
                    "duration": "0:00:00.000018",
                    "end_time": "2021-04-24 12:32:09.854140",
                    "message": "",
                    "start_time": "2021-04-24 12:32:09.854028",
                    "status": "SUCCESS"
                }
            ],
            "branches": {},
            "code_identities": [
                {
                    "code_identifer_dependable": false,
                    "code_identifier": "c223668de72afe98253bd6895b6b3389bc8099e5",
                    "code_identifier_message": "changes found in <Intentially removed>",
                    "code_identifier_type": "git",
                    "code_identifier_url": "INTENTIONALLY REMOVED, POINTS TO GIT REMOTE"
                }
            ],
            "data_catalog": [],
            "internal_name": "step2",
            "message": null,
            "mock": false,
            "name": "step2",
            "status": "SUCCESS",
            "step_type": "success",
            "user_defined_metrics": {}
        }
    },
    "tag": null,
    "use_cached": false
}
```

# ✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨
Congratulations! you just ran your first pipeline.

Now, lets take a step back and break down what happened.