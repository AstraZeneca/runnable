# Hello from magnus

Magnus is a data science pipeline definition and execution tool. It provides a way to:

- Define a pipeline steps and the flow.
- Run the pipeline in any environment, local is default.
- Store the run metadata and data catalogs and re-run in case of failures.

Once the pipeline is proven to be correct and functional in any environment, there is zero code change
required to deploy it elsewhere. The behavior of the runs are identical in all environments. Magnus
is not a queuing or scheduling engine, but delegates that responsibility to chosen deployment patterns.

### Short Summary

Magnus provides four capabilities for data teams:

- **Compute execution plan**: A DAG representation of work that you want to get done. Individual nodes of the DAG
could be simple python or shell tasks or complex deeply nested parallel branches or embedded DAGs themselves.

- **Run log store**: A place to store run logs for reporting or re-running older runs. Along with capturing the
status of execution,  the run logs also capture code identifiers (commits, docker image digests etc), data hashes and
configuration settings for reproducibility and audit.

- **Data Catalogs**: A way to pass data between nodes of the graph during execution and also serves the purpose of
versioning the data used by a particular run.

- **Secrets**: A framework to provide secrets/credentials at run time to the nodes of the graph.

### Design decisions:

- **Easy to extend**: All the four capabilities are just definitions and can be implemented in many flavors.

    - **Compute execution plan**: You can choose to run the DAG on your local computer, in containers of local computer
    or off load the work to cloud providers or translate the DAG to AWS step functions or Argo workflows.

    - **Run log Store**: The actual implementation of storing the run logs could be in-memory, file system, S3,
    database etc.

    - **Data Catalogs**: The data files generated as part of a run could be stored on file-systems, S3 or could be
    extended to fit your needs.

    - **Secrets**: The secrets needed for your code to work could be in dotenv, AWS or extended to fit your needs.

- **Pipeline as contract**: Once a DAG is defined and proven to work in local or some environment, there is absolutely
no code change needed to deploy it to other environments. This enables the data teams to prove the correctness of
the dag in dev environments while infrastructure teams to find the suitable way to deploy it.

- **Reproducibility**: Run log store and data catalogs hold the version, code commits, data files used for a run
making it easy to re-run an older run or debug a failed run. Debug environment need not be the same as
original environment.

- **Easy switch**: Your infrastructure landscape changes over time. With magnus, you can switch infrastructure
by just changing a config and not code.


Magnus does not aim to replace existing and well constructed orchestrators like AWS Step functions or
[argo](https://argoproj.github.io/workflows/) but complements them in a unified, simple and intuitive way.

## Documentation

[More details about the project and how to use it available here](https://astrazeneca.github.io/magnus-core/).

## Installation

### pip

magnus is a python package and should be installed as any other.

```shell
pip install magnus
```

# Example Run

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

To understand more about the input and output, please head over to the
[documentation](https://project-magnus.github.io/magnus-core/).
