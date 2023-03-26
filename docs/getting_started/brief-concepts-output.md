# Closer look at output

---

While the dag defines the work that has to be done, it is only a piece of the whole puzzle.

As clearly explained in [this paper by Sculley et al.](https://papers.nips.cc/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf),
the actual machine learning/data science related code is only fraction of all the systems that have to be in place
to make it work.

We implemented magnus with a clear understanding of the complexity while keeping the interface to the
data scientists/ML researchers as simple as possible.

---

Though the example pipeline we just ran did nothing useful, it helps in understanding the different *systems* in place.

--8<--
README.md:exampleOutput
--8<--

## Run id

Every run of magnus has a unique identifier called run_id. Magnus by default creates one based on timestamp but you
can provide one at run time for better control.

```magnus execute --file getting-started.yaml --run-id my_first --x 3```

## Reproducibility

All code breaks at some point and being able to replicate the exact cause of error is essential for a quick resolution.
Magnus tracks four possible sources of changes that could have led to a different outcome of an experiment.

* dag: The [dag_hash](../../concepts/run-log/#dag_hash) in the log is the SHA id of the actual dag.
* code: If the code is git versioned, magnus tracks the [code commit id](../../concepts/run-log/#code_identity)
and modified files as part of the logs. If the run is containerized, magnus also tracks the
docker image digest as part of the log.
* data: Any data generated as part of the nodes can be cataloged along with the
[SHA identity](../../concepts/run-log/#data_catalog) of the file.
* config: The run config used to make the run is also stored as part of the run logs along with the pipeline definition.


The [run log structure](../../concepts/run-log) of the output is exactly the same independent of where the
actual run happens. This should enable to replicate a run that happened in an K8 environment,
for example, in your local computer to debug.

## Step Log

Every step of the dag, has a [corresponding block](../../concepts/run-log/#structure_of_step_log) in the run log. The
name of the ```step``` is name of key in ```steps```.

Here is the step log for ```step shell``` of the example run

``` json
"steps": {
    ...,
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
    ...
}
```

## Attempts

As part of the attempt, we capture the start time, end time and the duration of the execution. Only task, success, fail
and as-is nodes have this block as it refers to the actual compute time used. In case of failure, magnus tries to
capture the exception message in the ```message``` block.

## Code identity

The git SHA id of the [code commit]((../../concepts/run-log/#code_identity)) is captured,
if the code is versioned using git. If the current branch was unclean,
magnus will warn the user about the dependability of the code id and lists the files that are different from the commit.

If the execution was in a container, magnus also adds the docker image digest as a code identity along with git sha id.

## Data catalog

```Step shell``` of the example run creates a file ```data.txt``` as part of the run in the data folder. As per the
configuration of the pipeline, we have instructed magnus to store all (*) contents of the ```data``` folder for
downstream steps using the catalog. The data catalog section of the step log captures the hash of the data and the
metadata related to it.

You can read more about catalog [here]((../../concepts/run-log/#data_catalog)).
