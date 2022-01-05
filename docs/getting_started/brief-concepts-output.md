# Closer look at output

---

While the dag defines the work that has to be done, it is only a piece of the whole puzzle.

As clearly explained in [this paper by Sculley et al.](https://papers.nips.cc/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf), 
the actual machine learning/data science related code is only fraction of all the systems that have to work together to make it work.

We implemented magnus with a clear understanding of the complexity while keeping the interface to the data scientists/ML researchers
as simple as possible.

---

Though the example pipeline we just ran did nothing useful, it helps in understanding the different *systems* in place.


``` json
{
    "dag_hash": "b2f3284a59b0097184f6f95d55b8f0be94694319",
    "original_run_id": null,
    "parameters": {},
    "run_id": "20210424123209_717c16",
    "status": "SUCCESS",
    "steps":{
      ...
    },
    "tag": null,
    "use_cached": false
}
```
## Tagging experiments

You can use the *tag* feature to logically group several executions, you can provide a tag at the run-time as below.

``` shell
magnus execute --file getting-started.yaml --tag example
``` 

## Enabling re-runs

All code breaks at some point and being able to replicate the exact cause of error is essential for a quick resolution. 
Magnus tracks three possible sources of changes that could have led to a different outcome of an experiment.

* dag: What was actually run as part of the experiment. The [dag_hash](../../concepts/run-log/#dag_hash) in the 
log is the SHA id of the actual dag
* code: If the code is git versioned, magnus tracks the [code commit id](../../concepts/run-log/#code_identity) 
and modified files as part of the logs. If the run is containarized, magnus also tracks the 
docker image digest as part of the log. 
* data: Any data generated as part of the nodes can be cataloged along with the 
[SHA identity](../../concepts/run-log/#data_catalog) of the file. 


You can re-run an older run by providing the run_id like so:

``` shell
magnus execute --file getting-started.yaml --run-id 20210424123209_717c16 --use-cached
``` 

which could give a log similar to:
``` json
{
    "dag_hash": "b2f3284a59b0097184f6f95d55b8f0be94694319",
    "original_run_id": "20210424123209_717c16",
    "parameters": {},
    "run_id": "20210424123209_717b32",
    "status": "SUCCESS",
    "steps":{
      ...
    },
    "tag": null,
    "use_cached": true
}
```

By comparing the two different run logs, you should be able to identify changes between them.

The [run log structure](../../concepts/run-log) of the output is exactly the same independent of where the actual run happens. This should enable 
to replicate a run that happened in an K8 environment, for example, in your local computer to debug.

## Step Log

Every step of the dag, has a [corresponding block](../../concepts/run-log/#structure_of_step_log) in the run log. 

Here is the step log for step1 of the example run

``` json

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
            "code_identities": [],
            "data_catalog": [],
            "internal_name": "step1",
            "message": null,
            "mock": false,
            "name": "step1",
            "status": "SUCCESS",
            "step_type": "as-is",
            "user_defined_metrics": {}
        }
}
```

### Attempts

Every step could be optionally retried by providing a retry field in the dag definition and every attempt would have 
and [attempt section](../../concepts/run-log/#attempts) as part of the step log.

We capture the start time, end time, duration and status of the attempt.


## Interaction in code

### Metrics

Magnus, by design, does not interfere with your ways of working. In almost all cases, you never need to ```import magnus``` in your code.

This design decision was made to enable you to try a different framework without changing your code base.

Magnus tracks any environment variable with prefix ```MAGNUS_TRACK_``` and stores them into 
[*user_defined_metrics*](../../concepts/run-log/#user_defined_metrics).
We provide a convenenience function which could be used to do the same, for example

``` python

from magnus import track_this

def my_cool_function():
  track_this(variance=0.6, alpha=0.1)

``` 

or alternatively:
``` python

import os

def my_cool_function():
  os.environ['MAGNUS_TRACK_variance'] = 0.6
  os.environ['MAGNUS_TRACK_alpha'] = 0.1

``` 

would result in the following block in the step log of the step.

```json
"user_defined_metrics": {
  "variance": 0.6,
  "alpha": 0.1
}
```

---
!!! Note

    All tracked data are case insensitive and the case is changed to lower to support Windows.
    Please read more information [here](https://stackoverflow.com/questions/19023238/why-python-uppercases-all-environment-variables-in-windows).

---


### Parameters

The same design principle of not interfering with your ways of working is applied in parameters. 

You can provide any input arguments to the functions by using [*parameters*](../../concepts/run-log/#parameters)

``` shell
magnus execute --file getting-started.yaml --arg1 hello --arg2 world
``` 

You can access parameters/arguments of a function either by:

``` python

def my_cool_function(arg1=None, arg2=None):
  pass

``` 
or 

``` python

import os

def my_cool_function():
  arg1 = os.environ['MAGNUS_PRM_arg1']
  arg2 = os.environ['MAGNUS_PRM_arg2']

``` 
or 

``` python

from magnus import get_parameter

def my_cool_function():
  arg1 = get_parameter('arg1')
  arg2 = get_parameter('arg2')

```

or 
``` python

from magnus import get_parameter

def my_cool_function():
  args = get_parameter()
  arg1 = args['arg1']
  arg2 = args['arg2']

```

Nodes can also add more parameters during execution and they are accessible by down-stream nodes in a 
[similar fashion](../../concepts/nodes/#passing_data).

---
!!! Note

    All parameters are case insensitive and the case is changed to lower to support Windows.
    Please read more information [here](https://stackoverflow.com/questions/19023238/why-python-uppercases-all-environment-variables-in-windows).

---


## Data Catalog

Similar to parameters, magnus can provision data generated from upstream nodes in the graph 
available to downstream nodes.
These data sets generated by a pipeline are cataloged and are available for debugging or 
reproducing results from previous runs.

The configuration can be provided at the pipeline definition or by interacting with the code base.

The below examples do exactly the same thing, you can choose one way depending upon the ease.


```yaml
step1:
  type: task
  command: my_cool_function
  next: step2
  catalog:
    get:
      - get_interesting_data.csv
    put:
      - put_interesting_data.csv
```

or 

```python
from magnus import get_from_catalog
from magnus import put_in_catalog

def my_cool_function():
  get_from_catalog('get_interesting_data.csv')
  

  # Do something with the data
  put_in_catalog('put_interesting_data.csv)

```

More information on the exact details and configuration is provided [here](../../concepts/catalog).

## Secrets Management

Magnus provides an interface to access secrets to the functions during execution time. The design decision of not 
importing magnus in code was violated only for security and integrity of the secrets.

An example of accessing a secret:

``` python
from magnus import get_secret

def my_cool_function():
  secret_value = get_secret('secret_name')

```
There are many ways of declaring secrets, please see [here for more information](../../concepts/secrets).