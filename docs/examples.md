# Examples

## A single node pipeline

Every pipeline in magnus should have a ```success``` node and ```fail``` node.
The starting node of the pipeline is denoted by ```start_at``` and every node needs to define the next
node to traverse during successful execution of the current node using ```next```.

Nodes can optionally mention the node to traverse during failure using ```on_failure```.

Example:

```python
# In my_module.py

def my_function():
    print('In the function, my_function of my_module')

```


The pipeline which contains one node to call the above function.

```yaml
dag:
  description: A single node pipeline
  start_at: step 1
  steps:
    step 1:
      type: task
      next: success
      command: my_module.my_function
      command_type: python
    success:
      type: success
    failure:
      type: fail
```

---

## Mocking a node in pipeline

In magnus, you can skip execution of a node or mock using a node of type ```as-is```.
This functionality is useful when you want to focus on designing the flow of code but not the specific implementation.

Example:

```yaml
dag:
  description: A single node pipeline with mock
  start_at: step 1
  steps:
    step 1:
      type: as-is # The function would not execute as this is as-is node
      next: success
      command: my_module.my_function
      command_type: python
    success:
      type: success
    failure:
      type: fail
```

---

## Using shell commands as part of the pipeline

In magnus, a pipeline can have shell commands as part of the pipeline. The only caveat in doing so is magnus
would not be able to support returning ```parameters```, ```secrets``` or any of the built-in functions. The cataloging
functionality of magnus still would work via the configuration file.

Parameters can be accessed by looking for environment variables with a prefix of ```MAGNUS_PRM_```.

Example: Step 1 of the below pipeline would

- Get all the files from the catalog to the ```compute_data_folder```.
- Execute the command python my_module.my_function in the shell.
- Put all the files from the ```compute_data_folder``` to the catalog.

```yaml
dag:
  description: A single node pipeline with shell
  start_at: step 1
  steps:
    step 1:
      type: task
      next: success
      command: python -m my_module.my_function # You can use this to call some executable in the PATH
      command_type: shell
      catalog:
        get:
          - "*"
        put:
          - "*"
    success:
      type: success
    failure:
      type: fail
```

---
## Using python lambda expressions in pipeline

You can use python lambda expressions as a task type. Please note that you cannot have ```_``` or ```__``` as part of
the expression. This is to prevent any malicious code to be passed into the expression. In the example below,
```step 1``` takes in a parameter ```x``` and returns the integer ```x + 1```.

Example:

```yaml
dag:
  description: A single node pipeline with python lambda
  start_at: step 1
  steps:
    step 1:
      command_type: python-lambda
      command: "lambda x: {'x': int(x) + 1}"
      next: success
    success:
      type: success
    failure:
      type: fail
```

---

## Using notebook in pipeline

You can use notebooks as a ```command_type``` of a step in the pipeline.  The only caveat in doing so is magnus
would not be able to support returning ```parameters```, ```secrets``` or any of the built-in functions. The cataloging
functionality of magnus still would work via the configuration file.

We use [papermill](https://papermill.readthedocs.io/en/latest/) to inspect the parameters and send them dynamically
from the parameter space.

The command refers to the notebook that you want to use as a task and it should point to the notebook.
The output notebook naming could be provided by using the ```command_config``` section or would be defaulted to the
notebook mentioned in ```command``` section post-fixed with ```_out```.


```yaml
dag:
  description: A single node pipeline with notebook
  start_at: step 1
  steps:
    step 1:
      command_type: notebook
      command: pre_processing.iypnb
      next: success
      command_config:
        notebook_output_path: notebooks/output.ipynb
    success:
      type: success
    failure:
      type: fail
```


---

## A multi node pipeline

A pipeline can have many nodes as part of its execution.

Example:

```python
# In my_module.py

def first_function():
    print('In the function, first_function of my_module')


def second_function():
    print('In the function, second_function of my_module')

```


The pipeline which calls first_function of the above module and then to the call the second_function is given below.

```yaml
dag:
  description: A multi node pipeline
  start_at: step 1
  steps:
    step 1:
      type: task
      next: step 2
      command: my_module.first_function
      command_type: python
    step 2:
      type: task
      next: success
      command: my_module.second_function
      command_type: python
    success:
      type: success
    failure:
      type: fail
```

---

## Using on-failure to handle errors

You can instruct magnus to traverse to a different node of the dag if the current node fails to execute.
A non-zero exit status of the python function or shell command is considered a failure.

The default behavior in case of a failure of a node is, if no ```on_failure``` is defined, is to
traverse to the ```fail``` node of the graph and mark the execution of the dag as failure.

The execution of a dag is considered failure if and only if the ```fail``` node of the graph is reached.

```python
# In my_module.py

def first_function():
    print('In the function, first_function of my_module')


def second_function():
    print('In the function, second_function of my_module')


def handle_error():
    print('Send an email notification')
    ## Some logic to send error notification
    ...

```

The pipeline definition to call ```my_module.handle_error``` in case of a failure of any node is defined below.


```yaml
dag:
  description: A multi node pipeline with on_failure
  start_at: step 1
  steps:
    step 1:
      type: task
      next: step 2
      command: my_module.first_function
      command_type: python
      on_failure: graceful exit
    step 2:
      type: task
      next: success
      command: my_module.second_function
      command_type: python
      on_failure: graceful exit
    graceful exit:
      type: task
      next: fail
      command: my_module.handle_error
      command_type: python
    success:
      type: success
    failure:
      type: fail
```

---
## Passing parameters between nodes

There are several ways we can pass parameters between nodes. Please note that this functionality is only for simple
python data types which can be JSON serializable. Use the catalog functionality to pass files across to different
nodes of the graph.

You can choose any of the methods to pass the parameters from below. All are compatible with each other.

The example pipeline to call all the below functions is given here:


```yaml
dag:
  description: A multi node pipeline to pass parameters
  start_at: step 1
  steps:
    step 1:
      type: task
      next: step 2
      command: my_module.first_function
      command_type: python
    step 2:
      type: task
      next: success
      command: my_module.second_function
      command_type: python
    success:
      type: success
    failure:
      type: fail
```

### Pythonically

```python
# In my_module.py

def first_function():
    print('In the function, first_function of my_module')
    return {'a': 4}


def second_function(a):
    print('In the function, second_function of my_module')
    print(a)

```

In the above code, ```first_function``` is returning a dictionary setting ```a``` to be 4. If the function was called
as a step in the magnus pipeline, magnus adds the key-value pair of ```a=4``` to the parameter space. Note that
```first_function``` can return a dictionary containing as many key-value pairs as needed, magnus would add all of them
to the parameter space.

```second_function``` is expecting a ```named``` argument ```a```. If the function was called as a step in the magnus
pipeline, magnus would look for a parameter ```a``` in the parameter space and assign it.

Very loosely, the whole process can be thought of as: ```second_function(**first_function())```. Since magnus holds
parameter space, the functions need not be consecutive and magnus handles the passing only the required arguments into
the function.


### Using in-built functions
You can also use the built-in functions that magnus provides to ```store``` and ```get``` parameters.

```python
# In my_module.py
from magnus import store_parameter, get_parameter

def first_function():
    print('In the function, first_function of my_module')
    store_parameter(a=4)


def second_function():
    print('In the function, second_function of my_module')
    a = get_parameter('a') # Get parameter with name provides only the named parameter.
    parameters = get_parameter() # Returns a dictionary of all the parameters
    print(a) # prints 4
    print(parameters) # prints {'a': 4}

```

### Using environment variables
The parameters can also be accessed by using environment variables. All magnus specific parameters would be prefixed
by ```MAGNUS_PRM_```. Any environment variable that is prefixed by ```MAGNUS_PRM_``` is also added to the parameter
space.

```python
# In my_module.py
import os

def first_function():
    print('In the function, first_function of my_module')
    os.environ['MAGNUS_PRM_a']=4


def second_function():
    print('In the function, second_function of my_module')
    a = os.environ['MAGNUS_PRM_a']
    print(a)

```

---

## Passing parameters to the first node of the pipeline

There are several ways to set parameters at the start of the execution of the pipeline. Please choose one that fits
your situation.

### During execution of pipeline by magnus

The step ```step parameters``` of the below pipeline expects a parameter ```x``` in the lambda expression.

```yaml
# in getting-started.yaml
dag:
  description: Getting started
  start_at: step parameters
  steps:
    step parameters:
      type: task
      command_type: python-lambda
      command: "lambda x: {'x': int(x) + 1}"
      next: success
    success:
      type: success
    fail:
      type: fail
```

!!! warning "Changed in v0.2"

You can pass the parameter during the execution of the run like below.

```shell
magnus execute --file getting-started.yaml --parameters-file parameters.yaml
```

```yaml
# in parameters.yaml
x: 3
```

### Using environment variables

For the same pipeline defined in ```getting-started.yaml```, you can also pass the parameters as environment variables
prefixed by ```MAGNUS_PRM_x```.

The below command does the same job of passing ```x``` as 3.

```shell
MAGNUS_PRM_x=3; magnus execute --file getting-started.yaml
```

You can pass in as many parameters as you want by prefixing them with ```MAGNUS_PRM_```. All parameters would be read
as ```string``` and have to casted appropriately by the code.

This method of sending parameters by environmental variables is independent of who does the pipeline execution.

---
## Using the catalog to pass artifacts between nodes

While parameters are used to transfer simple and JSON serializable data types, catalog can be used to make larger files
or artifacts available to down stream nodes. A typical configuration of catalog provider would be:

```yaml
catalog:
  type:  #defaults to file-system
  config:
    compute_data_folder: # defaults to data/
```

If no config is provided, magnus defaults to ```file-system```.

Logically magnus does the following:

- ```get``` files from the catalog before the execution to a specific ```compute data folder```
- execute the command
- ```put``` any files from the ```compute data folder``` back to the catalog.

### Using the configuration.

```yaml
dag:
  description: Getting started
  start_at: step shell make data
  steps:
    step shell make data:
      type: task
      command_type: shell
      command: mkdir data ; env >> data/data.txt
      next: step shell ls data
      catalog:
        put:
          - "*"
    step shell ls data:
      type: task
      command_type: shell
      command: ls data/
      next: success
      catalog:
        compute_data_folder: data/ # This is the default value too.
        get:
          - "*"
    success:
      type: success
    fail:
      type: fail
```

In the above dag definition, ```step shell make data``` makes a data folder and dumps the environmental variables into
```data.txt``` file and instructs the catalog to ```put``` all (i.e '*') files into the catalog for downstream nodes.

While the step ```step shell ls data``` instructs the catalog to ```get``` (i.e '*') files from the catalog and put
them in ```compute_data_folder``` which is ```data``` and executes the command to see the contents of the directory.

You can over-ride the ```compute_data_folder``` of a single step to any folder that you want as shown.

Glob patterns are perfectly allowed and you can it to selectively ```get``` or ```put``` files in the catalog.

### Using the in-built functions

You can interact with the catalog from the python code too if that is convenient.

```python
# In my_module.py
from pathlib import Path

from magnus import put_in_catalog, get_from_catalog

def first_function():
    print('In the function, first_function of my_module')
    Path('data').mkdir(parents=True, exist_ok=True)

    with open('data/data.txt', 'w') as fw:
      fw.write('something interesting)

    # filepath is required and can be a glob pattern
    put_in_catalog(filepath='data/data.txt')

def second_function():
    print('In the function, second_function of my_module')

    # name is required and can be a glob pattern.
    # destination_folder is defaulted to the compute_data_folder as defined in the config
    get_from_catalog(name='data.txt', destination_folder='data/')

```

The python function ```first_function``` makes the ```compute_data_folder``` and instructs the catalog to put it the
catalog. The python function ```second_function``` instructs the catalog to get the file by name ```data.txt``` from
the catalog and put it in the folder ```data/```. You can use glob patterns both in ```put_in_catalog``` or
```get_from_catalog```.

The corresponding pipeline definition need not even aware of the cataloging happening by the functions.

```yaml
dag:
  description: A multi node pipeline
  start_at: step 1
  steps:
    step 1:
      type: task
      next: step 2
      command: my_module.first_function
      command_type: python
    step 2:
      type: task
      next: success
      command: my_module.second_function
      command_type: python
    success:
      type: success
    failure:
      type: fail
```

---
## Using the catalog to source external data

In magnus, you can only ```get``` from catalog if the catalog location already exists. Calling ```put``` in catalog,
which safely makes the catalog location if it does not exist, before you are trying to ```get``` from the catalog
ensures that the catalog location is always present.

But there are situations where you want to call ```get``` before you ```put``` data in the catalog location by the
steps of the pipeline. For example, you want to source a data file generated by external processes and transform them
in your pipeline. You can achieve that by the fact all catalog providers (eg. file-system and extensions) use
```run_id``` as the directory (or partition) of the catalog.

To source data from external sources for a particular run,

- Create a ```run_id``` that you want to use for pipeline execution.
- Create the directory (or partition) in the catalog location by that ```run_id```
- Copy the contents that you want the pipeline steps to access in the catalog location.
- Run the magnus pipeline by providing the ```run_id``` i.e ```magnus execute --run-id run_id --file <>```

Since the catalog location already exists, ```get``` from the catalog will source the external data.

---
## Accessing secrets within code.

Secrets are the only service that magnus provides where you need to ```import magnus``` in your source code. This is
to ensure that the integrity of the secrets are held and handled safely.

A typical configuration of the secrets is:

```yaml
secrets:
  type:  #defaults to do-nothing
  config:
```

By default, magnus chooses a ```do-nothing``` secrets provider which holds no secrets. For local development,
```dotenv``` secrets manager is useful and the config is as below.

```yaml
secrets:
  type:  dotenv
  config:
    location: # defaults to .env
```

Example:

```
#Inside .env file
secret_name=secret_value#Any comment that you want to pass

```

Any content after # is ignored and the format is ```key=value``` pairs.

```python
# In my_module.py
from magnus import get_secret

def first_function():
    print('In the function, first_function of my_module')
    secret_value = get_secret('secret_name')
    print(secret_value) # Should print secret_value

    secrets = get_secret()
    print(secrets) # Should print {'secret_name': 'secret_value'}
```

The pipeline to run the above function as a step of the pipeline.

```yaml
secrets:
  type:  dotenv
  config:
    location: # defaults to .env

dag:
  description: Demo of secrets
  start_at: step 1
  steps:
    step 1:
      type: task
      next: success
      command: my_module.first_function
      command_type: python
    success:
      type: success
    failure:
      type: fail
```

---
## Parallel node

We will be using ```as-is``` nodes as part of the examples to keep it simple but the concepts of nesting/branching
remain the same even in the case of actual tasks.

Example of a parallel node:

```yaml
# in the yaml example-parallel.yaml
run_log_store:
  type: file-system

dag:
  description: DAG for testing with as-is and parallel
  start_at: step1
  steps:
    step1:
      type: as-is
      next: step2
    step2:
      type: parallel
      next: success
      branches:
        branch_1:
          start_at: step_1
          steps:
            step_1:
              type: as-is
              next: success
            success:
              type: success
            fail:
              type: fail
        branch_2:
          start_at: step_1
          steps:
            step_1:
              type: as-is
              next: success
            success:
              type: success
            fail:
              type: fail
    success:
      type: success
    fail:
      type: fail
```

You can execute the above dag by:

```magnus execute --file example-parallel.yaml```

The above run should produce a ```run_log``` in the ```.run_log_store``` directory with the ```run_id``` as filename.

The contents of the log should be similar to this:

<details>
  <summary>Click to show the run log</summary>


```json

{
    "run_id": "20220120131257",
    "dag_hash": "cf5cc7df88d4af3bc0936a9a8a3c4572ce4e11bc",
    "use_cached": false,
    "tag": null,
    "original_run_id": "",
    "status": "SUCCESS",
    "steps": {
        "step1": {
            "name": "step1",
            "internal_name": "step1",
            "status": "SUCCESS",
            "step_type": "as-is",
            "message": "",
            "mock": false,
            "code_identities": [
                {
                    "code_identifier": "2a5b33bdf60c4f0d38cae04ab3f988b3d1c6ed59",
                    "code_identifier_type": "git",
                    "code_identifier_dependable": false,
                    "code_identifier_url": <INTENTIONALLY_REMOVED>,
                    "code_identifier_message": <INTENTIONALLY_REMOVED>
                }
            ],
            "attempts": [
                {
                    "attempt_number": 0,
                    "start_time": "2022-01-20 13:12:57.999265",
                    "end_time": "2022-01-20 13:12:57.999287",
                    "duration": "0:00:00.000022",
                    "status": "SUCCESS",
                    "message": ""
                }
            ],
            "user_defined_metrics": {},
            "branches": {},
            "data_catalog": []
        },
        "step2": {
            "name": "step2",
            "internal_name": "step2",
            "status": "SUCCESS",
            "step_type": "parallel",
            "message": "",
            "mock": false,
            "code_identities": [
                {
                    "code_identifier": "2a5b33bdf60c4f0d38cae04ab3f988b3d1c6ed59",
                    "code_identifier_type": "git",
                    "code_identifier_dependable": false,
                    "code_identifier_url": <INTENTIONALLY_REMOVED>,
                    "code_identifier_message": <INTENTIONALLY_REMOVED>
                }
            ],
            "attempts": [],
            "user_defined_metrics": {},
            "branches": {
                "step2.branch_1": {
                    "internal_name": "step2.branch_1",
                    "status": "SUCCESS",
                    "steps": {
                        "step2.branch_1.step_1": {
                            "name": "step_1",
                            "internal_name": "step2.branch_1.step_1",
                            "status": "SUCCESS",
                            "step_type": "as-is",
                            "message": "",
                            "mock": false,
                            "code_identities": [
                                {
                                    "code_identifier": "2a5b33bdf60c4f0d38cae04ab3f988b3d1c6ed59",
                                    "code_identifier_type": "git",
                                    "code_identifier_dependable": false,
                                    "code_identifier_url": <INTENTIONALLY_REMOVED>,
                                    "code_identifier_message": <INTENTIONALLY_REMOVED>
                                }
                            ],
                            "attempts": [
                                {
                                    "attempt_number": 0,
                                    "start_time": "2022-01-20 13:12:58.090461",
                                    "end_time": "2022-01-20 13:12:58.090476",
                                    "duration": "0:00:00.000015",
                                    "status": "SUCCESS",
                                    "message": ""
                                }
                            ],
                            "user_defined_metrics": {},
                            "branches": {},
                            "data_catalog": []
                        },
                        "step2.branch_1.success": {
                            "name": "success",
                            "internal_name": "step2.branch_1.success",
                            "status": "SUCCESS",
                            "step_type": "success",
                            "message": "",
                            "mock": false,
                            "code_identities": [
                                {
                                    "code_identifier": "2a5b33bdf60c4f0d38cae04ab3f988b3d1c6ed59",
                                    "code_identifier_type": "git",
                                    "code_identifier_dependable": false,
                                    "code_identifier_url": <INTENTIONALLY_REMOVED>,
                                    "code_identifier_message": <INTENTIONALLY_REMOVED>
                                }
                            ],
                            "attempts": [
                                {
                                    "attempt_number": 0,
                                    "start_time": "2022-01-20 13:12:58.135551",
                                    "end_time": "2022-01-20 13:12:58.135732",
                                    "duration": "0:00:00.000181",
                                    "status": "SUCCESS",
                                    "message": ""
                                }
                            ],
                            "user_defined_metrics": {},
                            "branches": {},
                            "data_catalog": []
                        }
                    }
                },
                "step2.branch_2": {
                    "internal_name": "step2.branch_2",
                    "status": "SUCCESS",
                    "steps": {
                        "step2.branch_2.step_1": {
                            "name": "step_1",
                            "internal_name": "step2.branch_2.step_1",
                            "status": "SUCCESS",
                            "step_type": "as-is",
                            "message": "",
                            "mock": false,
                            "code_identities": [
                                {
                                    "code_identifier": "2a5b33bdf60c4f0d38cae04ab3f988b3d1c6ed59",
                                    "code_identifier_type": "git",
                                    "code_identifier_dependable": false,
                                    "code_identifier_url": <INTENTIONALLY_REMOVED>,
                                    "code_identifier_message": <INTENTIONALLY_REMOVED>
                                }
                            ],
                            "attempts": [
                                {
                                    "attempt_number": 0,
                                    "start_time": "2022-01-20 13:12:58.187648",
                                    "end_time": "2022-01-20 13:12:58.187661",
                                    "duration": "0:00:00.000013",
                                    "status": "SUCCESS",
                                    "message": ""
                                }
                            ],
                            "user_defined_metrics": {},
                            "branches": {},
                            "data_catalog": []
                        },
                        "step2.branch_2.success": {
                            "name": "success",
                            "internal_name": "step2.branch_2.success",
                            "status": "SUCCESS",
                            "step_type": "success",
                            "message": "",
                            "mock": false,
                            "code_identities": [
                                {
                                    "code_identifier": "2a5b33bdf60c4f0d38cae04ab3f988b3d1c6ed59",
                                    "code_identifier_type": "git",
                                    "code_identifier_dependable": false,
                                    "code_identifier_url": <INTENTIONALLY_REMOVED>,
                                    "code_identifier_message": <INTENTIONALLY_REMOVED>
                                }
                            ],
                            "attempts": [
                                {
                                    "attempt_number": 0,
                                    "start_time": "2022-01-20 13:12:58.233479",
                                    "end_time": "2022-01-20 13:12:58.233681",
                                    "duration": "0:00:00.000202",
                                    "status": "SUCCESS",
                                    "message": ""
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
                    "code_identifier": "2a5b33bdf60c4f0d38cae04ab3f988b3d1c6ed59",
                    "code_identifier_type": "git",
                    "code_identifier_dependable": false,
                    "code_identifier_url": <INTENTIONALLY_REMOVED>,
                    "code_identifier_message": <INTENTIONALLY_REMOVED>
                }
            ],
            "attempts": [
                {
                    "attempt_number": 0,
                    "start_time": "2022-01-20 13:12:58.280538",
                    "end_time": "2022-01-20 13:12:58.280597",
                    "duration": "0:00:00.000059",
                    "status": "SUCCESS",
                    "message": ""
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
            "type": "local",
            "config": {}
        },
        "run_log_store": {
            "type": "file-system",
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
</details>

The individual steps of the dag are named in [```dot path convention```](../concepts/run-log/#naming_step_log)

You can nest a ```parallel``` node, ```dag``` or a ```map``` node within parallel node to enable modular dag designs.

### Enabling parallel execution

Though the dag definition defines a ```parallel``` node, the execution of the dag and the parallelism is actually
controlled by the executor. In ```local``` mode, you can enable parallel branch execution by modifying the config.

```yaml
mode:
  type: local
  config:
    enable_parallel: "true"
```

Points to note:

- The ```enable_parallel``` flag in the config is a string "true"
- Run log stores which use a single file as their log source (eg. file-system) cannot reliably run parallel executions
  as race conditions to modify the same file can happen leaving the run log in inconsistent state. The logs of the
  execution would also warn the same. Partitioned run log stores (eg. db) can be reliable run log stores.

---
## Embedding dag within dag

You can embed dag's defined elsewhere into your dag.

For example, we can define a dag which works all by itself in sub-dag.yaml

```yaml
# in sub-dag.yaml
dag:
  description: sub dag
  start_at: step1
  steps:
    step1:
      type: as-is
      next: step2
    step2:
      type: as-is
      next: success
    success:
      type: success
    fail:
      type: fail

```

We can embed this dag into another dag as a node like below.

```yaml
dag:
  description: DAG for nested dag
  start_at: step_dag_within_dag
  steps:
    step_dag_within_dag:
      type: dag
      dag_definition: sub-dag.yaml # Should be the filepath to the dag you want to embed.
      next: success
    success:
      type: success
    fail:
      type: fail

```

Nested dag's should allow for a very modular design where individual dag's do well defined tasks but the nested dag
can stitch them to complete the whole task.

As with parallel execution, the individual steps of the dag are named in
[```dot path convention```](../concepts/run-log/#naming_step_log)

---
## Looping a branch over an iterable parameter

Often, you would need to do the same repetitive tasks over a list and magnus allows you to do that.

Example of dynamic branch looping is below.

```yaml
# in map-state.yaml
dag:
  description: DAG for map
  start_at: step1
  steps:
    step1:
      type: task
      command: "lambda : {'variables' : ['a', 'b', 'c']}"
      command_type: python-lambda
      next: step2
    step2:
      type: map
      iterate_on: variables
      iterate_as: x
      next: success
      branch:
        start_at: step_1
        steps:
          step_1:
            type: task
            command: "lambda x : {'state_' + str(x) : 5}"
            command_type: python-lambda
            next: success
          success:
            type: success
          fail:
            type: fail
    success:
      type: success
    fail:
      type: fail

```

In the above dag, step1 sets the parameters ```variables``` as list ```['a', 'b', 'c']```.
step2 is a node of type map which will iterate on ```variables``` and execute the ```branch``` defined as part of the
definition of step2 for every value in the iterable ```variables```.

The ```branch``` definition of the step2 basically creates one more parameter ```state_<variable>=5``` by the lambda
expression. You can see these parameters as part of the run log show below.

<details>
  <summary>Click to show the run log</summary>

``` json
{
    "run_id": "20220120150813",
    "dag_hash": "c0492a644b4f28f8441d669d9f0efb0f6d6be3d3",
    "use_cached": false,
    "tag": null,
    "original_run_id": "",
    "status": "SUCCESS",
    "steps": {
        "step1": {
            "name": "step1",
            "internal_name": "step1",
            "status": "SUCCESS",
            "step_type": "task",
            "message": "",
            "mock": false,
            "code_identities": [
                {
                    "code_identifier": "2a5b33bdf60c4f0d38cae04ab3f988b3d1c6ed59",
                    "code_identifier_type": "git",
                    "code_identifier_dependable": false,
                    "code_identifier_url": "INTENTIONALLY_REMOVED",
                    "code_identifier_message": "INTENTIONALLY_REMOVED"
                }
            ],
            "attempts": [
                {
                    "attempt_number": 0,
                    "start_time": "2022-01-20 15:08:14.069919",
                    "end_time": "2022-01-20 15:08:14.070484",
                    "duration": "0:00:00.000565",
                    "status": "SUCCESS",
                    "message": ""
                }
            ],
            "user_defined_metrics": {},
            "branches": {},
            "data_catalog": []
        },
        "step2": {
            "name": "step2",
            "internal_name": "step2",
            "status": "SUCCESS",
            "step_type": "map",
            "message": "",
            "mock": false,
            "code_identities": [
                {
                    "code_identifier": "2a5b33bdf60c4f0d38cae04ab3f988b3d1c6ed59",
                    "code_identifier_type": "git",
                    "code_identifier_dependable": false,
                    "code_identifier_url": "INTENTIONALLY_REMOVED",
                    "code_identifier_message": "INTENTIONALLY_REMOVED"
                }
            ],
            "attempts": [],
            "user_defined_metrics": {},
            "branches": {
                "step2.a": {
                    "internal_name": "step2.a",
                    "status": "SUCCESS",
                    "steps": {
                        "step2.a.step_1": {
                            "name": "step_1",
                            "internal_name": "step2.a.step_1",
                            "status": "SUCCESS",
                            "step_type": "task",
                            "message": "",
                            "mock": false,
                            "code_identities": [
                                {
                                    "code_identifier": "2a5b33bdf60c4f0d38cae04ab3f988b3d1c6ed59",
                                    "code_identifier_type": "git",
                                    "code_identifier_dependable": false,
                                    "code_identifier_url": "INTENTIONALLY_REMOVED",
                                    "code_identifier_message": "INTENTIONALLY_REMOVED"
                                }
                            ],
                            "attempts": [
                                {
                                    "attempt_number": 0,
                                    "start_time": "2022-01-20 15:08:14.162440",
                                    "end_time": "2022-01-20 15:08:14.162882",
                                    "duration": "0:00:00.000442",
                                    "status": "SUCCESS",
                                    "message": ""
                                }
                            ],
                            "user_defined_metrics": {},
                            "branches": {},
                            "data_catalog": []
                        },
                        "step2.a.success": {
                            "name": "success",
                            "internal_name": "step2.a.success",
                            "status": "SUCCESS",
                            "step_type": "success",
                            "message": "",
                            "mock": false,
                            "code_identities": [
                                {
                                    "code_identifier": "2a5b33bdf60c4f0d38cae04ab3f988b3d1c6ed59",
                                    "code_identifier_type": "git",
                                    "code_identifier_dependable": false,
                                    "code_identifier_url": "INTENTIONALLY_REMOVED",
                                    "code_identifier_message": "INTENTIONALLY_REMOVED"
                                }
                            ],
                            "attempts": [
                                {
                                    "attempt_number": 0,
                                    "start_time": "2022-01-20 15:08:14.209895",
                                    "end_time": "2022-01-20 15:08:14.210106",
                                    "duration": "0:00:00.000211",
                                    "status": "SUCCESS",
                                    "message": ""
                                }
                            ],
                            "user_defined_metrics": {},
                            "branches": {},
                            "data_catalog": []
                        }
                    }
                },
                "step2.b": {
                    "internal_name": "step2.b",
                    "status": "SUCCESS",
                    "steps": {
                        "step2.b.step_1": {
                            "name": "step_1",
                            "internal_name": "step2.b.step_1",
                            "status": "SUCCESS",
                            "step_type": "task",
                            "message": "",
                            "mock": false,
                            "code_identities": [
                                {
                                    "code_identifier": "2a5b33bdf60c4f0d38cae04ab3f988b3d1c6ed59",
                                    "code_identifier_type": "git",
                                    "code_identifier_dependable": false,
                                    "code_identifier_url": "INTENTIONALLY_REMOVED",
                                    "code_identifier_message": "INTENTIONALLY_REMOVED"
                                }
                            ],
                            "attempts": [
                                {
                                    "attempt_number": 0,
                                    "start_time": "2022-01-20 15:08:14.258519",
                                    "end_time": "2022-01-20 15:08:14.258982",
                                    "duration": "0:00:00.000463",
                                    "status": "SUCCESS",
                                    "message": ""
                                }
                            ],
                            "user_defined_metrics": {},
                            "branches": {},
                            "data_catalog": []
                        },
                        "step2.b.success": {
                            "name": "success",
                            "internal_name": "step2.b.success",
                            "status": "SUCCESS",
                            "step_type": "success",
                            "message": "",
                            "mock": false,
                            "code_identities": [
                                {
                                    "code_identifier": "2a5b33bdf60c4f0d38cae04ab3f988b3d1c6ed59",
                                    "code_identifier_type": "git",
                                    "code_identifier_dependable": false,
                                    "code_identifier_url": "INTENTIONALLY_REMOVED",
                                    "code_identifier_message": "INTENTIONALLY_REMOVED"
                                }
                            ],
                            "attempts": [
                                {
                                    "attempt_number": 0,
                                    "start_time": "2022-01-20 15:08:14.305524",
                                    "end_time": "2022-01-20 15:08:14.305754",
                                    "duration": "0:00:00.000230",
                                    "status": "SUCCESS",
                                    "message": ""
                                }
                            ],
                            "user_defined_metrics": {},
                            "branches": {},
                            "data_catalog": []
                        }
                    }
                },
                "step2.c": {
                    "internal_name": "step2.c",
                    "status": "SUCCESS",
                    "steps": {
                        "step2.c.step_1": {
                            "name": "step_1",
                            "internal_name": "step2.c.step_1",
                            "status": "SUCCESS",
                            "step_type": "task",
                            "message": "",
                            "mock": false,
                            "code_identities": [
                                {
                                    "code_identifier": "2a5b33bdf60c4f0d38cae04ab3f988b3d1c6ed59",
                                    "code_identifier_type": "git",
                                    "code_identifier_dependable": false,
                                    "code_identifier_url": "INTENTIONALLY_REMOVED",
                                    "code_identifier_message": "INTENTIONALLY_REMOVED"
                                }
                            ],
                            "attempts": [
                                {
                                    "attempt_number": 0,
                                    "start_time": "2022-01-20 15:08:14.353182",
                                    "end_time": "2022-01-20 15:08:14.353603",
                                    "duration": "0:00:00.000421",
                                    "status": "SUCCESS",
                                    "message": ""
                                }
                            ],
                            "user_defined_metrics": {},
                            "branches": {},
                            "data_catalog": []
                        },
                        "step2.c.success": {
                            "name": "success",
                            "internal_name": "step2.c.success",
                            "status": "SUCCESS",
                            "step_type": "success",
                            "message": "",
                            "mock": false,
                            "code_identities": [
                                {
                                    "code_identifier": "2a5b33bdf60c4f0d38cae04ab3f988b3d1c6ed59",
                                    "code_identifier_type": "git",
                                    "code_identifier_dependable": false,
                                    "code_identifier_url": "INTENTIONALLY_REMOVED",
                                    "code_identifier_message": "INTENTIONALLY_REMOVED"
                                }
                            ],
                            "attempts": [
                                {
                                    "attempt_number": 0,
                                    "start_time": "2022-01-20 15:08:14.401043",
                                    "end_time": "2022-01-20 15:08:14.401304",
                                    "duration": "0:00:00.000261",
                                    "status": "SUCCESS",
                                    "message": ""
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
                    "code_identifier": "2a5b33bdf60c4f0d38cae04ab3f988b3d1c6ed59",
                    "code_identifier_type": "git",
                    "code_identifier_dependable": false,
                    "code_identifier_url": "INTENTIONALLY_REMOVED",
                    "code_identifier_message": `"INTENTIONALLY_REMOVED"`
                }
            ],
            "attempts": [
                {
                    "attempt_number": 0,
                    "start_time": "2022-01-20 15:08:14.449759",
                    "end_time": "2022-01-20 15:08:14.449826",
                    "duration": "0:00:00.000067",
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
        "variables": [
            "a",
            "b",
            "c"
        ],
        "state_a": 5,
        "state_b": 5,
        "state_c": 5
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
</details>

The individual steps of the dag are named in  [```dot path convention```](../concepts/run-log/#naming_step_log).

### Enabling parallel execution

Though the dag definition defines a ```map``` node where the branches can be executed in parallel,
the execution of the dag and the parallelism is actually
controlled by the executor. In ```local``` mode, you can enable parallel branch execution by modifying the config.

```yaml
mode:
  type: local
  config:
    enable_parallel: "true"
```

Points to note:

- The ```enable_parallel``` flag in the config is a string "true"
- Run log stores which use a single file as their log source (eg. file-system) cannot reliably run parallel executions
  as race conditions to modify the same file can happen leaving the run log in inconsistent state. The logs of the
  execution would also warn the same. Partitioned run log stores (eg. db) can be reliable run log stores.

---
## Nesting and complex dags

Magnus does not limit you at all in nesting at any level. You have construct deep nesting levels easily and magnus
would execute them as you designed.

As a general coding practice, having deeply nested branches could be hard to read and maintain.

***NOTE***: There is a possibility that you can nest the same dag within the dag definition resulting in a infinite
loop. We are actively finding ways to detect these situations and warn you.

---
## Advanced use as-is

Node type ```as-is``` defined in magnus can be a very powerful tool in some deployment patterns.

For example in the below dag definition, the step ```step echo``` does nothing as part of ```local``` execution.

```yaml
mode:
  type: demo-renderer

run_log_store:
  type: file-system

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
      command: mkdir data ; env >> data/data.txt
      next: step echo
      catalog:
        put:
          - "*"
    step echo:
      type: as-is
      command_type: shell
      command_config:
        render_string: echo hello
      next: success
    success:
      type: success
    fail:
      type: fail
```

But a deployment pattern, like ```demo-renderer```, can use it to inject a command into the bash script. To test it out,
uncomment the config to change to mode to ```demo-renderer``` and the run log store to be ```file-system``` and
execute it like below.

```magnus execute --file getting-started.yaml```

should generate a bash script as show below in ```demo-bash.sh```.

```shell
for ARGUMENT in "${@:2}"
do
	KEY=$(echo $ARGUMENT | cut -f1 -d=)
	VALUE=$(echo $ARGUMENT | cut -f2 -d=)
	export "MAGNUS_PRM_$KEY"=$VALUE
done
magnus execute_single_node $1 step%parameters --file getting-started.yaml
exit_code=$?
echo $exit_code
if [ $exit_code -ne 0 ];
then
	 $(magnus execute_single_node $1 fail --file getting-started.yaml)
	exit 1
fi
magnus execute_single_node $1 step%shell --file getting-started.yaml
exit_code=$?
echo $exit_code
if [ $exit_code -ne 0 ];
then
	 $(magnus execute_single_node $1 fail --file getting-started.yaml)
	exit 1
fi
echo hello
exit_code=$?
echo $exit_code
if [ $exit_code -ne 0 ];
then
	 $(magnus execute_single_node $1 fail --file getting-started.yaml)
	exit 1
fi
magnus execute_single_node $1 success --file getting-started.yaml
```

The shell script is translation of the dag into a series of bash commands but notice the command ```echo hello``` as
part of the script. While the ```local``` mode interpreted that node as a stub or a mock node, the ```demo-renderer```
mode used the ```render_string``` variable of the node ```config``` to inject a script.

This feature is very useful when you want certain few steps (may be email notifications) to be only possible in
production like environments but want to mock the during dev/experimental set up.

***NOTE***: When trying to ```locally``` re-run a dag definition with ```as-is``` node used to inject scripts,
the run would start from ```as-is``` step onwards independent of the source of failure. You can change this
behavior by writing extensions which skip over ```as-is``` nodes during re-run.
## Controlling the log level of magnus

The default log level of magnus is WARNING but you can change it at the point of execution to one of
```['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET]``` by using the command line argument --log-level.

For example:

```magnus execute --file <dag definition file> --log-level DEBUG```

would set the magnus log level to DEBUG. This setting only affects magnus logs and will not alter your application log
levels.

---
## Order of configurations

Magnus supports many ways of providing configurations but there is a order of preference.

Magnus defaults to the following if no config is provided.

```yaml
mode:
  type: local
  config:
    enable_parallel: "false"

run_log_store:
  type: buffered

catalog:
  type: file-system
  config:
    compute_data_folder: data/
    catalog_location: .catalog

secrets:
  type: do-nothing

```

But you can over-ride these defaults by providing a ```magnus-config.yaml``` in the source directory. For example,
if the ```magnus-config.yaml``` file has the following contents, even if you do not provide a config in the dag
definition file, these would taken as default service providers.

```yaml
mode:
  type: local
  config:
    enable_parallel: "true" # false is the default

run_log_store:
  type: file-system

catalog:
  type: file-system
  config:
    compute_data_folder: data/ # default
    catalog_location: .catalog # default

secrets:
  type: dotenv
  config:
    location: .env # default
```

You can over-ride the defaults either set by magnus or ```magnus-config.yaml``` by providing the config in the dag
definition file.

For example, in the dag definition below only the ```secrets``` providers config is over-ridden by the config present
in the dag definition file. Compute mode, catalog and run log store configurations remain the same to defaults.

```yaml
secrets:
  type:  do-nothing


dag:
  description: Demo of secrets
  start_at: step 1
  steps:
    step 1:
      type: task
      next: success
      command: my_module.first_function
      command_type: python
    success:
      type: success
    failure:
      type: fail
```

Finally, you can also over-ride the configurations set in the dag definition file by providing a custom configuration
file containing only the configurations.

For example, you can provide a dag definition file as above with ```do-nothing``` secrets handler but by providing
the below configurations file at the run time, you can over-ride it to ```dotenv```.

```yaml
#in prod-configuration.yaml
secrets:
  type:  dotenv

```

The command to execute while providing the configuration file.

```magnus execute --file <dag definition file> --config-file prod-configuration.yaml```



The design thought is enable switching between different configurations by different actors involved in the data science
workflow. The engineering team could provide ```magnus-config.yaml``` that should be default to the team or project
for dev/experimental phase of the work but can over-ride the configuration during production deployment.


## Custom local extensions

Magnus was built with extensions in mind. For example, there could be catalog extension using s3 or object storage that
are generic enough to be open sourced back to the community. But there is always a chance where the extension is only
specific to your team or project. You can implement custom extensions to either compute mode, run log store, catalog or
secrets as part of your source folder and let magnus know to use them.

For example, consider the use case of a custom secrets handler that only serves your team needs, called CustomSecrets
which extends ```BaseSecrets``` provided by magnus like below. The secrets manager does nothing special and always
returns 'always the same' as the secret value.

```python
# Present in the src.custom_secrets folder of your project
from magnus.secrets import BaseSecrets

class CustomSecrets(BaseSecrets):
    """
    Does the same thing
    """

    service_name = 'custom-secrets'

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.secrets = {}

    def get(self, name: str = None, **kwargs) -> Union[str, dict]:
        """
        If a name is provided, return None else return empty dict.

        Args:
            name (str): The name of the secret to retrieve

        Raises:
            Exception: If the secret by the name is not found.

        Returns:
            [type]: [description]
        """
        if name:
            return 'always the same'
        return {'secret_key': 'always the same'}
```

You can instruct magnus to detect and use the ```CustomSecrets``` by providing it in the ```magnus-config.yaml``` like
below.

```yaml
# in magnus-config.yaml
extensions:
  - src.custom_secrets
```

Magnus would import the contents of the module defined in extensions and would delegate the responsibility of secrets
to ```CustomSecrets```.

We would love it if you share your custom extension code or the design aspect as it builds the community.
