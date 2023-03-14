# Nodes

---

Nodes are fundamentally the smallest logical unit of work that you want to execute. Though there is no explicit
guidelines on how big or small a node should be, we advice that the node becomes a part of narrative of the
whole project.

For example, lets take a scenario where you perform some data cleaning task before you are ready to transform/train
a machine learning model. The data cleaning task could be one single *task* node or single *dag* node
(which internally is a graph) if you have too many steps. The choice is completely yours to make and
depends on the narrative of the project.

Nodes in magnus can be logically split into 3 types:

- **Execution**: fundamentally this is a **python function call or Shell command** that you want to call as part of the
pipeline. Task and As-Is node is the only nodes of this type.

- **Status**: nodes that denote the eventual status of a graph/sub-graph. Success or Fail nodes are
examples of this type. All dag definitions should have **one and exactly one** node of this type and
the status of the dag is basically the type of status node it hits at the end.

- **Composite**: nodes that are **sub-graphs** by itself. Parallel, Dag and Map are examples of this type and
all three have different use cases. Nesting of composite nodes is possible, though we advise to keep the
nesting simple to promote readability.

---
!!! Note

    Node names cannot have . or % in them.
    Any valid python string is acceptable as a name of the step.

---

## Task

The smallest executable of the pipeline or in python language, the function call that you want to call as
part of the the pipeline. In magnus, a task node has the following configuration.

```yaml
step name:
  retry: 1 # Defaults to 1
  type: task
  next:
  command:
  command_type: # Defaults to python
  on_failure:  # Defaults to None
  mode_config: # Defaults to None
  catalog: # Defaults to None
    compute_data_folder:
    get:
    put:
```

Or via the Python SDK:

```python
from magnus import Task

first = Task(name: str, command: str, command_type: str = 'python',
            command_config: Optional[dict]=None, catalog: Optional[dict]=None,
            mode_config: Optional[dict]=None, retry: int = 1, on_failure: str = '', next_node:str=None)
```
The name given to the task has the same behavior as the ```step name``` given in the yaml definition.

### command (required)

The name of the actual function/shell executable you want to call as part of the pipeline.

For example, for the following function, the command would be ```my_module.my_cool_function```.

```python

# In my_module.py

def my_cool_function():
    pass
```

### command_type (optional)
Defaults to python if nothing is provided. For more information, please refer [command types](../command-types)

### retry (optional)
The number of attempts to make before failing the node. Default to 1.

For local executions, this is always be 1 independent of the actual ```retry``` value.
For cloud based implementations, the retry value is passed to the implementation.

### next (required)
The name of the node in the graph to go if the node succeeds.

```next``` is optional via SDK as it is assigned during pipeline construction stage.

### on_failure (optional)
The name of the node in the graph to go if the node fails.
This is optional as we would move to the fail node of the graph if one is not provided.

On_failure could be an use case where you want to send a failure notification before marking the run as failure.

### mode_config (optional)
Use this section to pass instructions to the executor.
For example, we can instruct the ```local-container``` mode to use a different docker image to run this step of the
pipeline.

Example usage of mode_config:

```yaml
# In config.yaml
mode:
  type: local-container
  config:
    docker_image: python:3.7

# In pipeline.yaml
dag:
  start_at: Cool function
  steps:
    Cool function:
      type: task
      command: my_module.my_cool_function
      next: Clean Up
    Clean Up:
      type: task
      command: clean_up.sh
      command_type: shell
      mode_config:
        docker_image: ubuntu:latest
      next: Success
    Success:
      type: success
    Fail:
      type: fail
```

Or the same pipeline via the Python SDK:

```python
# In pipeline.py
from magnus import Task, Pipeline

def pipeline():
  first = Task(name='Cool function', command='my_module.my_cool_function')
  second = Task(name='Clean Up', command='clean_up.sh', command_type='shell',
            mode_config={'docker_image': 'ubunutu:latest'})

  pipeline = pipeline(name='my pipeline')
  pipeline.construct([first, second])
  pipeline.execute(configuration_file='config.yaml')

if __name__ == '__main__':
    pipeline()

```

In the above example, while all the steps except for ```Clean Up``` happen in python3.7 docker image, the ```Clean Up```
happens in Ubuntu.

mode_config provides a way for dag to have customizable instructions to the executor.

### catalog (optional)

compute_data_folder: The folder where we need to sync-in or sync-out the data to the [catalog](../catalog).
If it is not provided,
it defaults to the global catalog settings.

get: The files to sync-in from the catalog to the compute data folder, prior execution.

put: The files to sync-out from the compute data folder to the catalog, post execution.

Glob pattern naming in get or put are fully supported, internally we use
[Pathlib match function](https://docs.python.org/3/library/pathlib.html#pathlib.PurePath.match)
to match the name to pattern.

Example catalog settings:
```yaml
catalog:
  compute_data_folder: data/
  get:
    - '*'
  put:
    - 'cleaned*'
```

or the same in Python SDK:

```python
from magnus import Task

catalog = {
  'compute_data_folder' : 'data/',
  'get': ['*'],
  'put': ['cleaned*']
}

first = Task(name='Cool function', command='my_module.my_cool_function', catalog=catalog)

```

In this, we sync-in all the files from the catalog to the compute data folder, data prior to the execution and
sync-out all files started with *cleaned* to the catalog after the execution.

<br>
Logically, magnus does the following when executing a task:

1. Check the catalog-get list for any files that have to be synced to compute data folder.
2. Inspect the function call to determine the arguments required to make the function call.
    Retrieve them from the parameters or fail if not present.
3. Check if the function call has to executed in case of re-runs. If the previous re-run  of the step
was successful, we skip it.
4. Make the actual function call, if we need to, and determine the result.
5. Check the catalog-put list for any files that have to be synced back to catalog from the compute data folder.


### next_node:

In python SDK, you need to provide the next node of the execution using ```next_node``` unless the node ends in
```success``` state. If you want to end the graph execution to fail state, you can use ```next_node='fail'```.


## Success

A status node of the graph. There should be **one and only one** success node per graph.
The traversal of the graph stops at this node with marking the run as success.
In magnus, this node can be configured as:

```yaml
step name:
  type: success
```

No other fields are required and should not be provided.

## Fail

A status node of the graph. There should be **one and only one** fail node per graph.
The traversal of the graph stops at this node with marking the run as fail. In magnus, this node can be configured as:

```yaml
step name:
  type: fail
```

No other fields are required and should not be provided.

## Parallel

Parallel node is a composite node that in it-self has sub-graphs. A good example is to construct independent
features of a training data in machine learning experiments. The number of branches in parallel node is static
and pre-determined. Each branch follows the same definition language as the graph.

The configuration of a parallel node could be done as:

```yaml
step name:
  type: parallel
  next:
  on_failure:
  branches:
    branch_a:
      ...
    branch_b:
      ...
```

---
!!! Note

    This is not yet available via Python SDK.

---


### next (required)
The name of the node in the graph to go if the node succeeds

### on_failure (optional)
The name of the node in the graph to go if the node fails.
This is optional as we would move to the fail node of the graph if one is not provided.

on_failure could be an use case where you want to send a failure notification before marking the run as failure.

### branches (required)

The branches of the step that you want to parallelize. Each branch follows the same definition as a dag in itself.

### Example

```yaml
Feature generation:
  type: parallel
  next: ML training
  branches:
    One hot encoding:
      start_at: encoder
      steps:
        encoder:
          type: task
          next: success_state
          command: my_encoder.encode
        success_state:
          type: success
        fail_state:
          type: fail
    Scaler:
      start_at: scale
      steps:
        scale:
          type: task
          next: success_state
          command: my_scaler.scale
        success_state:
          type: success
        fail_state:
          type: fail
```

In the example, "One hot encoding" and "Scaler" are two branches that are defined using the same definition
language as a dag and both together form the Feature generation step of the parent dag.


---
!!! Note

    A parallel state in the dag is just a definition, the actual implementation depends upon the mode
    and the support for parallelization.
---

## Dag

Dag is a composite node which has one branch defined elsewhere. It is used to logically separate the complex details
of a pipeline into modular units. For example, a typical data science project would have a data gathering, data
cleaning, data transformation, modelling, prediction as steps. And it is understandable that these individual steps
could get complex and require many steps to function. Instead of complicating the parent pipeline, we can abstract the
individual steps into its own dag nodes.

The configuration of a dag node is:

```yaml
step name:
  type: dag
  dag_definition:
  next:
  on_failure: # optional
```

---
!!! Note

    This is not yet available in Python SDK.

---


### dag_definition

The yaml file containing the dag definition in "dag" block of the file. The dag definition should follow the same rules
as any other dag in magnus.

### next (required)
The name of the node in the graph to go if the node succeeds

### on_failure (optional)
The name of the node in the graph to go if the node fails.
This is optional as we would move to the fail node of the graph if one is not provided.


### Example

```yaml
# Parent dag
dag:
  start_at: Data Cleaning
  steps:
    Data Cleaning:
      type: dag
      next: Data Transformation
      dag_definition: data-cleaning.yaml
    Data Transformation:
      type: dag
      next: Modelling
      dag_definition: data-transformation.yaml
    Modelling:
      type: dag
      next: Success
      dag_definition: modelling.yaml
    Success:
      type: success
    Fail:
      type: fail
```

```yaml
# data-cleaning.yaml
dag:
  start_at: Remove numbers
  steps:
    Remove numbers:
      type: task
      next: Remove special characters
      command: data_cleaning.remove_numbers
    Remove special characters:
      type: dag
      next: Success
      command: data_cleaning.remove_special_characters
    Success:
      type: success
    Fail:
      type: fail
```

In this example, the parent dag only captures the high level tasks required to perform a data science experiment
while the details of how data cleaning should be done are mentioned in data-cleaning.yaml.


## Map

Map is a composite node consisting of one branch that can be iterated over a parameter. A typical use case would be
performing the same data cleaning operation on a bunch of files or the columns of a data frame. The parameter over which
the branch is iterated over should be provided and also be available to the dag at the execution time.

The configuration of the map node:
```yaml
step name:
  type: map
  iterate_on:
  iterate_as:
  next:
  on_failure: # Optional
  branch:
```

---
!!! Note

    This is not yet available in Python SDK.

---


### iterate_on (required)
The name of the parameter to iterate on. The parameter should be of type List in python and should be available in the
parameter space.

### iterate_as (required)
The name of the argument that is expected by the task.

For example:

- Set a parameter by name x which is a list [1, 2, 3]
- A python task node as part of the map dag definition expects this argument as x_i as part function signature.
- You should set ```iterate_on``` as ```x``` and ```iterate_as``` as ```x_i```

### branch (required)
The branch to iterate over the parameter. The branch definition should follow the same rules as a dag definition.

### next (required)
The name of the node in the graph to go if the node succeeds

### on_failure (optional)
The name of the node in the graph to go if the node fails.
This is optional as we would move to the fail node of the graph if one is not provided.


### Example

```yaml
dag:
  start_at: List files
  steps:
    List files:
      type: task
      next: Clean files
      command: my_module.list_files
    Clean files:
      type: map
      next: Success
      iterate_on: file_list
      iterate_as: file_name
      branch:
        start_at: Task Clean Files
        steps:
          Task Clean Files:
            type: task
            command: my_module.clean_file
            next: success
          success:
            type: success
          fail:
            type: fail
    Success:
      type: success
    Fail:
      type: fail
```

In this example dag definition,

- We start with the step *List files*, that generates a list of files to be cleaned and sets it as a parameter
- The step *Clean files* contains a branch that would be iterated over the list of files found in the previous step.

To be comprehensive, here is the stub implementations of the python code

```python
# in my_module.py

def list_files():
    file_list = ['a', 'b', 'c']
    # do some compute to figure out the actual list of files would be
    # By returning a dictionary, you can set parameters that would be available for down stream steps.
    return {'file_list' : file_list}

def clean_file(file_name):
    # Retrieve the file or use catalog to retrieve the file and perform the cleaning steps
    pass
```


---
!!! Note

    A map state in the dag is just a definition, the actual implementation depends upon the mode
    and the support for parallelization.
---

## As-Is

As-is a convenience node or a designers node. It can be used to mock nodes while designing the overall pipeline design
without implementing anything in *interactive* modes. The same node can be used to render required
templates in *orchestration* modes.

The configuration of as-is node:

```yaml
step name:
  type: as-is
  command:
  next:

```

You can have arbitrary attributes assigned to the as-is node.


### command (optional)

The command is purely optional in as-is node and even if one is provided it is not executed.

### next (required)

The name of the node in the graph to go if the node succeeds


### Example as mock node

A very typical data science project workflow could be mocked by:

```yaml
dag:
  description: A mocked data science pipeline
  start_at: Data Cleaning
  steps:
    Data Cleaning:
      type: as-ias
      next: Data Transformation
    Data Transformation:
      type: as-is
      next: Modelling
    Modelling:
      type: as-is
      next: Deploy
    Deploy:
      type: as-is
      next: Success
    Success:
      type: success
    Fail:
      type: fail
```

In this example, we only wrote a skeleton of the pipeline and none of the steps are actually implemented.

### Example as template

Taking the same example, we can imagine that there is an executor which can deploy the trained ML model and requires
a template to be generated as part of the continuos integration.

```yaml
mode:
  type: <some mode which deploys trained models>

dag:
  description: A mocked data science pipeline
  start_at: Data Cleaning
  steps:
    Data Cleaning:
      type: task
      command: my_module.data_cleaning
      next: Data Transformation
    Data Transformation:
      type: task
      command: my_module.data_transformation
      next: Modelling
    Modelling:
      type: task
      command: my_module.modelling
      next: Deploy
    Deploy:
      type: as-is
      render_string: >
         python -m my_module.check_accuracy_threshold
         cp models/trained_models to s3://<some location>

      next: Success
    Success:
      type: success
    Fail:
      type: fail
```

In *interactive* modes the as-is does not do anything and succeeds every time but the same dag in *orchestrated* modes
can render a template that could be part of continuos integration process.

Data science and ML research teams would thrive in interactive modes, given their experimental nature of work. As-Is
nodes gives a way to do experiments without changing the dag definition once it is ready to be deployed.

As-is nodes also provide a way to inject scripts as steps for orchestrators that do not support all the
features of magnus. For example, if an orchestrator mode of your liking does not support map state, you can
use as-is to inject a script that behaves like a map state and triggers all the required jobs.


## Passing data

In magnus, we classify 2 kinds of data sets that can be passed around to down stream steps.

- Data: Processed files by an upstream step should be available for downstream steps when required.
[Catalog](../catalog) provides the way to do this.

- Parameters: Any JSON serializable data can be passed to down stream steps.

### Parameters from command line


Initial parameters to the application can be sent in via a parameters file.

Example:

```shell
magnus execute --file getting-started.yaml --parameters-file parameters.yaml
```

```yaml
# in parameters.yaml
arg1 : test
arg2: dev
```

Or via environmental variables: Any environmental variable with prefix ```MAGNUS_PRM_``` is considered as a magnus
parameter. Eg: ```MAGNUS_PRM_arg1=test``` or ```MAGNUS_PRM_arg2=dev```.

---
!!! Note

    Parameters via environmental variables over-ride the parameters defined via parameters file.
---



In this case, arg1 and arg2 are available as parameters to downstream steps.

### Storing parameters

Any JSON serializable dictionary returned from a task node is available as parameters to downstream steps.

Example:

```python

def my_cool_function():
  return {'arg1': 'hello', 'arg2': {'hello', 'world'} }

```

Or

```python

from magnus import store_parameter

def my_cool_function():
  store_parameter(arg1='hello', 'arg2'={'hello', 'world'})

```

Or

```python
import os
import json

def my_cool_function():
  os.environ['MAGNUS_PRM_' + 'arg1'] = 'hello'
  os.environ['MAGNUS_PRM_' + 'arg2'] = json.dumps({'hello', 'world'})
```

All the three above ways store arg1 and arg2 for downstream steps.

### Accessing parameters

Any parameters set either at command line or by upstream nodes can be accessed by:


``` python
def my_cool_function(arg1, arg2=None):
  pass

```
The function is inspected to find all *named* args and provided value if the key exists in the parameters.

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
Calling get_parameter with no key returns all parameters.


## Extensions

You can extend and implement your own ```node_types``` by extending the ```BaseNode``` class.

The base class has the following methods with only one of the two methods to be implemented for custom implementations.

If the ```node.is_composite``` is ```True```, implement the ```execute_as_graph``` method.
If the ```node.is_composite``` is ```False```, implement the ```execute``` method.

```python
# Source code present at magnus/nodes.py
from pydantic import BaseModel

class BaseNode:
    """
    Base class with common functionality provided for a Node of a graph.

    A node of a graph could be a
        * single execution node as task, success, fail.
        * Could be graph in itself as parallel, dag and map.
        * could be a convenience function like as-is.

    The name is relative to the DAG.
    The internal name of the node, is absolute name in dot path convention.
        This has one to one mapping to the name in the run log
    The internal name of a node, should always be odd when split against dot.

    The internal branch name, only applies for branched nodes, is the branch it belongs to.
    The internal branch name should always be even when split against dot.
    """

    node_type = ''
    required_fields: List[str] = []
    errors_on: List[str] = []

    class Config(BaseModel):
        mode_config: dict = {}

        def __init__(self, *args, **kwargs):
            next_node = kwargs.get('next', '')
            if next_node:
                del kwargs['next']
                kwargs['next_node'] = next_node

            super().__init__(*args, **kwargs)


    def __init__(self, name, internal_name, config, internal_branch_name=None):

        self.name = name
        self.internal_name = internal_name  #  Dot notation naming of the steps
        self.config = self.Config(**config)
        self.internal_branch_name = internal_branch_name  # parallel, map, dag only have internal names
        self.is_composite = False


    def execute(self, executor, mock=False, map_variable: dict = None, **kwargs):
        """
        The actual function that does the execution of the command in the config.

        Should only be implemented for task, success, fail and as-is and never for
        composite nodes.

        Args:
            executor (magnus.executor.BaseExecutor): The executor mode class
            mock (bool, optional): Don't run, just pretend. Defaults to False.
            map_variable (str, optional): The value of the map iteration variable, if part of a map node.
                Defaults to ''.

        Raises:
            NotImplementedError: Base class, hence not implemented.
        """
        raise NotImplementedError

    def execute_as_graph(self, executor, map_variable: dict = None, **kwargs):
        """
        This function would be called to set up the execution of the individual
        branches of a composite node.

        Function should only be implemented for composite nodes like dag, map, parallel.

        Args:
            executor (magnus.executor.BaseExecutor): The executor mode.

        Raises:
            NotImplementedError: Base class, hence not implemented.
        """
        raise NotImplementedError
```

---
!!! Note

    The BaseNode has many other methods which are *private* and typically do not need modifications.
    The Config datamodel of the custom class should have all the attributes that are required.

---



The custom extensions should be registered as part of the namespace: ```nodes``` for it to be
loaded.

```toml
# For example, as part of your pyproject.toml
[tool.poetry.plugins."nodes"]
"mail" = "YOUR_PACKAGE:MailTeam"
```
