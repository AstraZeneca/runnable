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


The pipeline which contians one node to call the above function. 

```yaml
dag:
  description: A single node pipeline
  start_at: step 1 
  steps:
    step 1:
      type: task 
      next: success
      command: my_module.my_funcion
      command_type: python
    success:
      type: success
    failure:
      type: fail
```


## Mocking a node in pipeline

In magnus, you can skip execution of a node or mock using a node of type ```as-is```.

Example:

```yaml
dag:
  description: A single node pipeline
  start_at: step 1 
  steps:
    step 1:
      type: as-is # The function would not execute as this is as-is node
      next: success
      command: my_module.my_funcion
      command_type: python
    success:
      type: success
    failure:
      type: fail
```


## Using shell commands as part of the pipline

In magnus, a pipeline can have shell commands as part of the pipeline. The only caveat in doing so is magnus
would not be able to support ```parameters```, ```secrets``` or any of the built-in functions. The cataloging 
functionality of magnus still would work via the configuration file. 

Example:

```yaml
dag:
  description: A single node pipeline
  start_at: step 1 
  steps:
    step 1:
      type: task 
      next: success
      command: ls # You can use this to call some executable in the PATH
      command_type: shell
    success:
      type: success
    failure:
      type: fail
```


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


The pipeline which contians first_function of the above module and then to the call the second_function is given
below.

```yaml
dag:
  description: A single node pipeline
  start_at: step 1 
  steps:
    step 1:
      type: task 
      next: step 2
      command: my_module.first_funcion
      command_type: python
    step 2:
      type: task 
      next: success
      command: my_module.second_funcion
      command_type: python
    success:
      type: success
    failure:
      type: fail
```


## Using on-failure to handle errors

You can instruct magnus to traverse to a different node of the dag if the current node fails to execute. 
A non-zero exit status of the python function or shell command is considered a failure. 

The default behavior in case of a failure of a node is, if no ```on_failure``` is defined, is to traverse to the ```fail``` node of the graph and mark the execution of the dag as failure.

The execution of a dag is considered failure if and only if the ```fail``` node of the graph is reached.

```python
# In my_module.py

def first_function():
    print('In the function, first_function of my_module')


def second_function():
    print('In the function, second_function of my_module')


def handle_error():
    print('Send an email notifiction')
    ## Some logic to send error notification
    ...

```

The pipeline definition to call ```my_module.handle_error``` in case of a failure of any node is defined below.


```yaml
dag:
  description: A single node pipeline
  start_at: step 1 
  steps:
    step 1:
      type: task 
      next: step 2
      command: my_module.first_funcion
      command_type: python
      on_failure: graceful exit
    step 2:
      type: task 
      next: success
      command: my_module.second_funcion
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

## Passing parameters between nodes

### Pythonically

### Using in-built functions

## Passing parameters to the first node of the pipeline

### Pythonically

### Using in-built functions

## Using the catalog to pass artifacts between nodes

### Using the configuration.

### Using the in-built functions

## Accessing secrets within code.

## Parallel node

### Enabling parallel execution

## Embedding dag within dag

## Looping a dag execution over a iterable parameter

### Enabling parallel execution

## Nesting and complex dags

## Controlling the log level of magnus

## Something about configurations
