# Closer look at input

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

## dag

A [directed acyclic graph (dag)](../../concepts/dag) is the definition of the work you want to perform. 
It defines a series of [nodes](../../concepts/nodes) and the rules of traversal between them. 

## Traversal of the dag

In magnus, the order of steps in the dag definition is not important. The traversal is as follows:

1. We start at start_at of the dag, which is "step parameters".
2. If "step parameters" successfully completed we move to *next* of "step parameters", which is "step shell".
3. if "step parameters" failed, we move to the failure node of the dag (fail). The step definition can over-ride this. 
4. We stop traversing once we reach one of success or fail nodes.

All dag definitions should have a *success* node and *fail* node. A step/node in the dag defines the next node to 
visit in the *next* section of the definition. A step/node can also define the next node to visit *on failure* 
of the node, if one is not provided we default to the fail node of the dag. 

You can also provide the maximum run time for a step or the entire dag in the definition. More information of all 
the features is [available here](../../concepts/dag). 


## Step/Node

A Step/Node defines a single *logical unit of work* in the dag. 

In the example, we use three different type of nodes:

### task

  Is some callable/executable code. 
  Python functions are default and fully [supported tasks](../../concepts/nodes/#task).

  As shown in the example, you can also use python lambda expressions with task type of python-lambda.
  
  Or shell with a caveat that [any interactions](../brief-concepts-output/#interaction_in_code) with magnus or 
  secret management within magnus is not possible.

### success
  
  A [node](../../concepts/nodes/#success) that marks the graph/sub-graph as success.

### fail
  
  A [node](../../concepts/nodes/#fail) that marks the graph/sub-graph as fail.


You can define more [complex node types (parallel, embedded dag, map) too](../../concepts/nodes/#nodes). 

## Parameters

Any unreserved keyword argument sent to command line would be added to the parameter 
space but as a string. For example, the command used to execute the pipeline,

```magnus execute --file getting-started.yaml --x 3```

sets the parameter *x* as "3".  

The lambda expression, ```lambda x: {'x': int(x) + 1}```, then can use the parameter and update it 
(in this case, x = x + 1 = 4) by returning a dictionary. The [parameter space](../../concepts/nodes/#passing_data) 
is updated with the key-value pair. Parameters can be passed to python functions using a similar logic.

Shell executions have access to the parameters too with key being prefixed by MAGNUS_PRM_. Any JSON serializable
key-value pairs can be used. You can confirm this by searching for ```MAGNUS_PRM_``` in ```data/data.txt```.

For larger content/files, please use the data [catalog](../../concepts/catalog) 
functionality.

---
!!! Note

    All parameter keys are case insensitive and the case is changed to lower to support Windows.
    Please read more information [here](https://stackoverflow.com/questions/19023238/why-python-uppercases-all-environment-variables-in-windows).

---


## Catalog

Catalog is a way to pass data files across nodes and also serves as a way to track data used/generated as part of the
execution. In the following instruction:

```yaml
step shell:
  type: task
  command_type: shell
  command: mkdir data ; env >> data/data.txt # For Linux/macOS
  #command: mkdir data
  next: success
  catalog:
    put:
      - "*"
```

we are instructing magnus to create a ```data``` folder and echo the environmental variables into ```data.txt``` in 
the command section while asking magnus to put the files the catalog after execution.

Logically, you can instruct magnus to:

- ```get``` files from the catalog before the execution to a specific ```compute data folder```
- execute the command 
- ```put``` the files from the ```compute data folder``` to the catalog.

By default, magnus would look into ```data``` folder but you can over-ride this by providing ```compute_folder``` in the
config. Glob patterns for file searching are allowed. Please read more about the catalog [here](../../concepts/catalog).