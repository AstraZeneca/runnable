# Closer look at input

---

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

## dag

A [directed acylic graph (dag)](../../concepts/dag) is the definition of the work you want to perform. 
It defines a series of [nodes](../../concepts/nodes) and the rules of traversal between them. 

In magnus, this dag is evaluated in two distinct ways

* Interactive: Where the dag is executed in a compute environment. The compute environment could be 
local or off-loaded to cloud.
* Orchestrated: Where the dag is used to render a 3rd party pipeline definition like AWS Step functions, 
K8's job specifications or argo workflows.

Independent of how the dag is evaluated or executed, the dag definition remains the same. In some sense, 
we see the dag definition as a **contract** between the data science team and ML engineering team. 

Data science teams should feel free to experiment and find a pipeline that works for them. Given that the 
pipeline is a simple yaml file, it can be version controlled and easily diff-able. 

The engineering team, responsible for productionizing the pipelines, can focus on the best tools to deploy 
the application in the infrastructure they have. 

This seperation of concern frees up both teams to do what they do best without inhibiting experimentation or 
engineering robustness. Though magnus comes with a few in-built evaluation **modes**, it is easy to 
write [extensions](../../concepts/modes-implementations/extensions/) that work for your team.

## Traversal of the dag

In magnus, the order of steps in the dag definition is not important. The traversal is as follows:

1. We start at start_at of the dag, which is step1.
2. If step1 successfully completed we move to "next" of step1, which is step2.
3. if step1 failed, we move to the failure node of the dag (step3). The step definition can over-ride this. 
4. We stop traversing once we reach one of success or fail nodes.

All dag definitions should have a success node and fail node. A step/node in the dag defines the next node to 
visit in the *next* section of the definition. A step/node can also define the next node to visit *on failure* 
of the node, if one is not provided we default to the fail node of the dag. 

You can also provide the maximum run time for a step or the entire dag in the definition. More information of all 
the features is [availble here](../../concepts/dag). 


## Step/Node

A Step/Node defines a single *logical unit of work* in the dag. 

Currently, we support the following types of nodes:

### task

  Is some callable/executable code. Python functions are default and fully [supported tasks](../../concepts/nodes/#task). It is also possible to use shell commands as task commands like below but [any interactions](../brief-concepts-output/#interaction_in_code) with magnus or secret management within magnus are not possible.

Example: For the definition like below, we call *my_cool_function()* of *my_python_code* while executing the node.

``` yaml
step1:
  type: task
  commmand: my_python_code.my_cool_function
  command_type: python
```

Functions can have arguments and they are resolved dynamically by looking into the [parameter space](../../concepts/nodes/#passing_data).

Example of a shell command as a task.

```yaml
step1:
  type: task
  command: some-shell-command
  command_type: shell
```

### dag
  
  is in itself a dag defined elsewhere. This [node type](../../concepts/nodes/#dag) could be useful in breaking down a complex pipeline 
  into smaller, logically seperate blocks of work. 

Example:
``` yaml
step1:
  type: task
  dag_definition: another_pipeline.yaml
```

The dag defined in *another_pipeline.yaml* is evaluated during the evaluation of step1.

---
!!! Note

    Nesting of composite nodes is currently not supported and being actively develpoed.

---

### parallel

  contains many sub-graphs as part of the node. This [node type](../../concepts/nodes/#parallel) is useful when you 
  know before-hand all the parallel computations that could be done. Eg: fixed set of feature transformations.

Example:
``` yaml
step1:
  type: parallel
  branches:
    branch_a:
      ...
    branch_b:
      ...
```

branch_a and branch_b are sub-graphs that follow the same definition language as the main graph. 

---
!!! Note

    Nesting of composite nodes is currently not supported and being actively develpoed.

---

### map
  
  contains a single branch that is dynamically iterated on a parameter. This [node](../../concepts/nodes/#map) is 
  useful when you want to repeat the same pipeline over a list. Eg: Performing a cleaning process over many files.

Example:
``` yaml
step1:
  type: map
  iterate_on: y
  branch:
    ...
```
Here branch follows the same definition language as the graph. The *parameter* y should be initilized by 
either the previous steps or part of input. 

---
!!! Note

    Nesting of composite nodes is currently not supported and being actively develpoed.

---

### as-is
  
  is a convenience node and is always marked as success. [As-is nodes](../../concepts/nodes/#as-is) are handy in 
  stubbing/mocking nodes when designing the pipeline. Another use-case is to use these nodes differently 
  in interactive modes while render a "glue" code as part of orchestration mode. 

Example:
``` yaml
step1:
  type: as-is
  render_string: >
    {
     "Type": "Task",
     "Resource": "arn:aws:states:::sns:publish",
     "Parameters": {
       "TopicArn": "arn:aws:sns:us-east-1:<account_id>:myTopic",
       "Message": "Hello",
       "MessageAttributes": {
         "my attribute no 1": {
           "DataType": "String",
           "StringValue": "value of my attribute no 1"
         },
         "my attribute no 2": {
           "DataType": "String",
           "StringValue": "value of my attribute no 2"
         }
       }
      }
```

In this example, the node does nothing in interactive modes (and therefore not interferring with experimentation) 
while during the actual process of rendering an AWS Step function can put the block of code 
required to send a notification. 

### success
  
  A [node](../../concepts/nodes/#success) that marks the graph/sub-graph as success.

### fail
  
  A [node](../../concepts/nodes/#fail) that marks the graph/sub-graph as fail.

