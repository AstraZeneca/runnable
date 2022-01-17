# Guide to Extensions

Magnus was written with complete understanding that we alone cannot cater to all the team needs. All
modules of magnus are written with extensibility as a non-negotiable design guideline. 

Before you want to extend the capabilities, lets have a look at the flow of code.
## Flow of code

A dag execution involves two different phases:

- **Traversal of the dag**: In this phase, we are concerned about how to traverse the dag and reach different
nodes of the graph. Lets call the compute environment of the dag traversal *CE-Traversal*.

- **Execution of the nodes**: In this phase, we are concerned about executing the node of the graph i.e, making the parameters accessible for the dag, making the catalog or secrets available for node execution. Lets call the compute environment of the node execution, *CE-Execution*.

With this set-up, we have 4 different possibilites.

- CE-Traversal by magnus, CE-Execution by magnus: In this mode of execution, magnus handles both the traversal and execution of the nodes and the compute environment for both the phases are the same, for example: *local* compute mode.

- CE-Traversal by mangus, CE-Execution by mangus but different environments:


## Configuraton

## Submitting Community Extensions