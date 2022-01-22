# Guide to Extensions

The idea behind magnus, in simple terms, is to decouple ```what``` should be done to ```how``` it is implemented.
So while  dag only defines the ```what``` part of the equation while different compute modes (along with services)
define how to make it happen.

All the services (compute modes, run log store, secrets, catalog) are written to align to the principle. All the
interactions with the services only happen via defined API's that all implementations of the service should implement.
The ```Base``` class of all the services are given the most general implementations to make extensions as easy as
possible.

## The quadrant of possibilities

Any dag execution has two distinct phases

- Traversal of the dag: In this phase, we are only interested in traversal rules of the dag.
- Execution of the node: In this phase, we are only interested in executing a specific node.

We can characterize a pipeline execution engine by asking two questions

- Who is responsible for the dag traversal?
- Is the compute environment same as traversal environment?

Taking the example of AWS step functions, AWS Step function is responsible for traversal and the compute environment is
not the same as traversal environment as most of the states relate to some kind of compute provided by AWS. The state
machine or workflow engine keeps track of the jobs in the compute environment or has some event based mechanism to
trigger the traversal after the node finishes execution.

Asking the same questions in the context of magnus, gives us 4 possible choices.

- Magnus traverses, execution environment same as traversal.
- Magnus traverses, execution environment not same as traversal.
- Magnus does not traverse, execution environment not same as traversal.
- Magnus does not traverse, execution environment same as traversal.

Magnus is designed to handle all the above 4 choices which makes the decoupling possible.

### ***Magnus traverses, execution environment same as traversal.***

There is only one possible way this can happen with magnus, i.e with ```local``` compute mode. Since both the traversal
and execution are the same, there is no change in configuration of services for execution and traversal.

### ***Magnus traverses, the execution environment != traversal***

In this mode, magnus is responsible for traversal of the graph but triggers the actual execution to happen in a
different environment to the traversal of the graph. For example, ```local-container``` mode, magnus is responsible
for traversal of the graph but spins up a container with the instruction to execute the node. Since the traversal
and execution environments are different, the configuration of services have to modified for execution and traversal.
This is implemented by using an [*Integration* pattern](../../concepts/integration/) that can be provided to control
the configuration during both phases.

Nearly all the other dag execution engines fall in this space. For example, AWS step functions or argo have a central
server that traverses the graph but the execution of the nodes happen in some containers or compute of the AWS.
We call this as *centralized executor*.

Interestingly, in magnus there are two ways to handle this scenario:

- Just like AWS Step functions or argo workflows, we can have a *centralized executor* which triggers the execution of
    nodes in the environment that the user wants. For example, ```local-container```.
- Since the dag definition is part of the source code, every node of the graph is fully aware of the whole graph. This
    enables some compute modes to let the execution environment decide the next job to trigger based on the status of
    the execution of the current node. We call this as *decentralized executor*.

    Detailed use case: We have internally tested an *magnus-extension*, that

    - Traverses the graph and triggers an AWS Batch job for the first node from the local computer.
    - The AWS Batch job role is given enough privileges to trigger another AWS Batch job from within the batch job.
    - After the execution of the first node in AWS Batch, read the dag definition to find the next node to trigger
    and sets up the AWS batch job accordingly.
    - The graph traversal ends when one of ```success``` nodes or ```fail``` nodes have reached.

The compute extension, ```local-aws-batch``` is planned to be released along with other *magnus-extensions*.

In our opinion, *decentralized executors* are ideal for experimentation phase as there could as many dag definitions as
needed by the team without blocking one another or causing merge conflicts. Since the compute can also be off-loaded to
compute providers, it does not block their local computers.


### ***Magnus does not traverse, execution environment not same as traversal.***

In this mode, magnus does not traverse the graph but translates the dag definition to something that the ```executor```
of user's choice. For example, ```demo-renderer``` mode available as part of the ```magnus-core``` package translates
a dag definition to a bash script, although technically the execution environment is same as the traversal in this
specific example.

Since the traversal and execution environments are different, the configuration of services have to modified for
execution and traversal. This is implemented by using an [*Integration* pattern](../../concepts/integration/)
that can be provided to control the configuration during both phases.

The actual execution of the step is still wrapped around by magnus, like in
[```demo renderer```](../../getting_started/example-deployment/).

The design process behind this is abstract the infrastructure or engineering processes behind production grade
deployments from the data science teams. This abstraction also lets the engineering teams continuously improve/test
different deployment patterns without disturbing the data science team.

The compute extension, ```aws-step-functions``` is planned to be released along with other *magnus-extensions*.

### ***Magnus does not traverse, execution environment same as traversal***

In this mode, the dag definition is translated into something that the ```executor``` of user's choice and we use
the ```as-is``` node to inject scripts that are beyond the control of magnus. An example of this behavior is shown
[here](../../examples/#advanced_use_as-is), where the ```render_string``` of ```as-is``` is used to inject scripts.

The design process is to provide the best possible chance for the dag definition to remain the same independent upon
the mode of execution.

## Submitting Community Extensions

We absolutely love community extensions to magnus and would also provide support in cases of complex extensions.

For all the extensions, you should also provide integration pattern between some of the magnus core compute patterns.
As of this writing, we consider ```local```, ```local-container``` and ```demo-renderer``` as core compute patterns and
we would be adding more to the list as we get more mature.
