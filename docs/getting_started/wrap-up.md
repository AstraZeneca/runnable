# Wrapping up

To summarize the journey so-far,

- We have defined a simple pipeline to show the features of magnus.

    - DAG definition and traversal rules.
    - Passing data between nodes.
    - Basic task types (task, success, fail) and execution environments (shell, python-lambda).
    - Data catalogs.

- We have executed the pipeline in ```local``` environment to demonstrate

    - The run log structure and its relation to the dag steps.
    - The config identities (run_config, dag hash).
    - The code identities (code commits).
    - The data catalog and data identity (data hash).

- We also "deployed" the pipeline as a bash script to demonstrate

    - Translation of the dag definition into language that compute environments understands.
    - proven the identical structure of run log/catalog independent of the environment.
    - proven the only change required to deploy is a config i.e no change in code/dag definition.

## Design

The design thought behind magnus has always been to **not** disturb the coding/engineering practices of the data teams 
or the infrastructure teams. We found the right abstraction layer to make the communication between these teams to be
the DAG definition i.e

- The data teams should focus on delivering and proving the correctness of the dag in environments that are friendly
to them. These could be ```local``` or any other environments that are experiment-friendly. 

- The infrastructure teams should focus on deploying the dag definition in production grade environments as per their
team practices or capabilities.

While both teams are looking at the same dag definition, their interpretation of it is different and should be 
decoupled. While the [example shown](../example-deployment/) is trivial, the rationale and the process of translating dag definitions is not very
far away from real world examples. 

## Testing

We also agree with dagster's observation of ["Data applications are notoriously difficult to test and are therefore 
often un- or under-tested."](https://docs.dagster.io/tutorial/intro-tutorial/testable)

In magnus, ```python``` commands are just regular functions that can be unit tested as the data teams chose to. 

Magnus itself is unit tested with a test coverage closer to 80% and with a lot of scenarios tested where we have noticed
failures in the past.

## Conclusion

We hope you got a good introduction to magnus and its features. We did not complicate the pipeline to keep it simple 
but there are many features that are interesting and might be of use to you in writing a robust pipeline. 

You can read about them in [concepts](../../concepts/nodes) or see [examples](../../examples/).

You can even write [extensions](../../extensions/extensions) to magnus to see a feature that we 
have not implemented.