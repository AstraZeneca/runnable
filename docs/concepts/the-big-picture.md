runnable revolves around the concept of pipelines or workflows and tasks that happen within them.

---

A [workflow](pipeline.md) is simply a series of steps that you want to execute for a desired outcome.

``` mermaid
%%{ init: { 'flowchart': { 'curve': 'linear' } } }%%
flowchart LR

    step1:::green
    step1([Step 1]) --> step2:::green
    step2([Step 2]) --> step3:::green
    step3([Step .. ]) --> step4:::green
    step4([Step n]) --> suc([success]):::green

    classDef green stroke:#0f0

```

To define a workflow, we need:

- [List of steps](pipeline.md/#steps)
- a [starting step](pipeline.md/#start_at)
- Next step

    - [In case of success](pipeline.md/#linking)
    - [In case of failure](pipeline.md/#on_failure)

- [Terminating](pipeline.md/#terminating)

The workflow can be defined either in ```yaml``` or using the [```python sdk```](../sdk.md).

---

A step in the workflow can be:


=== "task"

    A step in the workflow that does a logical unit work.

    The unit of work can be a [python function](task.md/#python_functions),
    a [shell script](task.md/#shell) or a
    [notebook](task.md/#notebook).

    A task can live in isolation and be independent of the workflow.

    All the logs, i.e stderr and stdout or executed notebooks are stored
    in [catalog](catalog.md) for easier access and debugging.



=== "stub"

    An [abstract step](stub.md) that is not yet fully implemented.

    For example in python:

    ```python
    def do_something():
        pass
    ```


=== "parallel"

    A step that has a definite number of [parallel workflows](parallel.md) executing
     simultaneously.

     In the below visualisation, the green lined steps happen in sequence and wait for the previous step to
    successfully complete.

    The branches lined in yellow run in parallel to each other but sequential within the branch.

    ```mermaid
    flowchart TD

        getFeatures([Get Features]):::green
        trainStep(Train Models):::green
        ensembleModel([Ensemble Modelling]):::green
        inference([Run Inference]):::green
        success([Success]):::green

        prepareXG([Prepare for XGBoost]):::yellow
        trainXG([Train XGBoost]):::yellow
        successXG([XGBoost success]):::yellow
        prepareXG --> trainXG --> successXG

        trainRF([Train RF model]):::yellow
        successRF([RF Model success]):::yellow
        trainRF --> successRF


        getFeatures --> trainStep
        trainStep --> prepareXG
        trainStep --> trainRF
        successXG --> ensembleModel
        successRF --> ensembleModel
        ensembleModel --> inference
        inference --> success


        classDef yellow stroke:#FFFF00
        classDef green stroke:#0f0


    ```


=== "map"

    A step that executes a workflow over an [iterable parameter](map.md).

    The step "chunk files" identifies the number of files to process and computes the start index of every
    batch of files to process for a chunk size of 10, the stride.

    "Process Chunk" pipelines are then triggered in parallel to process the chunk of files between ```start index```
    and ```start index + stride```

    ```mermaid
    flowchart TD
    chunkify([Chunk files]):::green
    success([Success]):::green

    subgraph one[Process Chunk]
        process_chunk1([Process Chunk]):::yellow
        success_chunk1([Success]):::yellow

        process_chunk1 --> success_chunk1
    end

    subgraph two[Process Chunk]
        process_chunk2([Process Chunk]):::yellow
        success_chunk2([Success]):::yellow

        process_chunk2 --> success_chunk2
    end

    subgraph three[Process Chunk]
        process_chunk3([Process Chunk]):::yellow
        success_chunk3([Success]):::yellow

        process_chunk3 --> success_chunk3
    end

    subgraph four[Process Chunk]
        process_chunk4([Process Chunk]):::yellow
        success_chunk4([Success]):::yellow

        process_chunk4 --> success_chunk4
    end

    subgraph five[Process Chunk]
        process_chunk5([Process Chunk]):::yellow
        success_chunk5([Success]):::yellow

        process_chunk5 --> success_chunk5
    end



    chunkify -- (stride=10, start_index=0)--> one --> success
    chunkify -- (stride=10, start_index=10)--> two --> success
    chunkify -- (stride=10, start_index=20)--> three --> success
    chunkify -- (stride=10, start_index=30)--> four --> success
    chunkify -- (stride=10, start_index=40)--> five --> success

    classDef yellow stroke:#FFFF00
    classDef green stroke:#0f0
    ```



---

A [step type of task](task.md) is the functional unit of the pipeline.

To be useful, it can:

- Access parameters

    - Either [defined statically](parameters.md/#initial_parameters) at the start of the
    pipeline
    - Or by [upstream steps](parameters.md/#parameters_flow)

- [Publish or retrieve artifacts](catalog.md) from/to other steps.
<!-- - [Publish metrics](experiment-tracking.md) that are interesting. -->
- Have [access to secrets](secrets.md).

All the above functionality is possible naturally with no intrusion into code base.

---

All executions of the pipeline should be:

- [Reproducible](run-log.md) for audit and data lineage purposes.
- Runnable in local environments for
[debugging failed runs](run-log.md/#retrying_failures).

---

Executions of pipeline should be scalable and use the infrastructure at
your disposal efficiently.

We achieve this by adding [one configuration file](../configurations/overview.md), rather than
changing the application code.
