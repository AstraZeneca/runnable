Magnus revolves around the concept of pipelines or workflows and tasks that happen within them.

---

A [workflow](../pipeline) is simply a series of steps that you want to execute for a desired outcome.

``` mermaid
%%{ init: { 'flowchart': { 'curve': 'linear' } } }%%
flowchart LR

    step1:::green
    step1([Step 1]) --> step2:::green
    step2([Step 2]) --> step3:::green
    step3([Step ...]) --> step4:::green
    step4([Step n]) --> suc([success]):::green

    classDef green stroke:#0f0

```

To define a workflow, we need:

- [List of steps](../pipeline/#steps)
- [starting step](../pipeline/#start_at)
- Next step

    - [In case of success](../pipeline/#linking)
    - [In case of failure](../pipeline/#on_failure)

- [Terminating](../pipeline/terminating)

The workflow can be defined either in ```yaml``` or using the ```python sdk```.

---

A step in the workflow can be:


=== "task"

    A step in the workflow that does a logical unit work.

    The unit of work can be a [python function](../task/#python_functions),
    a [shell script](../task/#shell) or a
    [notebook](../task/#notebook).

    All the logs, i.e stderr and stdout or executed notebooks are stored
    in [catalog](../catalog) for easier access and debugging.



=== "stub"

    An [abstract step](../stub) that is not yet fully implemented.

    For example in python:

    ```python
    def do_something():
        pass
    ```


=== "parallel"

    A step that has a defined number of [parallel workflows](../parallel) executing
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

A [step type of task](../task) is the functional unit of the pipeline.

To be useful, it can:

- Access parameters

    - Either [defined statically](../parameters/#initial_parameters) at the start of the
    pipeline
    - Or by [upstream steps](../parameters/#parameters_flow)

- [Publish or retrieve artifacts](../catalog) from/to other steps.
- [Publish metrics](../experiment-tracking) that are interesting.
- Have [access to secrets](../secrets).

All the above functionality is possible either via:

- Non intrusive ways: Your code does not have anything specific to magnus.

    - Application native way.
    - Or via environment variables.

- Or via the [python API](../interactions) which involves ```importing magnus``` in your code.

---

All executions of the pipeline should be:

- [Reproducible](../run-log) for audit and data lineage purposes.
- Runnable at local environments for
[debugging failed runs](../run-log/#retrying_failures).

---

Executions of pipeline should be scalable and use the infrastructure at
your disposal efficiently.

Pipelines should be portable between different infrastructure patterns.
Infrastructure patterns change all the time and
so are the demands from the infrastructure.

We achieve this by [changing configurations](../../configurations/overview), rather than
changing the application code.

For example a pipeline should be able to run:

- Local machines + local file system for data + database for logs + mlflow for experiment
tracking.
