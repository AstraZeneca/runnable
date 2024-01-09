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

    ```mermaid
    flowchart TD

        step1([Step 1]):::green
        parallel(Parallel Step):::green
        step3([Step 3]):::green
        success([Success]):::green

        step211([Branch 1 Step 1]):::yellow
        step212([Branch 1 Step 2]):::yellow
        branch1Suc([Branch 1 Success]):::yellow
        step211 --> step212 --> branch1Suc

        step22([Branch 2 Step 1]):::yellow
        branch2Suc([Branch 2 Success]):::yellow
        step22 --> branch2Suc


        step1 --> parallel
        parallel --> step211
        parallel --> step22
        branch1Suc --> step3
        branch2Suc --> step3
        step3 --> success

        classDef yellow stroke:#FFFF00
        classDef green stroke:#0f0


    ```


=== "pipeline"


=== "map"



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
