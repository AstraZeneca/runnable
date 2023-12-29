Parallel nodes in magnus allows you to run multiple pipelines in parallel and use your compute resources efficiently.

## Example

!!! note "Only stubs?"

    All the steps in the below example are ```stubbed``` for convenience. The functionality is similar
    even if the steps are execution units like ```tasks``` or any other nodes.

    We support deeply nested steps. For example, a step in the parallel branch can be a ```map``` which internally
    loops over a ```dag``` and so on. Though this functionality is useful, it can be difficult to debug and
    understand in large code bases.

Below is a stubbed out example of a pipeline that trains two models in parallel and create an ensemble model to
do the inference. The models XGBoost and Random Forest (RF model) are trained in parallel and training of the
ensemble model happens only after both models are (successfully) trained.

=== "Visualisation"

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

=== "Pipeline in yaml"

    ```yaml linenums="1"
    --8<-- "examples/concepts/parallel.yaml"
    ```

=== "python sdk"

    ```python linenums="1"
    --8<-- "examples/concepts/parallel.py"
    ```

All pipelines, nested or parent, have the same structure as defined in
[pipeline definition](../pipeline).

The parent pipeline defines a step ```Train models``` which is a parallel step.
The branches, XGBoost and RF model, are pipelines themselves.

## Traversal

A branch of a parallel step is considered success only if the ```success``` step is reached at the end.
The steps of the pipeline can fail and be handled by [on failure](../concepts/ppiline/on_failure) and
redirected to ```success``` if that is the desired behavior.

The parallel step is considered successful only if all the branches of the step have terminated successfully.


## Parameters

All the tasks defined in the branches of the parallel pipeline can
[access to parameters and data as usual](../task).


!!! warning

    The parameters can be updated by all the tasks and the last task to execute overwrites
    the previous changes.

    Since the order of execution is not guaranteed, its best to avoid mutating the same parameters in
    the steps belonging to parallel step.
