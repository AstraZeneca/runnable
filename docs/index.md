# Runnable

<figure markdown>
  ![Image title](assets/sport.png){ width="200" height="100"}
  <figcaption>Orchestrate your functions, notebooks, scripts anywhere!!</figcaption>
</figure>

<span style="font-size:0.75em;">
<a href="https://www.flaticon.com/free-icons/runner" title="runner icons">Runner icons created by Leremy - Flaticon</a>
</span>
---

## Example

The data science specific code is a well-known
[iris example from scikit-learn](https://scikit-learn.org/stable/auto_examples/linear_model/plot_iris_logistic.html).


```python linenums="1"
--8<-- "examples/iris_demo.py"
```


1. Return objects X and Y.
2. Store the file `iris_logistic.png` for future reference.
3. Define the sequence of tasks.
4. Define a pipeline with the tasks

The difference between native driver and runnable orchestration:

!!! tip inline end "Notebooks and Shell scripts"

    You can execute notebooks and shell scripts too!!

    They can be written just as you would want them, *plain old notebooks and scripts*.




<div class="annotate" markdown>

```diff

- X, Y = load_data()
+load_data_task = PythonTask(
+    function=load_data,
+     name="load_data",
+     returns=[pickled("X"), pickled("Y")], (1)
+    )

-logreg = model_fit(X, Y, C=1.0)
+model_fit_task = PythonTask(
+   function=model_fit,
+   name="model_fit",
+   returns=[pickled("logreg")],
+   )

-generate_plots(X, Y, logreg)
+generate_plots_task = PythonTask(
+   function=generate_plots,
+   name="generate_plots",
+   terminate_with_success=True,
+   catalog=Catalog(put=["iris_logistic.png"]), (2)
+   )


+pipeline = Pipeline(
+   steps=[load_data_task, model_fit_task, generate_plots_task], (3)

```
</div>

1. Return objects X and Y.
2. Store the file `iris_logistic.png` for future reference.
3. Define the sequence of tasks.

---

- [x] Absolutely no change in data science code to make it `runnable`
- [x] The ```driver``` function has an equivalent and intuitive runnable expression
- [x] Reproducible by default, runnable stores metadata about code/data/config for every execution.
- [x] The pipeline is `runnable` in any environment.


## But why runnable?

Obviously, there are a lot of orchestration tools in python. A well maintained and curated [list is
available here](https://github.com/EthicalML/awesome-production-machine-learning/).

Below is a rough comparison of `runnable` to others.


|Feature|runnable|Airflow|Argo workflows|Metaflow|ZenML|Kedro|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|Cross platform|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|:white_check_mark:|
|Bring your own infrastructure |:white_check_mark:|:x:|:x:|:x:|:x:|:white_check_mark:|
|Local executions|:white_check_mark:|:x:|:x:|:white_check_mark:|:white_check_mark:|:white_check_mark:|
|Bring your own code|:white_check_mark:|:x:|:x:|:x:|:x:|:x:|
|Reproducibility of executions|:white_check_mark:|:x:|:x:|:white_check_mark:|:white_check_mark:|:white_check_mark:|
|Easy to move on|:white_check_mark:|:X:|:x:|:x:|:x:|:white_check_mark:|
|End to end platform|:x:|:white_check_mark:|:x:|:white_check_mark:|:white_check_mark:|:x:|
|Task level orchestration|:x:|:white_check_mark:|:white_check_mark:|:x:|:x:|:x:|
|Notebook as tasks|:white_check_mark:|:x:|:x:|:x:|:x:|:x:|
|Unit testable pipelines|:white_check_mark:|:x:|:x:|:white_check_mark:|:white_check_mark:|:x:|
|Multi language support|:white_check_mark:|:white_check_mark:|:white_check_mark:|:X:|:x:|:x:|




They can be broadly classified in three categories:

- __Native orchestrators__: These orchestrators are responsible for task level orchestration,
resource management on chosen infrastructure. Examples:

    - Airflow
    - Argo workflows
    - AWS step functions


#### runnable is complimentary to these orchestrators and is designed to enable data teams use them effectively.

- __Platforms__: These are meant to provide end to end platform for training, deploying and
serving of ML models. Examples:

    - Dagster
    - Prefect
    - Flyte

    They have specific infrastructure requirements and are great if the entire organization buys into
    their philosophy and ways of working.

#### runnable is designed to work with your infrastructure and ways of working instead of dictating them.



- __Meta orchestrators__: Orchestrators using the native orchestrators but provide a simplified
SDK tailored for typical data oriented tasks. Examples include:

    - Kedro: cross platform transpiler.
    - Metaflow: A mix of platform and SDK.
    - ZenML: A mix of platform and SDK.

runnable is a _meta orchestrator_ with different design decisions.


<div class="grid cards" markdown>

-   :material-clock-fast:{ .lg .middle } __Easy to adopt, its mostly your code__

    ---

    Your application code remains as it is. Runnable exists outside of it.

    - No API's or decorators or imposed structure.
    - Most often it is a single file.

    [:octicons-arrow-right-24: Getting started](concepts/the-big-picture.md)

-    :building_construction:{ .lg .middle } __Bring your infrastructure__

    ---

    Runnable can be adapted to your infrastructure stack instead of dictating it.

    - Intentionally minimal in scope as a composer of pipelines in native orchestrators.
    - Every execution is ready to be deployed to production.

    [:octicons-arrow-right-24: Infrastructure](configurations/overview.md)

-   :memo:{ .lg .middle } __Reproducibility__

    ---

    Runnable tracks key information to reproduce the execution. All this happens without
    any additional code.

    [:octicons-arrow-right-24: Run Log](concepts/run-log.md)



-   :repeat:{ .lg .middle } __Retry failues__

    ---

    Debug any failure in your local development environment.

    [:octicons-arrow-right-24: Retry](#)

-   :microscope:{ .lg .middle } __Testing__

    ---

    Unit test your code and pipelines.

    [:octicons-arrow-right-24: Test](#)



-   :broken_heart:{ .lg .middle } __Move on__

    ---

    Moving away from runnable is as simple as deleting relevant files.


</div>
