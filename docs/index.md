# Runnable

<figure markdown>
  ![Image title](assets/sport.png){ width="200" height="100"}
  <figcaption>Orchestrate your functions, notebooks, scripts anywhere!!</figcaption>
</figure>

<span style="font-size:0.75em;">
<a href="https://www.flaticon.com/free-icons/runner" title="runner icons">Runner icons created by Leremy - Flaticon</a>
</span>


<hr style="border:2px dotted orange">

## Example

The below data science flavored code is a well-known
[iris example from scikit-learn](https://scikit-learn.org/stable/auto_examples/linear_model/plot_iris_logistic.html).


```python linenums="1"
--8<-- "examples/iris_demo.py"
```


1. Return two serialized objects X and Y.
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


---

- [x] ```Domain``` code remains completely independent of ```driver``` code.
- [x] The ```driver``` function has an equivalent and intuitive runnable expression
- [x] Reproducible by default, runnable stores metadata about code/data/config for every execution.
- [x] The pipeline is `runnable` in any environment.

<hr style="border:2px dotted orange">

## Why runnable?

Obviously, there are a lot of orchestration tools. A well maintained and curated [list is
available here](https://github.com/EthicalML/awesome-production-machine-learning/).

Broadly, they could be classed into ```native``` or ```meta``` orchestrators.

<figure markdown>
  ![Image title](assets/work_light.png#only-light){ width="600" height="300"}
  ![Image title](assets/work_dark.png#only-dark){ width="600" height="300"}
</figure>


### __native orchestrators__

- Focus on resource management, job scheduling, robustness and scalability.
- Have less features on domain (data engineering, data science) activities.
- Difficult to run locally.
- Not ideal for quick experimentation or research activities.

### __meta orchestrators__

- An abstraction over native orchestrators.
- Oriented towards domain (data engineering, data science) features.
- Easy to get started and run locally.
- Ideal for quick experimentation or research activities.

```runnable``` is a _meta_ orchestrator with simple API, geared towards data engineering, data science projects.
It works in conjunction with _native_ orchestrators and an alternative to [kedro](https://docs.kedro.org/en/stable/index.html)
or [metaflow](https://metaflow.org/).

```runnable``` could also function as an SDK for _native_ orchestrators as it always compiles pipeline definitions
to _native_ orchestrators.

<br>

<div class="grid cards" markdown>

-   :material-clock-fast:{ .lg .middle } __Easy to adopt, its mostly your code__

    ---

    Your application code remains as it is. Runnable exists outside of it.

    - No API's or decorators or any imposed structure.

    [:octicons-arrow-right-24: Getting started](concepts/index.md)

-    :building_construction:{ .lg .middle } __Bring your infrastructure__

    ---

    ```runnable``` is not a platform. It works with your platforms.

    - ```runnable``` composes pipeline definitions suited to your infrastructure.

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

    - mock/patch the steps of the pipeline
    - test your functions as you normally do.

    [:octicons-arrow-right-24: Test](#)



-   :broken_heart:{ .lg .middle } __Move on__

    ---

    Moving away from runnable is as simple as deleting relevant files.

    - Your application code remains as it is.


</div>

<hr style="border:2px dotted orange">

## Comparisons

--8<-- "examples/comparisons/README.md"
