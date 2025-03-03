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
