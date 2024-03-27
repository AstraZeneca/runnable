# Runnable

<figure markdown>
  ![Image title](assets/sport.png){ width="200" height="100"}
  <figcaption>Orchestrate your functions, notebooks, scripts anywhere!!</figcaption>
</figure>

<a href="https://www.flaticon.com/free-icons/runner" title="runner icons">Runner icons created by Leremy - Flaticon</a>

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

<div class="annotate" markdown>

```diff
-def main():
+def runnable_pipeline():

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



+# The sequence of tasks
+load_data_task >> model_fit_task >> generate_plots_task (3)
+pipeline = Pipeline(
+   steps=[load_data_task, model_fit_task, generate_plots_task],
+   start_at=load_data_task,+)

+pipeline.execute()

+return pipeline

```
</div>

1. Return objects X and Y.
2. Store the file `iris_logistic.png` for future reference.
3. Define the sequence of tasks.


!!! tip inline end "Notebooks and Shell scripts"

    You can execute notebooks and shell scripts too!!

    They can be written just as you would want them, *plain old notebooks and scripts*.


- [x] Absolutely no change in data science code to make it `runnable`
- [x] The ```driver``` function has an equivalent and intuitive runnable expression
- [x] Everything within the pipeline execution is stored for future reference
- [x] The pipeline is `runnable` in any environment.


## But why runnable?

There are abundant orchestration tools in python, a well maintained and curated [list is
available here](https://github.com/EthicalML/awesome-production-machine-learning/).
