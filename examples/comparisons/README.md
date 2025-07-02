To simplify the core of ```runnable```,  consider the following function:

```python
def func(x: int, y: pd.DataFrame):
    # Access some data, input.csv
    # do something with the inputs.
    # Write a file called output.csv for downstream steps.
    # return an output.
    return z
```

It takes

- *inputs* x (integer) and y (a pandas dataframe or any other object),
- processes input data, input.csv expected on local file system
- writes a file, output.csv to local filesystem
- returns z (a simple datatype or object)

The function in wrapped in runnable as:

```python
from somewhere import func
from runnable import PythonTask, pickled, Catalog

# instruction to get input.csv from catalog at the start of the step.
# and move output.csv to the catalog at the end of the step
catalog = Catalog(get=["input.csv"], put=["output.csv"])

# Call the function, func and expect it to return "z" while moving the files
# It is expected that "x" and "y" are parameters set by some upstream step.
# If the return parameter is an object, use pickled("z")
func_task = PythonTask(name="function", function=func, returns=["z"], catalog=catalog)
```

Briefly:

1. The function remains the same as written
2. The required data sets are put in place for the function execution
3. The required input parameters are inspected and passed in from the available parameters
4. After the function call, the return parameters are added to the parameter space
5. The processed data is stored for future use.

Below are the implementations in alternative frameworks. Note that
the below are the best of our understanding of the frameworks, please let us
know if there are better implementations.


Along with the observations,

- We have implemented [MNIST example in pytorch](https://github.com/pytorch/examples/blob/main/mnist/main.py)
in multiple frameworks for easier practical comparison.
- The tutorials are inspired from tutorials of popular frameworks to give a flavor of ```runnable```.

<hr style="border:2px dotted orange">

### metaflow

The function in metaflow's step would roughly be:

```python
from metaflow import step, conda, FlowSpec

class Flow(FlowSpec)

    @conda(libraries={...})
    @step
    def func_step(self):
        from somewhere import func
        self.z = func(self.x, self.y)

        # Use metaflow.S3 to move files
        # Move to next step.
        ...
```

Though the philosophy is similar, there are some implementation differences in:

1. Dependency management - metaflow requiring decorators while runnable works in the project environment
2. Dataflow - runnable moves data in and out via the configuration while in metaflow the user is expected to write code.
3. Support for notebooks - runnable allows notebooks to be steps.
4. Platform vs package - runnable is a package while metaflow takes a platform perspective



<hr style="border:2px dotted orange">

### kedro

The function in ```kedro``` implementation would roughly be:

Note that any movement of files should happen via data catalog.

```python
from kedro.pipeline import Pipeline, node, pipeline
from somewhere import func

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=func,
                inputs=["params:x", "y"],
                outputs=["z"],
                name="my_function",
            ),
            ...
        ]
    )

```



```kedro``` has a larger footprint in the domain code by the configuration files. It imposes
a structure and code organization while runnable does not have an opinion on the code structure.

```runnable``` supports notebooks, dynamic pipelines while kedro lacks support for these.
