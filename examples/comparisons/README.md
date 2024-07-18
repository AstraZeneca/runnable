For the purpose of comparisons, consider the following function:

```python
def func(x: int, y:pd.DataFrame):
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

- The API between ```runnable``` and ```metaflow``` are comparable.
- There is a mechanism for functions to accept/return parameters.
- Both support parallel branches, arbitrary nesting of pipelines.

The differences:

##### dependency management:

```runnable``` depends on the activated virtualenv for dependencies which is natural to python.
Use custom docker images to provide the same environment in cloud based executions.

```metaflow``` uses decorators (conda, pypi) to specify dependencies. This has an advantage
of abstraction from docker ecosystem for the user.

##### dataflow:

In ```runnable```, data flow between steps is by an instruction in runnable to ```glob``` files in
local disk and present them in the same structure to downstream steps.

```metaflow``` needs a code based instruction to do so.

##### notebooks:

```runnable``` allows notebook as tasks. Notebooks can take JSON style inputs and can return
pythonic objects for downstream steps.

```metaflow``` does not support notebooks as tasks.

##### infrastructure:

```runnable```, in many ways, is just a transpiler to your chosen infrastructure.

```metaflow``` is a platform with its own specified infrastructure.

##### modular pipelines

In ```runnable``` the individual pipelines of parallel and map states are
pipelines themselves and can run in isolation. This is not true in ```metaflow```.

##### unit testing pipelines

```runnable``` pipelines are testable using ```mocked``` executor where the executables can be mocked/patched.
In ```metaflow```, it depends on how the python function is wrapped in the pipeline.

##### distributed training

```metaflow``` supports distributed training.

As of now, ```runnable``` does not support distributed training but is in the works.


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

##### Footprint

```kedro``` has a larger footprint in the domain code by the configuration files. It is tightly structured and
provides a CLI to get started.

To use ```runnable``` as part of the project requires
adding a pipeline definition file (in python or yaml) and an optional configuration file.

##### dataflow

Kedro needs the data flowing through the pipeline via ```catalog.yaml``` which
provides a central place to understand the data.

In ```runnable```, the data is presented to the individual tasks as
requested by the ```catalog``` instruction.

##### notebooks

Kedro supports notebooks for exploration but not as tasks of the pipeline.

##### dynamic pipelines

```kedro``` does not support dynamic pipelines or map state.

##### distributed training

```kedro``` supports distributed training via a [plugin](https://github.com/getindata/kedro-azureml).

As of now, ```runnable``` does not support distributed training but is in the works.
