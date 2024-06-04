In this section, we take the familiar MNIST problem and implement it in different orchestration frameworks.

The [original source code](https://github.com/pytorch/examples/blob/main/mnist/main.py) is shown in [source.py](source.py)

The individual directories are orchestration specific implementations.

## Notes

For the purpose of comparisons, consider the following function:

```python
def func(x: int, y:pd.DataFrame):
    # do something with the inputs.
    # Write a file called data.csv for downstream steps.
    # return an output.
    return z
```

It takes *inputs* x (integer) and y (a pandas dataframe or any other object), does some processing and writes a file to local disk and returns z (a simple datatype or object)

The function in wrapped in runnable as:

```python
from somewhere import func
from runnable import PythonTask, pickled, Catalog

# instruction to move the file data.csv from local disk to a blob store
catalog = Catalog(put=["data.csv"])
# Call the function, func and expect it to return "z" while moving the files
# It is expected that "x" and "y" are parameters set by some upstream step.
# If the return parameter is an object, use pickled("z")
func_task = PythonTask(name="function", function=func, returns=["z"], catalog=catalog)
```


### metaflow

The function in metaflow's step would rougly be:

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

The differences between both:

- dependency management in ```runnable``` is expected to be user driven and "pythonic" (venv, poetry etc) while metaflow packages provides a per-step (or flow level) libraries.
- data flow between steps is by an instruction in runnable to ```glob``` files in
local disk and present them in the same structure to downstream steps. ```metaflow``` needs this to be via code.
- runnable allows ```notebooks``` to be a task, allowing simple data types to be
parameters while collecting objects from the notebook execution.
- metaflow has a *platform* side to it and comes up with prescribed infrastructure while runnable is, in many ways, a transpiler to your chosen infrastructure.
