Without any orchestrator, the simplest pipeline could be the below functions:


```python linenums="1"
def generate():
    ...
    # write some files, data.csv
    ...
    # return objects or simple python data types.
    return x, y

def consume(x, y):
    ...
    # read from data.csv
    # do some computation with x and y


# Stich the functions together
# This is the driver pattern.
x, y = generate()
consume(x, y)
```

## Runnable representation

The workflow in ```runnable``` would be:

```python linenums="1"
from runnable import PythonTask, pickled, catalog, Pipeline

generate_task = PythonTask(name="generate", function=generate,
                returns=[pickled("x"), y],
                catalog=Catalog(put=["data.csv"])

consume_task = PythonTask(name="consume", function=consume,
                catalog=Catalog(get=["data.csv"])

pipeline = Pipeline(steps=[generate_task, consume_task])
pipeline.execute()

```




- ```runnable``` wraps the functions ```generate``` and ```consume``` as [tasks](task.md).
- Tasks can [access and return](parameters.md/#access_returns) parameters.
- Tasks can also share files between them using [catalog](catalog.md).
- Tasks are stitched together as [pipeline](pipeline.md)
- The execution environment is configured via # todo


## Examples

All the concepts are accompanied by [examples](https://github.com/AstraZeneca/runnable/tree/main/examples).
