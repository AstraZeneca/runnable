[tasks](task.md) might also need to pass ```files``` between them.

## Concept

For example:

```python linenums="1"

def generate():
    with open("data.csv", "w"):
        # write content
        ...

def consume():
    with open("data.csv", "r"):
        # read content
        ...

generate()
consume()

```


## Syntax

The same can be represented in ```runnable``` as [catalog](../reference.md/#catalog).

For example, the above snippet would be:

=== "Python SDK"

    ```python linenums="1"

    from runnable import PythonTask, Pipeline, Catalog

    write_catalog = Catalog(put=["data.csv"])
    read_catalog = Catalog(get=["read.csv"])

    generate_task = PythonTask(name="generate", function=generate, catalog=write_catalog)
    consume_task = PythonTask(name="consume", function=consume, catalog=read_catalog)

    pipeline = Pipeline(steps=[generate_task, consume_task])
    pipeline.execute()
    ```


## Example

=== "Python SDK"

    ```python linenums="1"
    --8<-- "examples/04-catalog/catalog.py"
    ```
