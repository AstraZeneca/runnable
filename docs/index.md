:runner: Orchestrate python functions, notebooks or scripts on your local machine by just adding
*one file*.

:runner: + :cloud: Move to any cloud by adding *one more file*.

## functions

The below content is assumed to be ```examples/functions.py```

!!! note inline end "pydantic models"

    The functions should use pydantic models as their input and outputs.

    Pydantic models offer better representations of the input and output, inspired by
    [FastAPI's implementation](https://fastapi.tiangolo.com/features/#pydantic-features).

```python linenums="1"
--8<-- "examples/functions.py"
```


There is nothing special about the functions, they are *plain old python functions*.


## local :runner:


Replace the "driver" function with a *runnable* definition in either ```python sdk```
or ```yaml```.

!!! note inline end "steps"

    The steps are essentially a representation of the "driver" function.

    The gains by this definition for local executions are clearer by the metadata gathered
    during the exeuction.

=== "yaml"

    ``` yaml linenums="1"
    --8<-- "examples/python-tasks.yaml"
    ```

    1. Start the pipeline execution at step1
    2. The name of the step.
    3. The path to the python function
    4. Go to step2, if successful
    5. Go to success node, if successful
    6. Mark the execution as success



=== "python"

    ```python linenums="1"
    --8<-- "examples/python-tasks.py"
    ```

    1. The name of the step.
    2. The path to the python function
    3. ```terminate_with_success``` indicates that the pipeline is completed successfully. You can also use ```terminate_with_failure``` to indicate the pipeline fail.
    4. There are many ways to define dependencies within nodes, step1 >> step2, step1 << step2 or using depends_on.
    5. Start the pipeline execution at step1
    6. The list of steps to be executed, the order does not matter.
    7. Add ```success``` and ```fail``` nodes to the pipeline.
    8. Returns the metadata captured during the execution.

=== "metadata"

    Add the run log here

:sparkles: Thats it!! :sparkles:

By adding *one file* you created a pipeline. Your application code
did not change at all.

There is no boilerplate code, no adherence to structure, no intrusion into the
application code.

## cloud :runner:
