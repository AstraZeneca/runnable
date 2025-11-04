```parameters``` are data that can be passed from one ```task``` to another.

## Concept

For example, in the below snippet, the parameters ```x``` and ```y``` are passed from
```generate``` to ```consume```.

```python linenums="1"
x, y = generate() # returns x and y as output
consume(x, y) # consumes x, y as input arguments.
```

The data types of ```x``` and ```y``` can be:

- JSON serializable: int, string, float, list, dict including pydantic models.
- Objects: Any [dill](https://dill.readthedocs.io/en/latest/) friendly objects.


## Compatibility

Below table summarizes the input/output types of different task types.
For ex: notebooks can only take JSON serializable parameters as input
but can return json/pydantic/objects.

| Task     |  Input                    | Output                   |
| -------- | :---------------------:  | :----------------------: |
| python   | json, pydantic, object via function arguments  | json, pydantic, object as ```returns```  |
| notebook | json via cell tagged with ```parameters``` | json, pydantic, object  as ```returns``` |
| shell    | json via environment variables | json environmental variables as ```returns``` |



## Project parameters

Project parameters are values that can be provided to your pipeline from external sources. Runnable supports two main approaches for providing project parameters:

1. **YAML files** - Define parameters in structured configuration files
2. **Environment variables** - Set parameters using environment variables with the `RUNNABLE_PRM_` prefix

!!! info "Parameter Priority"

    Environment variables take precedence over YAML-based parameters. If the same parameter is defined in both a YAML file and as an environment variable, the environment variable value will be used.

### YAML-based parameters

Parameters can be defined in YAML files and passed to the pipeline execution:

```yaml title="parameters.yaml"
integer: 1
floater: 3.14
stringer: hello
pydantic_param:
  x: 10
  foo: bar
chunks: [1, 2, 3]
```

The YAML file is then used during pipeline execution:

```python
pipeline.execute(parameters_file="parameters.yaml")
```

### Environment variable parameters

Parameters can also be defined as environment variables prefixed by ```RUNNABLE_PRM_```:

```shell
export RUNNABLE_PRM_integer="1"
export RUNNABLE_PRM_floater="3.14"
export RUNNABLE_PRM_stringer="hello"
export RUNNABLE_PRM_pydantic_param="{'x': 10, 'foo': 'bar'}"
export RUNNABLE_PRM_chunks="[1, 2, 3]"
```

Environment variables are useful for:

- Quick experimentation without changing code
- Different configurations across environments (dev/staging/prod)
- Overriding specific parameters in YAML files

!!! warning "Type casting"

    Annotating the arguments of python function ensures the right data type of arguments.

    It is advised to ```cast``` the parameters in notebook tasks or shell.


### Parameter override example

The following example demonstrates how environment variables override YAML parameters:

```python linenums="1"
--8<-- "examples/03-parameters/static_parameters_python.py"
```

In this example:
- The YAML file (`examples/common/initial_parameters.yaml`) defines base parameter values
- The environment variable `RUNNABLE_PRM_envvar` is set to `"from env"`
- When the pipeline executes, it uses values from the YAML file for most parameters, but the `envvar` parameter comes from the environment variable, demonstrating the override behavior

### Accessing parameters

=== "python"

    The functions have arguments that correspond to the project parameters.

    Without annotations for nested params, they are sent in as dictionary.

=== "notebook & shell"

    The notebook has cell tagged with ```parameters``` which are substituted at run time.

    The shell script has access to them as environmental variables.

    Both notebook and shell tasks receive parameters from the same sources (YAML files and environment variables) with the same priority system.

    ```python linenums="1"
    --8<-- "examples/03-parameters/static_parameters_non_python.py"
    ```



## Access & returns

### access

The access of parameters returned by upstream tasks is similar to project parameters (see above section)


### returns

Tasks can return parameters which can then be accessed by downstream tasks.

The syntax is inspired by:

```python linenums="1"
def generate():
    ...
    return x, y

def consume(x, y):
    ...

x, y = generate() # returns x and y as output
consume(x, y) # consumes x, y as input arguments.
```

and implemented in ```runnable``` as:

=== "Python SDK"

    ```python linenums="1"
    from runnable import PythonTask
    # The returns syntax can be used for notebook and shell scripts too.
    generate_task = PythonTask(function="generate", returns=["x", "y"])
    consume_task = PythonTask(function="consume")

    ```

!!! warning "order of returns"

    The order of ```returns``` should match the order of the python function returning them.


### marking returns as ```metric``` or ```object```

JSON style parameters can be marked as a ```metric``` in
[python functions](task.md/#python-functions), [notebook](task.md/#notebook), [shell](task.md/#shell). Metric parameters can be accessed as normal parameters in downstream steps.

Returns marked as ```pickled``` in [python functions](task.md/#python-functions), [notebook](task.md/#notebook) are serialized using ```dill```.

### Example

```python linenums="1"
import pandas as pd

# Assuming a function return a pandas dataframe and a score
def generate():
    ...
    return df, score

# Downstream step consuming the df and score
def consume(df: pd.Dataframe, score: float):
    ...
```

=== "Python SDK"

    ```python linenums="1"
    from runnable import metric, pickled, PythonTask

    generate_task = PythonTask(function="generate",
                        returns=[pickled("df"),  # pickle df
                                metric("score")]) # mark score as metric

    consume_task = PythonTask(function="consume")

    ```



## Complete Example

=== "python"

    === "python"

        ```python linenums="1" hl_lines="26-37"
        --8<-- "examples/03-parameters/passing_parameters_python.py"
        ```


=== "notebook"

    To access parameters, the cell should be tagged with ```parameters```. Only
    JSON style parameters can be injected in.

    Any python variable defined during the execution of the notebook matching the
    name in ```returns``` is inferred as a parameter. The variable can be either
    JSON type or objects.

    === "python"

        ```python linenums="1" hl_lines="21-32"
        --8<-- "examples/03-parameters/passing_parameters_notebook.py"
        ```


=== "shell"

    Shell tasks can only access/return JSON style parameters

    === "python"

        ```python linenums="1" hl_lines="28-38"
        --8<-- "examples/03-parameters/passing_parameters_shell.py"
        ```
