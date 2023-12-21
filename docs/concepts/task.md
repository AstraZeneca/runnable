Task nodes are the execution units of the pipeline.

In magnus, a ```command``` in a task node can be [python functions](#python_functions),
[Jupyter notebooks](#notebook) or a [shell scripts](#shell).
All task nodes  can take arguments, retrieve and create files/objects and return
arguments, though their access patterns are different.


---

## Python functions

Python is the default ```command type``` of a task node. The ```command```
should be the dotted path to the python function.

!!! example "Dotted path"

    Assuming the below project structure:

    - The ```command``` for the ```outer_function``` should be ```outer_functions.outer_function```

    - The ```command``` for ```inner_function``` should be ```module_inner.inner_functions.inner_function```


        ```
        ..
        ├── outer_functions.py
        │   ├── outer_function()
        ├── module_inner
        │   ├── inner_functions.py
        │   |    ├── inner_function()
        ..

        ```

### Example

=== "python"

    ```python linenums="1" hl_lines="4-8"
    --8<-- "examples/concepts/simple.py"
    ```

=== "yaml"

    You can execute this by magnus execute -f examples/concepts/simple.yaml

    ```python linenums="1"
    --8<-- "examples/concepts/simple.yaml"
    ```


### Closer look

!!! tip inline end "Structuring"

    It is best to keep the application specific functions in a different module
    than the pipeline definition, if you are using Python SDK.

    In this example, we combined them as one module for convenience.


Lines 4-8 define the python function that we want to execute as part of the pipeline.
They are *plain old python functions*.

The rest of the python code (or yaml) defines and executes a pipeline that executes a task whose ```command```
is to execute this function.


### Accessing parameters

Please refer to [Initial Parameters](../parameters/#initial_parameters) for more information about setting
initial parameters.

Lets assume that the initial parameters are:

```yaml
--8<-- "examples/concepts/parameters.yaml"
```

- [x] Passing parameters between steps

=== "Natively"

    Internally, magnus stores the parameters in serialised json format.

    ### ^^Input arguments to the function^^

    Any arguments passed into the function should be at the root level of the json object.
    Arguments with type annotations will be casted appropriately. Arguments with no type casting will be sent in JSON format.

    For example, in line 13 and 28, arguments ```spam``` and ```eggs``` are at the root level in
    the yaml representation.

    ### ^^Output arguments of function^^

    Only pydantic models are allowed to be return types of a function. There is no need
    for any type annotation but is advised for a cleaner code.

    Output arguments are stored in JSON format by
    [model_dump](https://docs.pydantic.dev/latest/concepts/serialization/#modelmodel_dump),
    respecting the alias.

    The model structure of the pydantic model would be added to the root structure. This is
    useful when you want to add or modify parameters at the root level. For example, line 25
    would update the initial parameters.

    To update a subset of existing parameters at the root level, you can either create a new model or
    use [DynamicModel](https://docs.pydantic.dev/latest/concepts/models/#dynamic-model-creation).
    For example, lines 42-45 create a dynamic model to update the ```eggs``` parameter.


    !!! warning "caution"

         Returning "eggs" in line 42 would result in a new parameter "ham" at the root level
         as it looses the nested structure.



    ```python linenums="1"
    --8<-- "examples/concepts/task_native_parameters.py"
    ```


=== "Using the API"

    Magnus also has [python API](../../interactions) to access parameters.

    Use [get_parameter](../../interactions/get_parameter) to access a parameter at the root level.
    You can optionally specify the ```type``` by using ```cast_as``` argument to the API.
    For example, line 19 would cast ```eggs```parameter into ```EggsModel```.
    Native python types do not need any explicit ```cast_as``` argument.

    Use [set_parameter](../../interactions/set_parameter) to set parameters at the root level.
    Multiple parameters can be set at the same time, for example, line 26 would set both the ```spam```
    and ```eggs``` in a single call.

    The pydantic models would be serialised to JSON format using
    [model_dump](https://docs.pydantic.dev/latest/concepts/serialization/#modelmodel_dump), respecting the alias.


    ```python linenums="1"
    --8<-- "examples/concepts/task_api_parameters.py"
    ```

=== "Using environment variables"

    Any environment variable with ```MAGNUS_PRM_``` is understood to be a parameter in magnus.

    Before the execution of the ```command```, all the parameters at the root level are set as environment variables
    with the key prefixed by ```MAGNUS_PRM_```. Python functions that are called during the execution of the command
    can access them as environment variables.

    After the execution of the ```command```, the environment is "scanned" again to identify changes to the existing
    variables prefixed by ```MAGNUS_PRM_```. All updated variables are stored at the root level.

    Parameters set by environment variables over-ride the parameters defined by the initial parameters which can be
    handy to quickly experiment without modifying code or to dynamically adjust behavior when running in
    orchestrators like Argo or AWS step functions.

    ```python linenums="1"
    --8<-- "examples/concepts/task_env_parameters.py"
    ```



!!! abstract "Verbose?"

        We acknowledge that using pydantic models as our
        [Data transfer objects](https://stackoverflow.com/questions/1051182/what-is-a-data-transfer-object-dto) is verbose in comparison to using
        ```dict```.

        The advantages of using strongly typed DTO has long term advantages of implicit validation, typing hints
        in editors. This choice is inspired from [FastAPI's](https://fastapi.tiangolo.com/features/#pydantic-features)
        ways of working.


### Data flow

Catalog configuration.
Over-riding the catalog configuration.

=== "Using the API"


=== "Via configuration"


### Object flow

Using the API


### Execution logs

Show the tree structure in the catalog

---

## Notebook

### Example

### Accessing parameters


- [x] Initial parameters


=== "Natively"

=== "Using the API"

=== "Using environment variables"

- [x] Returning parameters

=== "Using the API"

=== "Using environment variables"


- [x] Passing parameters between steps

=== "Using the API"

=== "Using environment variables"


### Data flow

Catalog configuration.
Over-riding the catalog configuration.

=== "Using the API"


=== "Via configuration"


### Object flow

Using the API


### Execution logs

Show the tree structure in the catalog


---

## Shell


### Example

### Accessing parameters


- [x] Initial parameters

=== "Using environment variables"

- [x] Returning parameters

=== "Using environment variables"


- [x] Passing parameters between steps


=== "Using environment variables"


### Data flow

Catalog configuration.
Over-riding the catalog configuration.

=== "Via configuration"


### Execution logs

Show the tree structure in the catalog
