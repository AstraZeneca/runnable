## TODO: Concretly show an example!

In runnable, ```parameters``` are python data types that can be passed from one ```task```
to the next ```task```. These parameters can be accessed by the ```task``` either as
environment variables, arguments of the ```python function``` or using the
[API](../interactions.md).

## Initial parameters

The initial parameters of the pipeline can set by using a ```yaml``` file and presented
during execution

```--parameters-file, -parameters``` while using the [runnable CLI](../usage.md/#usage)

or by using ```parameters_file``` with [the sdk](..//sdk.md/#runnable.Pipeline.execute).

They can also be set using environment variables which override the parameters defined by the file.

=== "yaml"

    Deeply nested yaml objects are supported.

    ```yaml
    --8<-- "examples/concepts/parameters.yaml"
    ```


=== "environment variables"

    Any environment variables prefixed with ```runnable_PRM_ ``` are interpreted as
    parameters by the ```tasks```.

    The yaml formatted parameters can also be defined as:

    ```shell
    export runnable_PRM_spam="hello"
    export runnable_PRM_eggs='{"ham": "Yes, please!!"}'
    ```

    Parameters defined by environment variables override parameters defined by
    ```yaml```. This can be useful to do a quick experimentation without changing code.


## Parameters flow

Tasks can access and return parameters and the patterns are specific to the
```command_type``` of the task nodes. Please refer to [tasks](../concepts/task.md)
for more information.
