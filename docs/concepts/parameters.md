In magnus, ```parameters``` are python data types that can be passed from one ```task```
to the next ```task```. These parameters can be accessed by the ```task``` either as
environment variables, arguments of the ```python function``` or using the
[API](../../interactions).

## Initial parameters

The initial parameters of the pipeline can set by using a ```yaml``` file and presented
during execution

```--parameters-file, -parameters``` while using the [magnus CLI](../../usage/#usage)

or by using ```parameters_file``` with [the sdk](../../sdk/#magnus.Pipeline.execute).

They can also be set using environment variables which override the parameters defined by the file.

=== "yaml"

    Deeply nested yaml objects are supported.

    ```yaml
    --8<-- "examples/concepts/parameters.yaml"
    ```


=== "environment variables"

    Any environment variables prefixed with ```MAGNUS_PRM_ ``` are interpreted as
    parameters by the ```tasks```.

    The yaml formatted parameters can also be defined as:

    ```shell
    export MAGNUS_PRM_spam="hello"
    export MAGNUS_PRM_eggs='{"ham": "Yes, please!!"}'
    ```

    Parameters defined by environment variables override parameters defined by
    ```yaml```. This can be useful to do a quick experimentation without changing code.


## Parameters flow

Tasks can access and return parameters and the patterns are specific to the
```command_type``` of the task nodes. Please refer to [tasks](../task)
for more information.
