Magnus provides a rich definition of of step types.

<div class="annotate" markdown>

- [stub](/concepts/stub): A mock step which is handy during designing and debugging pipelines.
- [task](/concepts/task): To execute python functions, jupyter notebooks, shell scripts.
- [parallel](/concepts/parallel): To execute many tasks in parallel.
- [map](/concepts/map): To execute the same task over a list of parameters. (1)

</div>

1. Similar to ```map``` state in AWS step functions or ```loops``` in Argo workflows.


## stub

Used as a mock node or a placeholder before the actual implementation (1).
{ .annotate }

1.  :raised_hand: Equivalent to ```pass``` or ```...``` in python.


=== "yaml"

    ``` yaml
    --8<-- "examples/mocking.yaml"
    ```

=== "python"

    ```python
    --8<-- "examples/mocking.py"
    ```

    1. The name of the node can be as descriptive as you want. Only ```.``` or ```%``` are not allowed.
    2. Stub nodes can take arbitrary parameters; useful to temporarily mock a node. You can define the dependency on step1 using ```depends_on```
    3. ```terminate_with_success``` indicates that the dag is completed successfully. You can also use ```terminate_with_failure``` to indicate the dag failed.
    4. Add ```success``` and ```fail``` nodes to the dag.


## task

Used to execute a single unit of work. You can use [python](/concepts/task/#python_functions),
[shell](/concepts/task/#shell), [notebook](/concepts/task/#notebook) as command types.

!!! note annotate "Execution logs"

    You can view the execution logs of the tasks in the [catalog](/concepts/catalog) without digging through the
    logs from the underlying executor.


=== "Example functions"

    The below content is assumed to be ```examples/functions.py```

    ```python
    --8<-- "examples/functions.py"
    ```

=== "yaml"

    ``` yaml
    --8<-- "examples/python-tasks.yaml"
    ```

    1. Note that the ```command``` is the [path to the python function](/concepts/task/#python_functions).
    2. ```python``` is default command type, you can use ```shell```, ```notebook``` too.

=== "python"

    ```python
    --8<-- "examples/python-tasks.py"
    ```

    1. Note that the command is the [path to the function](/concepts/task/#python_functions).
    2. There are many ways to define dependencies within nodes, step1 >> step2, step1 << step2 or during the definition of step1, we can define a next step.
    3. ```terminate_with_success``` indicates that the dag is completed successfully. You can also use ```terminate_with_failure``` to indicate the dag failed.
    4. Add ```success``` and ```fail``` nodes to the dag.
