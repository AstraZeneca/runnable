Stub nodes in runnable are just like
[```Pass``` state](https://docs.aws.amazon.com/step-functions/latest/dg/amazon-states-language-pass-state.html)
in AWS Step Functions or ```pass``` in python code. It is a placeholder and useful when you want to debug or
design your pipeline.

Stub nodes can take arbitrary number of parameters and is always a success.

## Example

!!! note annotate inline end "Intuition"

    Designing a pipeline is similar to writing a modular program. Stub nodes are handy to create a placeholder
    for some step that will be implemented in the future.

    During debugging, changing a node to ```stub``` will let you focus on the actual bug without having to
    execute the additional steps.


=== "yaml"

    In the below example, all the steps are ```stub``` nodes. The only required field is
    the ```next``` which is needed for graph traversal. As seen in ```step 2``` definition,
    they can have arbitrary fields.


    ``` yaml hl_lines="20-24"
    --8<-- "examples/mocking.yaml"
    ```

=== "python"

    In the below example, all the steps are ```stub``` nodes.

    ``` python hl_lines="21-24"
    --8<-- "examples/mocking.py"
    ```

The only required field is the ```name```, ```next``` which is needed for graph traversal.

- yaml definition needs ```next``` to be defined as part of the step definition.
- python SDK can define the ```next``` when linking the nodes as part of the pipeline.
