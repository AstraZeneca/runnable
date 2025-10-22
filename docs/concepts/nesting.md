As seen from the definitions of [parallel](../concepts/parallel.md) or
[map](../concepts/map.md), the branches are pipelines
themselves. This allows for deeply nested workflows in **runnable**.

Technically there is no limit in the depth of nesting but there are some practical considerations.


- Not all workflow engines that runnable can transpile the workflow to support deeply nested workflows.
AWS Step functions and Argo workflows support them.

- Deeply nested workflows are complex to understand and debug during errors.


## Example


=== "Python SDK"


    You can run this pipeline by ```python examples/06-parallel/nesting.py```

    ```python linenums="1"
    --8<-- "examples/06-parallel/nesting.py"
    ```

=== "YAML (Legacy)"

    You can run this pipeline by ```runnable execute examples/parallel/nesting.yaml```

    ```yaml linenums="1"
    --8<-- "examples/06-parallel/nesting.yaml"
    ```
