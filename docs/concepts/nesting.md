As seen from the definitions of [parallel](../parallel) or [map](../map), the branches are pipelines
themselves. This allows for deeply nested workflows in **magnus**.

Technically there is no limit in the depth of nesting but there are some practical considerations.


- Not all workflow engines that magnus can transpile the workflow to support deeply nested workflows.
AWS Step functions and Argo workflows support them.

- Deeply nested workflows are complex to understand and debug during errors.


## Example


```python
--8<-- "examples/concepts/nesting.py"
```
