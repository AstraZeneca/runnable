# Demo Renderer

In this compute mode, we translate the dag into a bash script to demonstrate the idea of dag translation. Composite
nodes like ```parallel```, ```dag``` and ```map``` are not allowed as part of the definition.

In this set up, we ignore max run time set on the dag completely.

## Configuration


The full configuration of the executor is:

```yaml
executor:
  type: demo-renderer
```

The parameters that have to be passed could be done either via environment variables prefixed by ```MAGNUS_PRM_```
or by the command line like the [example shown here](../../../getting_started/example-deployment/#execution).
