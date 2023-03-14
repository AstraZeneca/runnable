# Local

Local mode is an interactive mode. In this mode, magnus does the traversal of the graph and execution of nodes
on the local compute from which it is executed.

In this set up, we ignore max run time set on the dag completely.

All types of secrets, catalog and run log store are compatible with this mode. And this compute mode is default if
no mode if provided in the dag definition.

## Configuration

The full configuration of local mode is:

```yaml
mode:
  type: local
  config:
    enable_parallel:
```

### Enabling parallel

By default, none of the branches in parallel or a map node are executed in parallel.
You can enable it by setting enable_parallel to True.
