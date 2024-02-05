**Magnus** is designed to make effective collaborations between data scientists/researchers
and infrastructure engineers.

All the features described in the [concepts](../../concepts/the-big-picture) are
aimed at the *research* side of data science projects while configurations add *scaling* features to them.


Configurations are presented during the execution:

For ```yaml``` based pipeline, use the ```--config-file, -c``` option in the [magnus CLI](../../usage/#usage).

For [python SDK](../../sdk/#magnus.Pipeline.execute), use the ```configuration_file``` option or via
environment variable ```MAGNUS_CONFIGURATION_FILE```

## Default configuration

```yaml
--8<-- "examples/configs/default.yaml"
```

1. Execute the pipeline in the local compute environment.
2. The run log is not persisted but present in-memory and flushed at the end of execution.
3. No catalog functionality, all catalog operations are effectively no-op.
4. No secrets functionality, all secrets are effectively no-op.
5. No experiment tracking tools, all interactions with experiment tracking tools are effectively no-op.
Run log still captures the metrics, but are not passed to the experiment tracking tools.

The default configuration for all the pipeline executions runs on the [local compute](../executors/local), using a
```buffered``` run log store with no catalog or secrets or experiment tracking functionality.



## Format

The configuration file is in yaml format and the typical structure is:

```yaml
service:
  type: service provider
  config:
    ...
```

where service is one of ```executor```, ```catalog```, ```experiment_tracker```,
 ```secrets``` or ```run_log_store```.
