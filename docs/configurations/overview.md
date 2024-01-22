Configurations are the way to choose between different providers of below services.

- [executor](../../concepts/executor)
- [catalog](../../concepts/catalog)
- [experiment tracker](../../concepts/experiment-tracking)
- [secrets](../../concepts/secrets)
- [run log store](../../concepts/run-log)


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
