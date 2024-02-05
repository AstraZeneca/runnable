All the steps of the pipeline are executed in the local compute environment in the same shell
as it was triggered.

- [x] Provides the most comfortable environment for experimentation and development.
- [ ] The scalability is constrained by the local compute environment.
- [ ] Not possible to provide specialized compute environments for different steps of the pipeline.


!!! warning inline end "parallel executions"

    Run logs that use a single json (eg. file-system) are not compatible with parallel
    executions due to race conditions to write the same file by different processes.

    Use ```chunked``` run log stores (eg. chunked-fs).



## Configuration

```yaml
executor: local
config:
  enable_parallel: false # (1)
```

1. By default, all tasks are sequentially executed. Provide ```true``` to enable tasks within
[parallel](../../concepts/parallel) or [map](../../concepts/map) to be executed in parallel.



All the examples in the concepts section are executed using ```local``` executors.
