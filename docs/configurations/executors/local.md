All the steps of the pipeline are executed in the local compute environment in the same shell
as it was triggered.

- [x] Provides the most comfortable environment for experimentation and development.
- [ ] The scalability is constrained by the local compute environment.
- [ ] Not possible to provide specialized compute environments for different steps of the pipeline.
- [ ] All the steps within ```parallel``` or ```map``` nodes are executed sequentially.



## Configuration

```yaml
executor: local
```

All the examples in the concepts section are executed using ```local``` executors.
