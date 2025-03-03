
Execute all the steps of the pipeline in containers.

- [x] Provides a way to test the containers and the execution of the pipeline in local environment.
- [x] Any failure in cloud native container environments can be replicated in local environments.
- [x] Ability to provide specialized compute environments for different steps of the pipeline.
- [ ] The scalability is still constrained by the resources in local environment.


## Configuration

::: extensions.pipeline_executor.local_container.LocalContainerExecutor
    options:
        show_root_heading: false
        show_bases: false
        members: false
        show_docstring_description: true
        heading_level: 3


!!! tip "Debugging"

    ```auto_remove_container``` allows you to run the failed container independently to
    identify the issue that caused the failure.


All the examples in the concepts section can be executed using the below configuration:

```yaml
pipeline-executor:
  type: local-container
  config:
    docker_image: <your docker image>
```

## Dynamic docker image

The docker image can provided at the run time by using environmental variables.

For example:

```yaml
pipeline-executor:
  type: local-container
  config:
    docker_image: $docker_image
```

The ```$docker_image``` will be replaced by the environmental variable
```RUNNABLE_VAR_docker_image``` during run time. The same rule applies to overrides too.
