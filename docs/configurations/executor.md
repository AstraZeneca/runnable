## Local

All the steps of the pipeline are executed in the local compute environment in the same shell
as it was triggered.

- [x] Provides the most comfortable environment for experimentation and development.
- [ ] The scalability is constrained by the local compute environment.
- [ ] Not possible to provide specialized compute environments for different steps of the pipeline.

!!! warning inline end "parallel executions"

    Not all run log stores are compatible for parallel execution. Please choose from
    run log stores that are compatible for parallel execution.


### Options

```yaml
executor: local
config:
  enable_parallel: false # (1)
```

1. By default, all tasks are sequentially executed. Provide ```true``` to enable tasks within
[parallel](../../concepts/parallel) or [map](../../concepts/map) to be executed in parallel.



## Local container

Execute all the steps of the pipeline in containers. Please refer to the
[note on containers](#container_environments) on building images.

- [x] Provides a way to test the containers and the execution of the pipeline in local environment.
- [x] Any failure in cloud native container environments can be replicated in local environments.
- [x] Ability to provide specialized compute environments for different steps of the pipeline.
- [ ] The scalability is still constrained by the resources in local environment.

### options

```yaml
executor: local-container
config:
  docker_image: <required>
  enable_parallel: false # (1)
  auto_remove_container: true # (2)
  environment:
    ...
```

 1. By default, all tasks are sequentially executed. Provide ```true``` to enable tasks within
[parallel](../../concepts/parallel) or [map](../../concepts/map) to be executed in parallel.
2. Set it to false, to debug a failed container.

The ```docker_image``` field is required and default image to execute tasks
of the pipeline. Individual [tasks](../../concepts/task) can over-ride the
default image by providing an ```executor_config``` as part of step definition.

#### Example

=== "Local container config"

    ```yaml linenums="1" hl_lines="4"
    --8<-- "examples/configs/local-container.yaml"
    ```

    1. Use local-container executor type to execute the pipeline.
    2. By default, all the tasks are executed in the docker image . Please
    refer to [building docker images](#container_environments)
    3. Pass any environment variables that are needed for the container.
    4. Store the run logs in the file-system. Magnus will handle the access to them
    by mounting the file system into the container.


=== "python sdk"

    ```python linenums="1"
    --8<-- "examples/configs/simple.py"
    ```


=== "yaml"

    ```yaml linenums="1"
    --8<-- "examples/configs/simple.yaml"
    ```

=== "Run log"


=== "Step override"

    In the below pipeline definition, the step ```shell``` executes in the default docker image
    while the step ```custom docker image``` executes in the docker image in a different image.

    ```yaml linenums="1" hl_lines="21-23"
    --8<-- "examples/configs/step-overrides-container.yaml"
    ```

    1. executor_config should be specified per executor and exposes all the configuration
    variables from the global config apart from ```enable_parallel```.




## Argo workflows



## Container environments

### Pipeline definition

Executing pipelines in containers needs a ```yaml``` based definition of the pipeline which is referred during the
[task execution](../../concepts/executor/#step_execution).


Any execution of the pipeline [defined by SDK](../../sdk) generates the pipeline definition in ```yaml``` format
for all executors apart from the ```local``` executor. Follow the below steps to create the docker image.


<div class="annotate" markdown>

1. Optionally (but highly recommended) version your code using git.
2. Build the docker image with the ```yaml``` file-based definition as part of the image. We recommend
tagging the docker image with the short git sha to uniquely identify the docker image (1).
3. Define a [variable to temporarily hold](#dynamic_name_of_the_image) the docker image name in the
pipeline definition, if the docker image name is not known.
4. Execute the pipeline using the [magnus CLI](../../cli).

</div>

1. Avoid using generic tags such as [```latest```](https://docs.docker.com/develop/dev-best-practices/).

### Dynamic name of the image


All containerized executors have a circular dependency problem.

- The docker image tag is only known after the creation of the image with the ```yaml``` file-based definition.
- But the ```yaml``` file-based definition needs the docker image tag as part of the definition.



!!! warning inline end

    Not providing the required environment variable will raise an exception.

To resolve this, magnus supports ```variables``` in the configuration of executors, both global and in step
overrides. Variables should follow the
[python template strings](https://docs.python.org/3/library/string.html#template-strings)
syntax and are replaced with environment variable prefixed by ```MAGNUS_VAR_<identifier>```.

Concretely, ```$identifier``` is replaced by ```MAGNUS_VAR_<identifier>```.


### Dockerfile

magnus should be installed in the docker image and available in the path. An example dockerfile is provided
below.

!!! note inline end "non-native orchestration"

    Having magnus to be part of the docker image adds additional dependencies for python to be present in the docker
    image. In that sense, magnus is technically non-native container orchestration tool.

    Facilitating native container orchestration, without magnus as part of the docker image, results in a complicated
    specification of files/parameters/experiment tracking losing the value of native interfaces to these essential
    orchestration concepts.

    With the improvements in python packaging ecosystem, it should be possible to distribute magnus as a
    self-contained binary and reducing the dependency on the docker image.

#### TODO: Change this to a proper example.
```dockerfile linenums="1"
--8<-- "examples/Dockerfile"
```
