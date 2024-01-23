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


---


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
  placeholders:
    ...
```

 1. By default, all tasks are sequentially executed. Provide ```true``` to enable tasks within
[parallel](../../concepts/parallel) or [map](../../concepts/map) to be executed in parallel.
2. Set it to false, to debug a failed container.

The ```docker_image``` field is required and default image to execute tasks
of the pipeline. Individual [tasks](../../concepts/task) can over-ride the
default image by providing an ```executor_config``` as part of step definition.

#### Example

Nearly all the examples seen in concepts can be executed using
the ```local-container``` configuration. Below is one simple example to concretely show
the patterns.

=== "Local container config"

    Assumed to be present at ```examples/configs/local-container.yaml```

    The docker image is a [variable](#dynamic_name_of_the_image) and
    dynamically set during execution.

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

    Running the SDK defined pipelines for any container based executions [happens in
    multi-stage process](#container_environments).

    1. Generate the ```yaml``` definition file by:
    ```MAGNUS_CONFIGURATION_FILE=examples/configs/local-container.yaml python examples/concepts/simple.py```
    2. Build the docker image with yaml definition in it, called magnus:demo in current example.
    3. Execute the pipeline via the magnus CLI,
    ```MAGNUS_VAR_default_docker_image=magnus:demo  magnus execute -f magnus-pipeline.yaml -c examples/configs/local-container.yaml```


    ```python linenums="1" hl_lines="24"
    --8<-- "examples/concepts/simple.py"
    ```

    1. You can provide a configuration file dynamically by using the environment
    variable ```MAGNUS_CONFIGURATION_FILE```. Please see [SDK for more details](../../sdk).



=== "yaml"

    For yaml based definitions, the execution order is to:

    1. Build the docker image with the yaml definition in it, called magnus:demo in current example.
    2. Execute the pipeline via the magnus CLI:
    ```MAGNUS_VAR_default_docker_image=magnus:demo magnus execute -f examples/concepts/simple.yaml -c examples/configs/local-container.yaml```

    ```yaml linenums="1"
    --8<-- "examples/concepts/simple.yaml"
    ```

=== "Run log"

    The run log structure is the same as any other ```local``` executions apart from
    an additional code identity with the information about the docker image.


    ```json linenums="1" hl_lines="24-30"
    {
      "run_id": "shortest-stallman-2113",
      "dag_hash": "d467805d7f743d459a6abce95bedbfc6c1ecab67",
      "use_cached": false,
      "tag": "",
      "original_run_id": "",
      "status": "SUCCESS",
      "steps": {
          "simple": {
              "name": "simple",
              "internal_name": "simple",
              "status": "SUCCESS",
              "step_type": "task",
              "message": "",
              "mock": false,
              "code_identities": [
                  {
                      "code_identifier": "ef142998dc315ddbd9aa10e016128c872de6e6e1",
                      "code_identifier_type": "git",
                      "code_identifier_dependable": true,
                      "code_identifier_url": "https://github.com/AstraZeneca/magnus-core.git",
                      "code_identifier_message": ""
                  },
                  {
                      "code_identifier": "sha256:e5cc0936aad4d3cacb3075290729ce834dd2d9c89ea24eea609d7664f99ce50f",
                      "code_identifier_type": "docker",
                      "code_identifier_dependable": true,
                      "code_identifier_url": "local docker host",
                      "code_identifier_message": ""
                  }
              ],
              "attempts": [
                  {
                      "attempt_number": 1,
                      "start_time": "2024-01-22 21:13:53.676698",
                      "end_time": "2024-01-22 21:13:53.678976",
                      "duration": "0:00:00.002278",
                      "status": "SUCCESS",
                      "message": "",
                      "parameters": {}
                  }
              ],
              "user_defined_metrics": {},
              "branches": {},
              "data_catalog": [
                  {
                      "name": "simple.execution.log",
                      "data_hash": "03ba204e50d126e4674c005e04d82e84c21366780af1f43bd54a37816b6ab340",
                      "catalog_relative_path": "shortest-stallman-2113/simple.execution.log",
                      "catalog_handler_location": "/tmp/catalog/",
                      "stage": "put"
                  }
              ]
          },
          "success": {
              "name": "success",
              "internal_name": "success",
              "status": "SUCCESS",
              "step_type": "success",
              "message": "",
              "mock": false,
              "code_identities": [
                  {
                      "code_identifier": "ef142998dc315ddbd9aa10e016128c872de6e6e1",
                      "code_identifier_type": "git",
                      "code_identifier_dependable": true,
                      "code_identifier_url": "https://github.com/AstraZeneca/magnus-core.git",
                      "code_identifier_message": ""
                  }
              ],
              "attempts": [
                  {
                      "attempt_number": 1,
                      "start_time": "2024-01-22 21:13:53.807381",
                      "end_time": "2024-01-22 21:13:53.807834",
                      "duration": "0:00:00.000453",
                      "status": "SUCCESS",
                      "message": "",
                      "parameters": {}
                  }
              ],
              "user_defined_metrics": {},
              "branches": {},
              "data_catalog": []
          }
      },
      "parameters": {},
      "run_config": {
          "executor": {
              "service_name": "local-container",
              "service_type": "executor",
              "enable_parallel": false,
              "placeholders": {}
          },
          "run_log_store": {
              "service_name": "file-system",
              "service_type": "run_log_store"
          },
          "secrets_handler": {
              "service_name": "do-nothing",
              "service_type": "secrets"
          },
          "catalog_handler": {
              "service_name": "file-system",
              "service_type": "catalog"
          },
          "experiment_tracker": {
              "service_name": "do-nothing",
              "service_type": "experiment_tracker"
          },
          "pipeline_file": "examples/concepts/simple.yaml",
          "parameters_file": null,
          "configuration_file": "examples/configs/local-container.yaml",
          "tag": "",
          "run_id": "shortest-stallman-2113",
          "variables": {
              "default_docker_image": "magnus:demo"
          },
          "use_cached": false,
          "original_run_id": "",
          "dag": {
              "start_at": "simple",
              "name": "",
              "description": null,
              "steps": {
                  "simple": {
                      "type": "task",
                      "name": "simple",
                      "next": "success",
                      "on_failure": "",
                      "executor_config": {},
                      "catalog": null,
                      "max_attempts": 1,
                      "command": "examples.concepts.simple.simple_function",
                      "command_type": "python",
                      "node_name": "simple"
                  },
                  "success": {
                      "type": "success",
                      "name": "success"
                  },
                  "fail": {
                      "type": "fail",
                      "name": "fail"
                  }
              }
          },
          "dag_hash": "d467805d7f743d459a6abce95bedbfc6c1ecab67",
          "execution_plan": "chained"
      }
    }
    ```


### Step override

Individual steps of the pipeline can over-ride the default configuration by providing an
```executor_config``` in the definition of the step. The parameter ```enable_parallel```
can only be set at the global level.

```executor_config``` should be defined per executor and is only applicable for that specific
executor.

#### Example


=== "Configuration"

    Assumed to be present at ```examples/configs/local-container.yaml```

    The docker image is a [variable](#dynamic_name_of_the_image) and
    dynamically set during execution and is the default docker image to run
    all the tasks.

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

    As seen in the above example,
    running the SDK defined pipelines for any container based executions [happens in
    multi-stage process](#container_environments).

    1. Generate the ```yaml``` definition file by:
    ```MAGNUS_CONFIGURATION_FILE=examples/configs/local-container.yaml python examples/executors/step_overrides_container.py```
    2. Build the docker image with yaml definition in it. In this example, we build
    two docker images.

      1. magnus:3.8 as the default_docker_image.
      2. magnus:3.9 as the custom_docker_image.

      Both the docker images are same except for the python version.

    3. Execute the pipeline via the magnus CLI,
    ```MAGNUS_VAR_default_docker_image=magnus:3.8  MAGNUS_VAR_custom_docker_image=magnus:3.9 magnus execute -f magnus-pipeline.yaml -c examples/configs/local-container.yaml```


    You should see the console output of the ```step 1``` to be ```3.8``` while the python
    version for ```step 2``` to be 3.9

    ```python linenums="1" hl_lines="4"
    --8<-- "examples/executors/step_overrides_container.py"
    ```


=== "yaml"

    For yaml based definitions, the execution order is to:

    1. Build the docker image with the yaml definition in it. In this example, we build
    two docker images.

      1. magnus:3.8 as the default_docker_image.
      2. magnus:3.9 as the custom_docker_image.

      Both the docker images are same except for the python version.

    2. Execute the pipeline via the magnus CLI:
    ```MAGNUS_VAR_default_docker_image=magnus:3.8 MAGNUS_VAR_custom_docker_image=magnus:3.9 magnus execute -f examples/executors/step_overrides_container.yaml -c examples/configs/local-container.yaml```

    You should see the console output of the ```step 1``` to be ```3.8``` while the python
    version for ```step 2``` to be 3.9

    ```yaml linenums="1" hl_lines="4"
    --8<-- "examples/executors/step_overrides_container.yaml"
    ```


### Placeholders






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
