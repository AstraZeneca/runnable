
Execute all the steps of the pipeline in containers. Please refer to the
[note on containers](container-environments.md) on building images.

- [x] Provides a way to test the containers and the execution of the pipeline in local environment.
- [x] Any failure in cloud native container environments can be replicated in local environments.
- [x] Ability to provide specialized compute environments for different steps of the pipeline.
- [ ] The scalability is still constrained by the resources in local environment.


!!! warning inline end "parallel executions"

    Run logs that use a single json (eg. file-system) are not compatible with parallel
    executions due to race conditions to write the same file by different processes.

    Use ```chunked``` run log stores (eg. chunked-fs).



## Configuration

```yaml
executor: local-container
config:
  docker_image: <required>
  enable_parallel: false # (1)
  auto_remove_container: true # (2)
  run_in_local: false # (3)
  environment: # (4)
    ...
  overrides: # (5)
    ...
```

 1. By default, all tasks are sequentially executed. Provide ```true``` to enable tasks within
[parallel](../../concepts/parallel.md) or [map](../../concepts/map.md) to be executed in parallel.
2. Set it to false, to debug a failed container.
3. Setting it to true will behave exactly like a [local executor](local.md).
4. Pass any environment variables into the container.
5. Please refer to [step overrides](#step_override) for more details.

The ```docker_image``` field is required and default image to execute tasks
of the pipeline. Individual [tasks](../../concepts/task.md) can
[override](#step_override) the global defaults of executor by providing ```overrides```


!!! tip "Debugging"

    ```auto_remove_container``` allows you to run the failed container independently to
    identify the issue that caused the failure.

    ```run_in_local``` allows you to execute a few tasks in local environment to allow
    debugging and also selectively choose which step to run in container.


## Example

Nearly all the examples seen in concepts can be executed using
the ```local-container``` configuration. Below is one simple example to concretely show
the patterns.

=== "Configuration"

    Assumed to be present at ```examples/configs/local-container.yaml```

    The docker image is a [variable](container-environments.md/#dynamic_name_of_the_image) and
    dynamically set during execution.

    ```yaml linenums="1" hl_lines="4"
    --8<-- "examples/configs/local-container.yaml"
    ```

    1. Use local-container executor type to execute the pipeline.
    2. By default, all the tasks are executed in the docker image . Please
    refer to [building docker images](container-environments.md/#dynamic_name_of_the_image)
    3. Pass any environment variables that are needed for the container.
    4. Store the run logs in the file-system. runnable will handle the access to them
    by mounting the file system into the container.


=== "python sdk"

    Running the SDK defined pipelines for any container based executions [happens in
    multi-stage process](container-environments.md).

    1. Generate the ```yaml``` definition file by:
    ```runnable_CONFIGURATION_FILE=examples/configs/local-container.yaml python examples/concepts/simple.py```
    2. Build the docker image with yaml definition in it, called runnable:demo in current example.
    3. Execute the pipeline via the runnable CLI,
    ```runnable_VAR_default_docker_image=runnable:demo  runnable execute -f runnable-pipeline.yaml -c examples/configs/local-container.yaml```


    ```python linenums="1" hl_lines="24"
    --8<-- "examples/concepts/simple.py"
    ```

    1. You can provide a configuration file dynamically by using the environment
    variable ```runnable_CONFIGURATION_FILE```. Please see [SDK for more details](../../sdk.md).



=== "yaml"

    For yaml based definitions, the execution order is to:

    1. Build the docker image with the yaml definition in it, called runnable:demo in current example.
    2. Execute the pipeline via the runnable CLI:
    ```runnable_VAR_default_docker_image=runnable:demo runnable execute -f examples/concepts/simple.yaml -c examples/configs/local-container.yaml```

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
                      "code_identifier_url": "https://github.com/AstraZeneca/runnable-core.git",
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
                      "code_identifier_url": "https://github.com/AstraZeneca/runnable-core.git",
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
              "default_docker_image": "runnable:demo"
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

## Compatibility


## Step override

Individual steps of the pipeline can over-ride the default configuration by referring to the
specific ```override``` defined in ```overrides``` section of the executor configuration.

```override``` should be defined per executor and is only applicable for that specific
executor.

### Example


=== "Configuration"

    Assumed to be present at ```examples/executors/local-container-override.yaml```

    In the example below, we define the default configuration in the executor configuration.
    We also provide a override ```custom_docker_image``` which overrides some of the default
    configuration parameters.


    ```yaml linenums="1" hl_lines="7-11"
    --8<-- "examples/executors/local-container-override.yaml"
    ```

=== "python sdk"

    As seen in the above example,
    running the SDK defined pipelines for any container based executions [happens in
    multi-stage process](container-environments.md).

    1. Generate the ```yaml``` definition file by:
    ```runnable_CONFIGURATION_FILE=examples/executors/local-container-override.yaml python examples/executors/step_overrides_container.py```
    2. Build the docker image with yaml definition in it. In this example, we build
    two docker images.

        * runnable:3.8 as the default_docker_image.
        * runnable:3.9 as the custom_docker_image.

      Both the docker images are same except for the python version.

    3. Execute the pipeline via the runnable CLI,
    ```runnable_VAR_default_docker_image=runnable:3.8  runnable_VAR_custom_docker_image=runnable:3.9 runnable execute -f runnable-pipeline.yaml -c examples/executors/local-container-override.yaml```


    You should see the console output of the ```step 1``` to be ```3.8``` and key to be "value"
    while the python version for ```step 2``` to be 3.9 and key to be "not-value".

    ```python linenums="1" hl_lines="26"
    --8<-- "examples/executors/step_overrides_container.py"
    ```


=== "yaml"

    For yaml based definitions, the execution order is to:

    1. Build the docker image with the yaml definition in it. In this example, we build
    two docker images.


        * runnable:3.8 as the default_docker_image.
        * runnable:3.9 as the custom_docker_image.


        Both the docker images are same except for the python version.


    2. Execute the pipeline via the runnable CLI:
    ```runnable_VAR_default_docker_image=runnable:3.8 runnable_VAR_custom_docker_image=runnable:3.9 runnable execute -f examples/executors/step_overrides_container.yaml -c examples/executors/local-container-override.yaml```

    You should see the console output of the ```step 1``` to be ```3.8``` and key to be "value"
    while the python version for ```step 2``` to be 3.9 and key to be "not-value".

    ```yaml linenums="1" hl_lines="29-30"
    --8<-- "examples/executors/step_overrides_container.yaml"
    ```
