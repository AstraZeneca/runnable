Executors are the heart of magnus, they traverse the workflow and execute the tasks within the
workflow while coordinating with different services
(eg. [run log](../concepts/run-log.md), [catalog](../concepts/catalog.md), [secrets](../concepts/secrets.md) etc)

To enable workflows run in varied computational environments, we distinguish between two core functions of
any workflow engine.


`Graph Traversal`

:   Involves following the user-defined workflow graph to its eventual conclusion.
    The navigation process encompasses the sequential execution of tasks or complex tasks
    such as parallel paths. It also includes decision-making regarding the
    pathway to follow in case of task failure and the upkeep of the
    overall status of graph execution.

`Executing Individual Steps`

:   This refers to the concrete execution of the task as specified by the user
    along with allowing for data flow between tasks.
    This could involve activities such as launching a container or initiating a SQL query,
    among others.

## Graph Traversal

In magnus, the graph traversal can be performed by magnus itself or can be handed over to other
orchestration frameworks (e.g Argo workflows, AWS step functions).

### Example

Below is a simple pipeline definition that does one task of printing "Hello World".

```yaml linenums="1"
--8<-- "examples/concepts/task_shell_simple.yaml"
```

The above pipeline can be executed by the *default* config to execute it locally or could be
translated to argo specification just by changing the configuration.

=== "Default Configuration"

    The configuration defines the local compute to the execution environment with the ```run log```
    being completely in memory and buffered with no other services active.

    You can execute the pipeline in default configuration by:

    ```magnus execute -f examples/concepts/task_shell_simple.yaml```

    ``` yaml linenums="1"
    --8<-- "examples/configs/default.yaml"
    ```

    1. Run the pipeline in local environment.
    2. Use the buffer as run log, this will not persist the run log to disk.
    3. Do not move any files to central storage.
    4. Do not use any secrets manager.
    5. Do not integrate with any experiment tracking tools

=== "Argo Configuration"

    In this configuration, we are using [argo workflows](https://argoproj.github.io/argo-workflows/)
    as our workflow engine. We are also instructing the workflow engine to use a docker image,
    ```magnus:demo``` defined in line #4, as our execution environment. Please read
    [containerised environments](../configurations/executors/container-environments.md) for more information.

    Since magnus needs to track the execution status of the workflow, we are using a ```run log```
    which is persistent and available in for jobs in kubernetes environment.


    You can execute the pipeline in argo configuration by:

    ```magnus execute -f examples/concepts/task_shell_simple.yaml -c examples/configs/argo-config.yaml```

    ``` yaml linenums="1"
    --8<-- "examples/configs/argo-config.yaml"
    ```

    1. Use argo workflows as the execution engine to run the pipeline.
    2. Run this docker image for every step of the pipeline. The docker image should have the same directory structure
    as the project directory.
    3. Mount the volume from Kubernetes persistent volumes (magnus-volume) to /mnt directory.
    4. Resource constraints for the container runtime.
    5. Since every step runs in a container, the run log should be persisted. Here we are using the file-system as our
    run log store.
    6. Kubernetes PVC is mounted to every container as ```/mnt```, use that to surface the run log to every step.


=== "Transpiled Workflow"

    In the below generated argo workflow template:

    - Lines 10-17 define a ```dag``` with tasks that corresponding to the tasks in
    the example workflow.
    - The graph traversal rules follow the the same rules as our workflow. The
    step ```success-success-ou7qlf``` in line #15 only happens if the step ```shell-task-dz3l3t```
    defined in line #12 succeeds.
    - The execution fails if any of the tasks fail. Both argo workflows and magnus ```run log```
    mark the execution as failed.


    ```yaml linenums="1"
    apiVersion: argoproj.io/v1alpha1
    kind: Workflow
    metadata:
      generateName: magnus-dag-
      annotations: {}
      labels: {}
    spec:
      activeDeadlineSeconds: 172800
      entrypoint: magnus-dag
      podGC:
        strategy: OnPodCompletion
      retryStrategy:
        limit: '0'
        retryPolicy: Always
        backoff:
          duration: '120'
          factor: 2
          maxDuration: '3600'
      serviceAccountName: default-editor
      templates:
        - name: magnus-dag
          failFast: true
          dag:
            tasks:
              - name: shell-task-4jy8pl
                template: shell-task-4jy8pl
                depends: ''
              - name: success-success-djhm6j
                template: success-success-djhm6j
                depends: shell-task-4jy8pl.Succeeded
        - name: shell-task-4jy8pl
          container:
            image: magnus:demo
            command:
              - magnus
              - execute_single_node
              - '{{workflow.parameters.run_id}}'
              - shell
              - --log-level
              - WARNING
              - --file
              - examples/concepts/task_shell_simple.yaml
              - --config-file
              - examples/configs/argo-config.yaml
            volumeMounts:
              - name: executor-0
                mountPath: /mnt
            imagePullPolicy: ''
            resources:
              limits:
                memory: 1Gi
                cpu: 250m
              requests:
                memory: 1Gi
                cpu: 250m
        - name: success-success-djhm6j
          container:
            image: magnus:demo
            command:
              - magnus
              - execute_single_node
              - '{{workflow.parameters.run_id}}'
              - success
              - --log-level
              - WARNING
              - --file
              - examples/concepts/task_shell_simple.yaml
              - --config-file
              - examples/configs/argo-config.yaml
            volumeMounts:
              - name: executor-0
                mountPath: /mnt
            imagePullPolicy: ''
            resources:
              limits:
                memory: 1Gi
                cpu: 250m
              requests:
                memory: 1Gi
                cpu: 250m
      templateDefaults:
        activeDeadlineSeconds: 7200
        timeout: 10800s
      arguments:
        parameters:
          - name: run_id
            value: '{{workflow.uid}}'
      volumes:
        - name: executor-0
          persistentVolumeClaim:
            claimName: magnus-volume


    ```


As seen from the above example, once a [pipeline is defined in magnus](../concepts/pipeline.md) either via yaml or SDK, we can
run the pipeline in different environments just by providing a different configuration. Most often, there is
no need to change the code or deviate from standard best practices while coding.


## Step Execution

!!! note

    This section is to understand the internal mechanism of magnus and not required if you just want to
    use different executors.


Independent of traversal, all the tasks are executed within the ```context``` of magnus.

A closer look at the actual task implemented as part of transpiled workflow in argo
specification details the inner workings. Below is a snippet of the argo specification from
lines 18 to 34.

```yaml linenums="18"
- name: shell-task-dz3l3t
  container:
    image: magnus-example:latest
    command:
    - magnus
    - execute_single_node
    - '{{workflow.parameters.run_id}}'
    - shell
    - --log-level
    - WARNING
    - --file
    - examples/concepts/task_shell_simple.yaml
    - --config-file
    - examples/configs/argo-config.yaml
  volumeMounts:
    - name: executor-0
      mountPath: /mnt
```

The actual ```command``` to run is not the ```command``` defined in the workflow,
i.e ```echo hello world```, but a command in the CLI of magnus which specifies the workflow file,
the step name and the configuration file.

### Context of magnus

Any ```task``` defined by the user as part of the workflow always runs as a *sub-command* of
magnus. In that sense, magnus follows the
[decorator pattern](https://en.wikipedia.org/wiki/Decorator_pattern) without being part of the
application codebase.

In a very simplistic sense, the below stubbed-code explains the context of magnus during
execution of a task.

```python linenums="1"

def execute_single_node(workflow, step_name, configuration):

    ##### PRE EXECUTION #####
    # Instantiate the service providers of run_log and catalog
    # These are provided as part of the configuration.
    run_log = configuration.get_run_log() # (1)
    catalog = configuration.get_catalog() # (2)

    step = workflow.get_step(step_name) # (3)

    # Get the current parameters set by the initial parameters
    # or by previous steps.
    existing_parameters = run_log.get_parameters()
    # Get the data requested by the step and populate
    # the data folder defined in the catalog configuration
    catalog.get_data(step.get_from_catalog) # (4)

    # Choose the parameters to pass into the function and
    # the right data type.
    task_parameters = filter_and_cast_parameters(existing_parameters, step.task) # (5)

    ##### END PRE EXECUTION #####
    try:
      # We call the actual task here!!
      updated_parameters = step.task(**task_parameters) # (6)
    except:
      update_status_in_run_log(step, FAIL)
      send_error_response() # (7)

    ##### POST EXECUTION #####
    run_log.update_parameters(updated_parameters) # (8)
    catalog.put_data(step.put_into_catalog) # (9)
    update_status_in_run_log(step, SUCCESS)
    send_success_response() # (10)
    ##### END POST EXECUTION #####
```

1. The [run log](../concepts/run-log.md) maintains the state of the execution of the tasks and subsequently the pipeline. It also
holds the latest state of parameters along with captured metrics.
2. The [catalog](../concepts/catalog.md) contains the information about the data flowing through the pipeline. You can get/put
artifacts generated during the current execution of the pipeline to a central storage.
3. Read the workflow and get the [step definition](../concepts/task.md) which holds the ```command``` or ```function``` to
execute along with the other optional information.
4. Any artifacts from previous steps that are needed to execute the current step can be
[retrieved from the catalog](../concepts/catalog.md).
5. The current function or step might need only some of the
[parameters casted as pydantic models](../concepts/task.md/#accessing_parameters), filter and cast them appropriately.
6. At this point in time, we have the required parameters and data to execute the actual command. The command can
internally request for more data using the [python API](..//interactions.md) or record
[experiment tracking metrics](../concepts/experiment-tracking.md).
7. If the task failed, we update the run log with that information and also raise an exception for the
workflow engine to handle. Any [on-failure](../concepts/pipeline.md/#on_failure) traversals are already handled
as part of the workflow definition.
8. Upon successful execution, we update the run log with current state of parameters for downstream steps.
9. Any artifacts generated from this step are [put into the central storage](../concepts/catalog.md) for downstream steps.
10. We send a success message to the workflow engine and mark the step as completed.
