[Argo workflows](https://argo-workflows.readthedocs.io/en/latest/) is a powerful
container orchestration framework for Kubernetes and it can run on any Kubernetes environment.

**magnus** will transpile pipeline definition to argo specification during the pipeline execution which
you can then upload to the cluster either manually or via CICD (recommended).

- [x] Execute the pipeline in any cloud environment.
- [x] Massively scalable.
- [x] Ability to provide specialized compute environments for different steps of the pipeline.
- [ ] Expects a mature cloud kubernetes environment and expertise.

Magnus provides *sensible* defaults to most of the configuration variables but it is highly advised
to get inputs from infrastructure teams or ML engineers in defining the configuration.


## Configuration

Only ```image``` is the required parameter. Please refer to the
[note on containers](../container-environments) on building images.


```yaml linenums="1"
executor:
  type: argo
  config:
    name:
    annotations:
    labels:
    namespace:
    image: <required>
    pod_gc:
    max_workflow_duration_in_seconds:
    node_selector:
    parallelism:
    service_account_name:
    resources:
    retry_strategy:
    max_step_duration_in_seconds:
    tolerations:
    image_pull_policy:
    expose_parameters_as_inputs:
    output_file:
    secrets_from_k8s:
```


### Defaults


!!! warning "Default values"

    Ensure that these default values fit your needs to avoid unexpected behavior.

<div class="annotate" markdown>

| Parameter      | Default | Argo Field |
| :-----------: | :-------------: | :------------: |
| name | ```magnus-dag-``` | ```generateName``` |
| annotations | ```{}``` | ```annotations``` of ```metadata``` |
| labels | ```{}``` | ```labels``` |
| namespace | ```None``` | ```namespace```|
| pod_gc       | ```OnPodCompletion```  | ```podGC``` |
| max_workflow_duration_in_seconds | 86400 seconds = 1 day | ```activeDeadlineSeconds``` of spec |
| node_selector | ```{}``` | ```nodeSelector``` |
| parallelism | ```None``` | ```parallelism``` of spec |
| service_account_name | ```None``` | ```serviceAccountName``` of spec |
| resources | limits: 1Gi of memory and 250m of CPU | ```resources``` of the container |
| retry_strategy | ```None``` | ```retryStrategy``` of the spec |
| max_step_duration_in_seconds | 60 * 60 * 2 = 2 hours | ```activeDeadlineSeconds``` of container |
| tolerations | ```{}``` | ```tolerations``` of the container |
| image_pull_policy | ```"" ``` | ```imagePullPolicy``` of the container |


</div>

### Notes

- name: Using a name provides a logical way to organize pipelines.
- pdd_gc: In development phase or during debugging, set this to ```None``` to debug the pod in case
of failure.
- node_selector and tolerations: Gives you the ability to selectively choose a node to run your task.
- parallelism: This can make ```parallel``` and ```map``` steps run sequentially. To control the
parallelism of a ```map``` or ```parallel```, provide an ```override``` in the overrides section.
- resources: they have the same structure as K8's manifests.
```yaml
resources:
  limits:
    memory: 2Gi
    cpu: 1
  gpu: 1
```

#### Useful tips

!!! Tip

    - pod_gc: During development phase of the workflow, set it to empty to debug any failed pods.
    - parallelism of workflow: Setting it to 1, would result in one task running at all times.
    - resources: Set both limits and requests to appropriate values that the container needs.
    - retry strategy: Along with limit, you can also set backoff strategy.
    - max_step_duration_in_seconds: Magnus sets the timeout to be 1 hour longer.
    - Assigns to not pull for any tag other than ```latest```. Ensure that the docker image corresponds to the correct image that you want to execute.
