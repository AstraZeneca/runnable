# Kubeflow

[Kubeflow pipelines](https://www.kubeflow.org/docs/components/pipelines/v1/introduction/) is a popular tool to
orchestrate ML jobs. Magnus can *transpile* the pipeline definition to a Kubeflow pipeline.

## Features not implemented

- We have not yet implemented parallel, dag or map state in this extension.
- Kubeflow does not have an easy way to have "on_failure", this feature is not yet implemented.
- We do not use any ```volumes``` or ```pvolumes``` resources defined by kubeflow as part of this extension.

In general, we might prefer to transpile to Argo directly instead of going via Kubeflow in the future.

## Additional dependencies

Please install magnus-extensions via:

```pip install magnus_extensions[kubeflow]```

or

```poetry add magnus_extensions[kubeflow]```

Since kubeflow is a cloud based orchestration tool, other services which are not accessible by cloud would not work.

## Configuration

The full configuration to use Kubeflow extension:

```yaml
executor:
  type: "kfp"
  config:
    docker_image: str
    output_file: str
    cpu_limit: str
    memory_limit: str
    cpu_request: str
    memory_request: str
    gpu_limit: int
    enable_caching: bool
    image_pull_policy: str
    secrets_from_k8s: dict
```

Individual steps of the dag can over-ride the configuration by providing a ```mode_config``` section.

```yaml
dag:
  steps:
    step:
    ...
      executor_config:
        kfp:
          docker_image: # Overrides the default docker_image
          secrets_from_k8s: # Overrides the default secrets_from_k8's
            ...
          cpu_limit: # Overrides the default cpu_limit
          cpu_request: # Overrides the default cpu_request
          memory_limit: # Overrides the default memory_limit
          memory_request: # Overrides the default memory_request
          image_pull_policy: # Overrides the default image pull policy
     ...
```

- ### docker_image:

The docker image to use to run the dag/step.

- ### output_file:

The name of the output file to write the generated pipeline definition via Kubeflow.
Defaults to ```pipeline.yaml```.

- ### cpu_limit:

The default cpu limit from K8's. Defaults to 250m.

- ### memory_limit:

The default memory limit from K8's. Defaults to 1G.

- ### cpu_request:

The default cpu to request from K8's. If not provided, it is the same as default_cpu_limit.

- ### memory_request:

The default memory request from K8's. if not provided, it is the same as default_memory_limit.

- ### gpu_limit:

The gpu's to request from K8's. Defaults to 0.

- ### enable_caching:

Controls the caching behavior of Kubeflow, defaults to False.

- ### image_pull_policy:

Set to "Always", the available options are: "IfNotPresent", "Always", "Never".


!!! Warning

    Use "IfNotPresent" cautiously, as the check happens on the tag of the docker image and an improper versioning strategy
    might result in wrong docker images being used.

- ### secrets_from_k8s:

Use secrets stored in underlying K8's while running the containers.
The format is ```EnvVar=SecretName:Key``` where

    - EnvVar is the name of the Environment variable the secret should be in the container.
    - SecretName: The name of the secret in K8's.
    - Key: The key in the secret that should be exposed in the container.


## Parameters

All the parameters that are defined via the ```--parameters``` option are available via the WebUI of kubeflow to
configure during run time.


## Design Guidelines

- As the container is generated before the kubeflow pipeline is generated, it is a good idea to parameterize the docker
image id as part of the definition.

```yaml
mode:
  type: "kfp"
  config:
    docker_image: ${docker_image}

```

The variable ```docker_image``` can then be provided during the run time of magnus.

```shell
export MAGNUS_VAR_docker_image=my_cool_image
magnus execute --file getting-started.yaml --config kubeflow.yaml
```

- Generate and deploy the kubeflow pipelines as part of the CI.
