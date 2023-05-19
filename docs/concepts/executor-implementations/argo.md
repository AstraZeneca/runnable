# Argo workflows

[Argo workflows](https://argoproj.github.io/argo-workflows/) is a powerful workflow specification engine for K8's.
Magnus can *transpile* the pipeline definition to a Argo workflow specification.

## Features not implemented

- We have not yet implemented parallel, dag or map state in this extension.
- on_failure has not be implemented yet.


## Additional dependencies

Please install magnus-extensions via:

```pip install magnus_extensions```

or

```poetry add magnus_extensions```


Since argo is a cloud based orchestration tool, other services which are not accessible by cloud would not work.

## Configuration

The full configuration to use Argo extension:

```yaml
executor:
  type: "argo"
  config:
    docker_image: str
    output_file: str
    cpu_limit: str
    memory_limit: str
    cpu_request: str
    memory_request: str
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
        argo:
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

All the parameters that are defined via the ```--parameters``` option are available via the Argo CLI or WebUI of
kubeflow to configure during run time.

## Design Guidelines

- As the container is generated before the argo workflow is generated, it is a good idea to parameterize the docker
image id as part of the definition.

```yaml
mode:
  type: "argo"
  config:
    docker_image: ${docker_image}

```

The variable ```docker_image``` can then be provided during the run time of magnus.

```shell
export MAGNUS_VAR_docker_image=my_cool_image
magnus execute --file getting-started.yaml --config kubeflow.yaml
```

- Generate and deploy the argo pipelines as part of the CI.
