# Jobs on Kubernetes

Kubernetes is a powerful cloud agnostic platform and this extension provides a way to run batch jobs on Kubernetes.
Note that this extension is only for jobs and not for any pipelines. Please refer to argo or Kubeflow to run pipelines
on Kubernetes.

## Additional dependencies

Magnus extensions needs additional packages to use this extension. Please install magnus-extensions via:

```pip install "magnus_extensions[k8s]"```

or

```poetry add "magnus_extensions[k8s]"```

Since kubernetes is a cloud based job scheduler, other services which are not accessible by cloud would not work.

## Configuration:

```yaml
executor:
  type: "k8s-job"
  config:
    config_path: str # Required
    docker_image: str # Required
    namespace: str # Defaults to "default"
    cpu_limit: str # Defaults to "250m"
    memory_limit: str # Defaults to "1G"
    gpu_limit: int # Defaults to 0
    gpu_vendor: str # Defaults to "nvidia.com/gpu"
    cpu_request: str # Defaults to cpu_limit
    memory_request: str # Defaults to memory_limit
    active_deadline_seconds: int # Defaults to 2 hours
    ttl_seconds_after_finished: int   #  Defaults to 1 minute
    image_pull_policy: str # Defaults to  "Always"
    secrets_from_k8s: dict # EnvVar=SecretName:Key
    persistent_volumes: dict # volume-name:mount_path
    labels: Dict[str, str]
```


- ### config_path

The location of the kubeconfig file to submit jobs.

- ### docker_image

The docker image to use to run the job. The docker image should be accessible from the Kubernetes cluster.

- ### namespace

The namespace of the Kubernetes cluster to submit the jobs to. It defaults to "default".

- ### cpu_limit

The default CPU limit for Kubernetes job. Defaults to "250m".
[Please refer to this documentation to understand more](https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/)

- ### memory_limit

The default memory limit for Kubernetes job. Defaults to 1G
[Please refer to this documentation to understand more](https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/)

- ### gpu_limit

The default GPU limit for Kubernetes job. Defaults to 0.
[Please refer to this documentation to understand more](https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/)

- ### gpu_vendor

The GPU type to use for Kubernetes job. The cluster should support the GPU type for this to work.
Defaults to nvidia.com/gpu.
[Please refer to this documentation to understand more.]https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/


- ### cpu_request

The default CPU request for Kubernetes job. Defaults to cpu_limit.
[Please refer to this documentation to understand more](https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/)

- ### memory_request

The default memory request for Kubernetes job. Defaults to memory_limit
[Please refer to this documentation to understand more](https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/)

- ### active_deadline_seconds

The maximum amount of time that the job can run on the kubernetes cluster. Defaults to 2 hours.
Please use this value appropriately for your job.

[Please refer to this documentation to understand more.](https://kubernetes.io/docs/concepts/workloads/controllers/job/#job-termination-and-cleanup)

- ### ttl_seconds_after_finished

The amount of time that the job/pod should be active after completing the job. Defaults to 1 minute.
Please increase this time (in seconds) if you want to look into more debugging information.


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

- ### persistent_volumes

Volumes to mount from the underlying cluster onto the container during the execution of the job.

The format is ```name-of-the-volume:mountpoint```.

- ### labels

Any labels that you wish to apply to the job.
