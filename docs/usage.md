
## Installation

**runnable** is a python package and should be installed like any other python package. The minimum python version
is ```3.8```

```shell
pip install runnable
```

We recommend the installation in a virtual environment using ```poetry``` or any other package manager.

### Extras

The below extras expand the functionality of ```runnable``` to different environments.

They can be installed by ```"pip install runnable[<extra>]"```

- ```docker``` : enable pipelines/jobs in a container
- ```notebook``` : enables notebooks as tasks/jobs
- ```k8s``` : enables running jobs in kubernetes or minikube clusters
- ```s3``` : enables using ```s3``` buckets for ```run log store``` and ```catalog```


## Usage

### Execute a pipeline

Pipelines defined in **runnable** can be via the python sdk.


The options are detailed below:

```shell

runnable execute

The runnable CLI is primarily used for running Python-based pipelines. While some legacy YAML functionality may still exist in the CLI, we recommend using the Python SDK for all new pipeline development.

```


### Execute a job

Jobs defined in **runnable** can be via the python sdk

<hr style="border:2px dotted orange">

## Examples

All the examples in the documentation are present in the ```examples``` directory of
[the repo](https://github.com/AstraZeneca/runnable-core) with instructions on how to run them.

All the examples are tested, with multiple configurations, as part of our CI test suite.
