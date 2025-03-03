
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
- ```torch``` : enables to run pytorch jobs or as tasks in pipeline


## Usage

### Execute a pipeline

Pipelines defined in **runnable** can be either via [python sdk](reference.md) or ```yaml``` based definitions.


The options are detailed below:

```shell

runnable execute

╭─ Arguments ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *    yaml_file      TEXT      The pipeline definition file [default: None] [required]                                                              │
│      run_id         [RUN_ID]  An optional run_id, one would be generated if its not provided [env var: RUNNABLE_RUN_ID]                            │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --config      -c      TEXT                              The configuration file specifying the services                                             │
│ --parameters  -p      TEXT                              Parameters, in yaml,  accessible by the application                                        │
│ --log-level           [INFO|DEBUG|WARNING|ERROR|FATAL]  The log level [default: WARNING]                                                           │
│ --tag                 TEXT                              A tag attached to the run                                                                  │
│ --help                                                  Show this message and exit.                                                                │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

```


### Execute a job

Jobs defined in **runnable** can be either via [python sdk](reference.md) or ```yaml``` based definitions.

The options are detailed below:

```shell
Usage: runnable submit-job [OPTIONS] JOB_DEFINITION_FILE

╭─ Arguments ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *    job_definition_file      TEXT  The yaml file containing the job definition [default: None] [required]                                         │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --config      -c      TEXT                              The configuration file specifying the services                                             │
│ --parameters  -p      TEXT                              Parameters, in yaml,  accessible by the application                                        │
│ --log-level           [INFO|DEBUG|WARNING|ERROR|FATAL]  The log level [default: WARNING]                                                           │
│ --tag                 TEXT                              A tag attached to the run                                                                  │
│ --run-id              TEXT                              An optional run_id, one would be generated if its not provided                             │
│ --help                                                  Show this message and exit.                                                                │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```


<hr style="border:2px dotted orange">

## Examples

All the examples in the documentation are present in the ```examples``` directory of
[the repo](https://github.com/AstraZeneca/runnable-core) with instructions on how to run them.

All the examples are tested, with multiple configurations, as part of our CI test suite.
