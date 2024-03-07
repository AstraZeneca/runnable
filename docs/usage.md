
## Installation

**runnable** is a python package and should be installed like any other python package. The minimum python version
is ```3.8```

```shell
pip install runnable
```

We recommend the installation in a virtual environment using ```poetry``` or any other package manager.

### Extras

#### Docker

To run the pipelines/functions/notebooks in a container, install runnable with docker functionality.

```shell
pip install "runnable[docker]"
```

#### Notebook

To use notebooks as tasks, install runnable with ```notebook``` functionality.

```shell
pip install "runnable[notebook]"
```


<hr style="border:2px dotted orange">

## Usage

Pipelines defined in **runnable** can be either via [python sdk](sdk.md) or ```yaml``` based definitions.

To execute a pipeline, defined in ```yaml```, use the **runnable** cli.
The options are detailed below:

- ```-f, --file``` (str): The pipeline definition file, defaults to pipeline.yaml
- ```-c, --config-file``` (str): [config file](configurations/overview.md) to be used for the run [default: None]
- ```-p, --parameters-file``` (str): [Parameters](concepts/parameters.md)  accessible by the application [default: None]
- ```--log-level``` : The log level, one of ```INFO | DEBUG | WARNING| ERROR| FATAL``` [default: INFO]
- ```--tag``` (str): A tag attached to the run[default: ]
- ```--run-id``` (str): An optional run_id, one would be generated if not provided
- ```--use-cached``` (str): Provide the previous run_id to re-run.

<hr style="border:2px dotted orange">

## Examples

All the examples in the documentation are present in the ```examples``` directory of
[the repo](https://github.com/AstraZeneca/runnable-core) with instructions on how to run them.

All the examples are tested, with multiple configurations, as part of our CI test suite.
