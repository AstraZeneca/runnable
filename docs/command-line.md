# Command line options

## Executing a pipeline

You can execute a pipeline by the following command:

```shell
magnus execute
```

---
!!! Note

   For the above command to work, make sure you are in the environment where magnus was installed.

   If you are using poetry, you can also invoke magnus by ```poetry run magnus execute```

---

The complete options available are:

```
Usage: magnus execute [OPTIONS]

  Entry point to executing a pipeline. This command is most commonly used
  either to execute a pipeline or to translate the pipeline definition to
  another language.

  You can re-run an older run by providing the run_id of the older run in
  --use-cached. Ensure that the catalogs and run logs are accessible by the
  present configuration.

Options:
  -f, --file TEXT                 The pipeline definition file  [default:
                                  pipeline.yaml]
  -v, --var-file TEXT             Variables used in pipeline definition
  -c, --config-file TEXT          config file, in yaml, to be used for the run
  -p, --parameters-file TEXT      Parameters, in yaml,  accessible by the
                                  application
  --log-level [INFO|DEBUG|WARNING|ERROR|FATAL]
                                  The log level  [default: WARNING]
  --tag TEXT                      A tag attached to the run
  --run-id TEXT                   An optional run_id, one would be generated
                                  if not provided
  --use-cached TEXT               Provide the previous run_id to re-run.
  --help                          Show this message and exit.
```

### Dag definition/config

The file containing the dag definition and the config to be used.

Provided by ```-f```, ```--file``` option on magnus cli.

Defaults to ```pipeline.yaml``` if nothing is provided.

### Variables file

The yaml file containing the variables or placeholder values in the dag definition file.

Provided by ```-v```, ```--var-file``` option on magnus cli.

Defaults to None, if nothing is provided.
[Read more about parameterized definitions](../../concepts/dag/#parameterized_definition).


### Configurations file

The yaml file containing the configurations used to run magnus. The configurations provided here would over-ride any
configuration variables.

Provided by ```-c```, ```--config-file``` option on magnus cli.

Defaults to None, if nothing is provided.
Read more about different ways you can configure magnus runs here.


!!! warning "Changed in v0.2"

### Parameters file

The yaml file containing the initial set of parameters that the application can access. Individual steps of the
pipeline can still add/update parameters as required.

Provided by ```-p```, ```--parameters-file``` option to magnus cli.

Defaults to None, if nothing is provided.
You can also pass parameters by environmental variables prefixed by ```MAGNUS_PRM_```

### Log level

To control the logging level of magnus only. This does not affect your application logs what so ever.

Provided by ```--log-level``` option on magnus cli.

Available options are: DEBUG, INFO, WARNING, ERROR, CRITICAL.

Defaults to INFO if nothing is provided.

### Tag

A friendly way to tag experiments or runs together.

Provided by ```--tag``` option on magnus cli.

Defaults to None if nothing is provided.

### Run id

An unique run identifier for the run.

Provided by ```--run-id``` on magnus cli.

We generate one based on Timestamp if one is not provided.


### Use cached

Enables you to re-run a previous run provided by the run-id.

Example:

```shell
magnus execute --file example.yaml --run-id 20210506051758 --use-cached old_run_id
```


## Executing a single step

This method could be used to run a single step in isolation.

The complete options are:

```
Usage: magnus execute_step [OPTIONS] STEP_NAME

  Entry point to executing a single step of the pipeline.

  This command is helpful to run only one step of the pipeline in isolation.
  Only the steps of the parent dag could be invoked using this method.

  You can re-run an older run by providing the run_id of the older run in
  --use-cached. Ensure that the catalogs and run logs are accessible by the
  present configuration.

  When running map states, ensure that the parameter to iterate on is
  available in parameter space.

Options:
  -f, --file TEXT                 The pipeline definition file  [default:
                                  pipeline.yaml]
  -v, --var-file TEXT             Variables used in pipeline definition
  -c, --config-file TEXT          config file, in yaml, to be used for the run
  -p, --parameters-file TEXT      Parameters, in yaml,  accessible by the
                                  application
  --log-level [INFO|DEBUG|WARNING|ERROR|FATAL]
                                  The log level  [default: WARNING]
  --tag TEXT                      A tag attached to the run
  --run-id TEXT                   An optional run_id, one would be generated
                                  if not provided
  --use-cached TEXT               Provide the previous run_id to re-run.
  --help                          Show this message and exit.
```

The options have the same meaning as executing a pipeline.

**Design thought:** This method could be handy to debug a single node of the pipeline or run a single step of the pipeline
in other environments by changing the config.

## Extensions

Magnus internally uses click to perform CLI operations and base command is given below.

```python
@with_plugins(iter_entry_points('magnus.cli_plugins'))
@click.group()
@click.version_option()
def cli():
    """
    Welcome to magnus. Please provide the command that you want to use.
    All commands have options that you can see by magnus <command> --help
    """
    pass

```

You can provide custom extensions to the command line capabilities by extending the namespace ```magnus.cli_plugins```

```toml
# For example, as part of your pyproject.toml
[tool.poetry.plugins."magnus.cli_plugins"]
"aws-ecr = "YOUR_PACKAGE:push_to_ecr"
```

This extension than can be used as

```magnus aws-ecr <parameters and options>```
