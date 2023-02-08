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


### Configurations file

The yaml file containing the configurations used to run magnus. The configurations provided here would over-ride any
configuration variables.

Provided by ```-c```, ```--config-file``` option on magnus cli.

Defaults to None, if nothing is provided.
Read more about different ways you can configure magnus runs here.



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

## Executing a Jupyter notebook

This method could be used to run a Jupyter notebook in any environment.

The complete options are:

```

Usage: magnus execute_notebook [OPTIONS] FILENAME

  Entry point to execute a Jupyter notebook in isolation.

  The notebook would be executed in the environment defined by the config file or default if none.

Options:
  -c, --config-file TEXT          config file, in yaml, to be used for the run
  -p, --parameters-file TEXT      Parameters, in yaml,  accessible by the
                                  application
  --log-level [INFO|DEBUG|WARNING|ERROR|FATAL]
                                  The log level  [default: WARNING]
  -d, --data-folder TEXT          The catalog data folder
  -put, --put-in-catalog TEXT     The data to put from the catalog
  --tag TEXT                      A tag attached to the run
  --run-id TEXT                   An optional run_id, one would be generated
                                  if not provided
  --help                          Show this message and exit.

```

## Executing a python function

This method could be used to run a python function in any environment.

The complete options are:

```
Usage: magnus execute_function [OPTIONS] COMMAND

  Entry point to execute a python function in isolation.

  The function would be executed in the environment defined by the config file
  or default if none.

Options:
  -c, --config-file TEXT          config file, in yaml, to be used for the run
  -p, --parameters-file TEXT      Parameters, in yaml,  accessible by the
                                  application
  --log-level [INFO|DEBUG|WARNING|ERROR|FATAL]
                                  The log level  [default: WARNING]
  -d, --data-folder TEXT          The catalog data folder
  -put, --put-in-catalog TEXT     The data to put from the catalog
  --tag TEXT                      A tag attached to the run
  --run-id TEXT                   An optional run_id, one would be generated
                                  if not provided
  --help                          Show this message and exit.
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
  -c, --config-file TEXT          config file, in yaml, to be used for the run
  -p, --parameters-file TEXT      Parameters, in yaml,  accessible by the
                                  application
  --log-level [INFO|DEBUG|WARNING|ERROR|FATAL]
                                  The log level  [default: WARNING]
  --tag TEXT                      A tag attached to the run
  --run-id TEXT                   An optional run_id, one would be generated
                                  if not provided
  --use-cached TEXT               Provide the previous run_id to re-run.
  --help                          Show this message and exit.```

```

The options have the same meaning as executing a pipeline.

**Design thought:** This method could be handy to debug a single node of the pipeline or run a single step of the pipeline
in other environments by changing the config.


## Building docker images

This method is a utility tool to assist in building docker images.

It is preferred that you have a docker file that you can provide to the utility tool using the ```-f/--docker-file```
option. We can auto-generate a opinionated dockerfile but it is unlikely to fit your needs perfectly.

For the auto-generation of the dockerfile:

- You can provide the style of dependency management. Currently, poetry, pipenv are supported. Any other would revert
to using requirements.txt dependency style.
- The base image is python 3.7
- By default, we add only git tracked contents into the ```app``` folder of the image. But you can over-ride it
with ```--all``` option to add all content to the image.

Please be aware that using ```--all``` might add sensitive data into the docker image.

The options available are:

```
Usage: magnus build_docker [OPTIONS] IMAGE_NAME

  A utility function to create docker images from the existing codebase.

  It is advised to provide your own dockerfile as much as possible. If you do
  not have one handy, you can use --dry-run functionality to see if the auto-
  generated one suits your needs.

  If you are auto-generating the dockerfile: BEWARE!! Over-riding the default
  options assumes you know what you are doing! BEWARE!!

  1). By default, only git tracked files are added to the docker image.

  2). The auto-generated dockerfile uses, python 3.7 as the default image and
  adds the current folder.

Options:
  -f, --docker-file TEXT  The dockerfile to be used. If None, we generate one
  -s, --style TEXT        The method used to get requirements  [default:
                          poetry]
  -t, --tag TEXT          The tag assigned to the image  [default: latest]
  -c, --commit-tag        Use commit id as tag. Over-rides tag option
                          [default: False]
  -d, --dry-run           Generate the dockerfile, but NOT the image
                          [default: False]
  --git-tracked / --all   Controls what should be added to image. All vs git-
                          tracked  [default: git-tracked]
  --help                  Show this message and exit.
```



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
