# Command line options

You can execute a pipeline by the following command:

```shell
magnus execute
```

---
!!! Note

   For the above command to work, make sure you are in the environment where magnus was installed.

   If you are using poetry, you can also invoke magnus by ```poetry run magnus execute```

---

## Dag definition/config

The file containing the dag definition and the config to be used.

Provided by ```-f```, ```--file``` option on magnus cli.

Defaults to ```pipeline.yaml``` if nothing is provided.

## Variables file

The yaml file containing the variables or placeholder values in the dag definition file.

Provided by ```-v```, ```--var-file``` option on magnus cli.

Defaults to None, if nothing is provided.
[Read more about parameterized definitions](../../concepts/dag/#parameterized_definition).


## Configurations file

The yaml file containing the configurations used to run magnus. The configurations provided here would over-ride any
configuration variables.

Provided by ```-c```, ```--config-file``` option on magnus cli.

Defaults to None, if nothing is provided.
Read more about different ways you can configure magnus runs here.


## Log level

To control the logging level of magnus only. This does not affect your application logs what so ever.

Provided by ```--log-level``` option on magnus cli.

Available options are: DEBUG, INFO, WARNING, ERROR, CRITICAL.

Defaults to INFO if nothing is provided.

## Tag

A friendly way to tag experiments or runs together.

Provided by ```--tag``` option on magnus cli.

Defaults to None if nothing is provided.

## Run id

An unique run identifier for the run.

Provided by ```--run-id``` on magnus cli.

We generate one based on Timestamp if one is not provided.


## Use cached

Enables you to re-run a previous run provided by the run-id.

Example:

```shell
magnus execute --file example.yaml --run-id 20210506051758 --use-cached old_run_id
```
