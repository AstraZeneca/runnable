# Command line options

You can execute a pipeline by the following command:

```shell
magnus execute
```

## Dag definition/config

The file containing the dag definition and the config to be used. 

Provided by ```-f```, ```--file``` option on magnus cli.

Defaults to ```pipeline.yaml``` if nothing is provided.

## Variables file

The yaml file containing the variables or placeholder values in the dag definition file. 

Provided by ```-v```, ```--var-file``` option on magnus cli. 

Defaults to None, if nothing is provided. 
[Read more about parameterised defintions](../../concepts/dag/#parameterized_definition).

## Log level

To control the logging level of magnus only. This does not affect your application logs what so ever. 

Provided by ```--log-level``` option on magnus cli. 

Available options are: DEBUG, INFO, WARNING, ERROR, CRITICAL.

Defaults to INFO if nothing is provided. 

## Tag

A friendly way to tag exeperiments or runs together. 

Provided by ```--tag``` option on magnus cli. 

Defaults to None if nothing is provided. 

## Run id

An unique run identifier for the run. 

Provided by ```--run-id``` on magnus cli.

We generate one based on Timestamp if one is not provided. 

Run Id has a nuance around it in magnus, please [read more here](../../concepts/run-log/#run_id). 

## Use cached

Enables you to re-run a previous run provided by the run-id.

This run would have its own unique run_id as explained here. 

Example:

```shell
magnus execute --file example.yaml --run-id 20210506051758_51b665 --use-cached
```

## Use cached force

Magnus does a check on the dag definition before attempting a re-run and it will fail if the dag definitions
are not exactly the same. You can force a re-run by using this option. 

Example:

```shell
magnus execute --file example.yaml --run-id 20210506051758_51b665 --use-cached-force
```