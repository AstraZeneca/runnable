# Overview

[Run log](../run-log) stores a lot of information about the execution along with the metrics captured
during the execution of the pipeline.


## Example


=== "Using the API"

    The highlighted lines in the below example show how to [use the API](../../interactions/#magnus.track_this)

    Any pydantic model as a value would be dumped as a dict, respecting the alias, before tracking it.

    ```python linenums="1" hl_lines="3 17-19"
    --8<-- "examples/concepts/experiment_tracking_api.py"
    ```


=== "Using environment variables"

    The highlighted lines in the below example show how to use environment variables to track metrics.

    Only string values are allowed to be environment variables. Numeric values sent in as strings are converted
    to int/float before storing them as metrics.

    There is no support for boolean values in environment variables.

    ```yaml linenums="1" hl_lines="16-18"
    --8<-- "examples/concepts/experiment_tracking_env.yaml"
    ```


### Run log entry


## Step parameter


### Example


## Client context


### Example


## Grouping experiments

## Experiment tracking tools
