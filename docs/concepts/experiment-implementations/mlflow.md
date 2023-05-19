# MLFlow

[MLFlow](https://mlflow.org/docs/latest/index.html) is a popular experiment tracking tool.
We provide an integration of magnus with MLFlow.

## Features not yet implemented

Currently the only way to pass in the credentials for the MLFlow is via environment variables. It should be possible to
provide the credentials as magnus secrets and use it in the future.

## Additional dependencies

Magnus extensions needs mlflow capabilities for this. You can install it via

```pip install "magnus_extensions[mlflow]"```

or via:

```poetry add "magnus_extensions[mlflow]"```


## Configuration

The full configuration to use mlflow as experiment tracker:

```
experiment_tracker:
  type: mlflow
  config:
    server_url: str
    autolog: False
```

!!! Warning

Auto log features might need additional dependencies that we do not install. Please install the required.

### server_url:

This is a required parameter for the experiment tracker. This should be the URL of MLFlow server.

### autolog:

To enable auto logging features of mlflow.

## How does it work?

Detailed explanation of how
[experiment tracking behaves is documented here](https://astrazeneca.github.io/magnus-core/concepts/experiment-tracking/).

Any parameters that is tracked with magnus via ```track_this``` would be logged into the run log store of magnus and
also sent to ```set_metric``` of mlflow client. Since the ```set_metric``` expects only numeric values in tracking,
the tracking parameter is sent to ```log_param``` of mlflow client if it is identified to be non numeric.

### Client context

You can also get the client context of mlflow client by calling ```get_experiment_tracker_context``` method of magnus.

For example:

```python

from magnus import get_experiment_tracker_context


with context as get_experiment_tracker_context():
    # Do anything with mlflow client

```
