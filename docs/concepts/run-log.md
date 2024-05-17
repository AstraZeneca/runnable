```runnable``` makes reproducibility easy without any intervention from the developer.

Execution of the pipeline, in any environment, generates a ```run log``` which stores meta information of the
execution.

Executions also stores the ```stdout``` and ```stderr```
of the task at the log level used during the run.


## Code identity

For non-container based executions, the ```git``` sha is captured along with
other necessary attributes.

### Example

```json linenums="1"
"steps": {
    "step_name": {
    ...
    "code_identities": [
        {
            "code_identifier": "259904152753ccb326ab71804ac6b2f343ee6182",
            "code_identifier_type": "git",
            "code_identifier_dependable": false, // (1)
            "code_identifier_url": "https://github.com/AstraZeneca/runnable.git",
            "code_identifier_message": "changes found in docs/concepts/run-log.md" // (2)
        }
    }
}
```

1. Implies that the branch is not clean.
2. Emits all the files that are different from the HEAD.


For container based executions, the container digest and name is captured.

## Parameters

The input and output parameters at the point of execution of all the tasks is captured.

### Example

```json linenums="1"
"steps": {
    "step_name":{
        ...
        "attempts":[
            {
                ...
                "input_parameters": {
                    "X": {
                        "kind": "object",
                        "value": "X",
                        "reduced": true,
                        "description": "Pickled object stored in catalog as: X"
                    },
                    "Y": {
                        "kind": "object",
                        "value": "Y",
                        "reduced": true,
                        "description": "Pickled object stored in catalog as: Y"
                    }
                },
                "output_parameters": {
                    "logreg": {
                        "kind": "object",
                        "value": "logreg",
                        "reduced": true,
                        "description": "Pickled object stored in catalog as: logreg"
                    }
                }
            }
        ]
    }

```

## Metrics

Any parameters marked as ```metrics``` are stored too.


### Example

```json linenums="1"
"steps": {
    "step_name":{
        ...
        "attempts":[
            {
                ...
                "user_defined_metrics": {
                    "score": {
                        "kind": "metric",
                        "value": 0.6,
                        "reduced": true,
                        "description": 0.6
                    }
                }
            }
        ]
    }

```

## Data

Any ```files``` moved between tasks are stored in the ```catalog``` along with
meta information stored in the ```run log```.

### Example

```json linenums="1"
"steps": {
    "step_name":{
        ...
        "data_catalog": [
            {
                "name": "iris_logistic.png",
                "data_hash": "1a119ee3496f72d7cdd379b658aa79dc0eee38923d270ef7adf61dcb8f033f06",
                "catalog_relative_path": "best-hamilton-0300/iris_logistic.png",
                "catalog_handler_location": ".catalog",
                "stage": "put"
            }
        ]
    }
}

```


## Retrying failures

The structure of the run log remains the same independent of the ```executor``` used to execute.
This enables to debug failures during the execution in complex environments to be easily
reproduced in local environments and fixed.

Make the ```catalog``` and ```run log``` generated during the failed execution
accessible to the ```retry``` executor and the execution starts from the failed
step.

Refer to [retry](../configurations/executors/retry.md) for more information.
