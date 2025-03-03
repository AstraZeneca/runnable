# Overview

!!! note "Opt out"

    Pipelines need not use the ```secrets``` if the preferred tools of choice is
    not implemented in runnable. The default configuration of ```do-nothing``` is no-op by design.
    We kindly request to raise a feature request to make us aware of the eco-system.


Most complex pipelines require secrets to hold sensitive information during task execution.
They could be database credentials, API keys or any information that need to present at
the run-time but invisible at all other times.

The secrets are always exposed as environmental variables.

A typical example would be a task requiring the database connection string to connect
to a database.


```python title="Using the secrets API"

class CustomObject:

    @property
    def connection_object(self):
        import os
        connection_string = os.environ.get("connection_string")
        # Do something with the secrets
```

Please refer to [configurations](../configurations/secrets.md) for available implementations.

## Example

=== "dotenv format"

    The dotenv format for providing secrets. Ideally, this file should not be part of the
    version control but present during development phase.

    The file is assumed to be present in ```examples/secrets.env``` for this example.

    It follows the same format as [python-dotenv](https://github.com/theskumar/python-dotenv)


=== "Example configuration"

    Configuration to use the dotenv format file.

    ```yaml linenums="1"
    --8<-- "examples/configs/dotenv.yaml"
    ```

    1. Use dotenv secrets manager.
    2. Location of the dotenv file, defaults to ```.env``` in project root.
