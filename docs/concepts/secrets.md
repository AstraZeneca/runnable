# Overview

!!! note "Opt out"

    Pipelines need not use the ```secrets``` if the preferred tools of choice is
    not implemented in runnable. The default configuration of ```do-nothing``` is no-op by design.
    We kindly request to raise a feature request to make us aware of the eco-system.


Most complex pipelines require secrets to hold sensitive information during task execution.
They could be database credentials, API keys or any information that need to present at
the run-time but invisible at all other times.

runnable provides a [clean API](../interactions.md/#runnable.get_secret) to access secrets
and independent of the actual secret provider, the interface remains the same.

A typical example would be a task requiring the database connection string to connect
to a database.


```python title="Using the secrets API"

class CustomObject:

    @property
    def connection_object(self):
        from runnable import get_secret
        connection_string = get_secret("connection_string")
        # Do something with the secrets
```

Please refer to [configurations](../configurations/secrets.md) for available implementations.

## Example

=== "dotenv format"

    The dotenv format for providing secrets. Ideally, this file should not be part of the
    version control but present during development phase.

    The file is assumed to be present in ```examples/secrets.env``` for this example.

    ```shell linenums="1"
    --8<-- "examples/secrets.env"
    ```

    1. Shell scripts style are supported.
    2. Key value based format is also supported.


=== "Example configuration"

    Configuration to use the dotenv format file.

    ```yaml linenums="1"
    --8<-- "examples/configs/dotenv.yaml"
    ```

    1. Use dotenv secrets manager.
    2. Location of the dotenv file, defaults to ```.env``` in project root.


=== "Pipeline in python"

    ```python linenums="1" hl_lines="12-13"
    --8<-- "examples/secrets.py"
    ```

    1. The key of the secret that you want to retrieve.
