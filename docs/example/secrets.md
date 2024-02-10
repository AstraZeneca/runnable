Secrets are required assets as the complexity of the application increases. Magnus provides a
python API to get secrets from various sources.

!!! info annotate inline end "from magnus import get_secret"

    Secrets is the only interface that you are required to "import magnus" in your python application.

    Native python and Jupyter notebooks can use this API. We currently do not support shell tasks with
    secrets from this interface. (1)

1. Using environment variables to access secrets is one pattern works in all environments.

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
