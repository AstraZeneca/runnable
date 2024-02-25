**runnable** provides an interface to secrets managers
[via the API](../interactions.md/#runnable.get_secret).

Please refer to [Secrets in concepts](../concepts/secrets.md) for more information.

## do-nothing

A no-op implementation of a secret manager. This is useful when you do not have need for
secrets in your application.

### configuration

```yaml
secrets:
  type: do-nothing

```

Note that this is the default configuration if nothing is specified.


<hr style="border:2px dotted orange">

## Environment Secret Manager

A secrets manager to access secrets from environment variables. Many cloud based executors, especially
K8's, have capabilities to send in secrets as environment variables and this secrets provider could
used in those environments.

### Configuration

```yaml
secrets:
  type: env-secrets-manager
  config:
    prefix: "" # default value
    suffix: "" # default value
```

Use ```suffix``` and ```prefix``` the uniquely identify the secrets.
The actual key while calling the secrets manager via the API, ```get_secret(secret_key)``` is
```<prefix><secret_key><suffix>```.

### Example


=== "Pipeline"

    Below is a simple pipeline to demonstrate the use of secrets.

    The configuration file to use can be dynamically specified via the environment variable
    ```runnable_CONFIGURATION_FILE```.

    The example can be found in ```examples/secrets_env.py```

    ```python
    --8<-- "examples/secrets_env.py"
    ```

=== "Default Configuration"

    We can execute the pipeline using this configuration by:
    ```secret="secret_value" runnable_CONFIGURATION_FILE=examples/configs/secrets-env-default.yaml python examples/secrets_env.py```

    The configuration file is located at ```examples/configs/secrets-env-default.yaml```

    ```yaml
    --8<-- "examples/configs/secrets-env-default.yaml"
    ```

=== "Prefixed and Suffixed Configuration"

    We can execute the pipeline using this configuration by:
    ```runnable_secret="secret_value" runnable_CONFIGURATION_FILE=examples/configs/secrets-env-ps.yaml python examples/secrets_env.py```

    The configuration file is located at ```examples/configs/secrets-env-ps.yaml```

    ```yaml
    --8<-- "examples/configs/secrets-env-ps.yaml"
    ```

<hr style="border:2px dotted orange">

## dotenv

```.env``` files are routinely used to provide configuration parameters and secrets during development phase. runnable can dotenv files as a secret store and can surface them to tasks.


### Configuration


```yaml
secrets:
  type: dotenv
  config:
    location: .env # default value

```

The format of the ```.env``` file is ```key=value``` pairs. Any content after ```#``` is considered
as a comment and will be ignored. Using ```export``` or ```set```, case insensitive, as used
for shell scripts are allowed.

### Example

=== ".env file"

    Assumed to be present at ```examples/secrets.env```

    ```shell linenums="1"
    --8<-- "examples/secrets.env"
    ```

    1. Shell scripts style are supported.
    2. Key value based format is also supported.


=== "Example configuration"

    Configuration to use the dotenv format file.

    Assumed to be present at ```examples/configs/dotenv.yaml```

    ```yaml linenums="1"
    --8<-- "examples/configs/dotenv.yaml"
    ```

    1. Use dotenv secrets manager.
    2. Location of the dotenv file, defaults to ```.env``` in project root.


=== "Pipeline in python"

    The example is present in ```examples/secrets.py```

    ```python linenums="1" hl_lines="12-13"
    --8<-- "examples/secrets.py"
    ```

    1. The key of the secret that you want to retrieve.
