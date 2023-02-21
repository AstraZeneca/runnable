# Overview

Secrets are essential in making your data science projects secure and collaborative. They could be database credentials,
API keys or any information that need to present at the run-time but invisible at all other times.
Magnus provides a clean interface to access/store secrets and independent of the actual secret provider,
the interface remains the same.

As with all modules of magnus, there are many secrets providers and if none fit your needs, it is easier to write
one of your to fit your needs. In magnus, all secrets are key value pairs.

## Configuration

Configuration of a Secrets is as follows:

```yaml
secrets:
  type:
  config:
```

### type

The type of secrets provider you want. This should be one of the secrets types already available.

There is no default secrets provider.

### config

Any configuration parameters the secret provider accepts.


## Interaction with other services

Other service providers, like run log store or catalog, can access the secrets by using the
```global_executor.secrets_handler``` of ```pipeline``` module during the run time. This could be useful for
constructing connection strings to database or AWS connections.

For example:

```python

class CustomObject:

    @property
    def connection_object(self):
        from magnus.pipeline import global_executor
        secrets = global_exector.secrets_handler.get_secrets()
        # Do something with the secrets

```

## Interaction within code

Secrets is the only implementation that requires you to ```import magnus``` in the code to access secrets.
This is mostly to follow the best safety guidelines.

Once a secret configuration is defined as above, you can access the secret by using ```get_secret``` of magnus.
If a key is provided to the API, we return only the value associated with the secret by the key.
If a key is not provided, we return all the key value secret pairs provided.
The API would raise an exception if a secret by the key requested does not exist.

Currently, there is no provision to update/edit secrets via code.


For example if the secret key-value pairs are:

```yaml
secret_answer: 42
secret_question: everything
```

And for the code:
```python
# In my_module.py
from magnus import get_secret

def my_cool_function():

    secret = get_secret('secret_answer')

    all_secrets = get_secret()

```

secret would have a value of ```42``` while all_secrets would be a dictionary
```{'secret_answer': 42, 'secret_question': 'everything'}```


## Parameterized definition

As with any part of the magnus configuration, you can parameterize the configuration of secrets to switch between
providers without changing the base definition.

Please follow the example provided [here](../dag/#parameterized_definition) for more information.


## Extensions

You can easily extend magnus to bring in your custom provider, if a default
implementation does not exist or you are not happy with the implementation.

To implement your custom secret class, please extend BaseSecret class of magnus whose definition is given below.

```python
from pydantic import BaseModel

class BaseSecrets:
    """
    A base class for Secrets Handler.
    All implementations should extend this class.

    Note: As a general guideline, do not extract anything from the config to set class level attributes.
          Integration patterns modify the config after init to change behaviors.
          Access config properties using getters/property of the class.

    Raises:
        NotImplementedError: Base class and not implemented
    """
    service_name = ''

    class Config(BaseModel):
        pass

    def __init__(self, config: dict, **kwargs):
        config = config or {}
        self.config = self.Config(**config)

    def get(self, name: str = None, **kwargs) -> Union[str, dict]:
        """
        Return the secret by name.
        If no name is give, return all the secrets.

        Args:
            name (str): The name of the secret to return.

        Raises:
            NotImplementedError: Base class and hence not implemented.
        """
        raise NotImplementedError
```

The custom extensions should be registered as part of the namespace: ```magnus.secrets.BaseSecrets``` for it to be
loaded.

```toml
# For example, as part of your pyproject.toml
[tool.poetry.plugins."magnus.secrets.BaseSecrets"]
"k8s-secrets" = "YOUR_PACKAGE:K8sSecret"
```
