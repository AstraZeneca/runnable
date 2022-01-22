To extend and implement a custom secrets handler, you need to over-ride the appropriate methods of the ```Base``` class.


Please refer to [*Guide to extensions* ](../../../extensions/extensions/) for a detailed explanation and the need for
implementing a *Integration* pattern along with the extension.

Extensions that are being actively worked on and listed to be released as part of ```magnus-extensions```

- aws-secrets-manager : Using aws-secrets-manager as secret store.

```python
# You can find this in the source code at: magnus/secrets.py along with a few example
# implementations of do-nothing and dotenv
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

    def __init__(self, config: dict, **kwargs):  # pylint: disable=unused-argument
        self.config = config or {}

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
