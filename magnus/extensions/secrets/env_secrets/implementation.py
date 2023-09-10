import logging
import os
from typing import Union

from magnus import BaseSecrets, defaults, exceptions

logger = logging.getLogger(defaults.LOGGER_NAME)


class EnvSecretsManager(BaseSecrets):
    """
    A secret manager via environment variables.

    This secret manager returns nothing if the key does not match
    """

    service_name: str = "env-secrets-manager"
    prefix: str = ""
    suffix: str = ""

    def get(self, name: str = None, **kwargs) -> Union[str, dict]:
        """
        If a name is provided, we look for that in the environment.
        If a environment variable by that name is not found, we raise an Exception.

        If a name is not provided, we return an empty dictionary.

        Args:
            name (str): The name of the secret to retrieve

        Raises:
            Exception: If the secret by the name is not found.

        Returns:
            [type]: [description]
        """
        if name:
            try:
                return os.environ[f"{self.prefix}{name}{self.suffix}"]
            except KeyError as _e:
                logger.exception(f"Secret {self.prefix}{name}{self.suffix} not found in environment")
                raise exceptions.SecretNotFoundError(secret_name=name, secret_setting="environment") from _e

        matched_secrets = {}
        for key in os.environ:
            if key.startswith(self.prefix) and key.endswith(self.suffix):
                matched_secrets[key[len(self.prefix) : -len(self.suffix)]] = os.environ[key]

        return matched_secrets
