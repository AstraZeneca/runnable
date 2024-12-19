import logging
import os
from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict

import runnable.context as context
from runnable import defaults, exceptions

logger = logging.getLogger(defaults.LOGGER_NAME)


# --8<-- [start:docs]
class BaseSecrets(ABC, BaseModel):
    """
    A base class for Secrets Handler.
    All implementations should extend this class.

    Raises:
        NotImplementedError: Base class and not implemented
    """

    service_name: str = ""
    service_type: str = "secrets"
    model_config = ConfigDict(extra="forbid")

    @property
    def _context(self):
        return context.run_context

    @abstractmethod
    def get(self, name: str, **kwargs) -> str:
        """
        Return the secret by name.

        Args:
            name (str): The name of the secret to return.

        Raises:
            NotImplementedError: Base class and hence not implemented.
            exceptions.SecretNotFoundError: Secret not found in the secrets manager.
        """
        raise NotImplementedError


# --8<-- [end:docs]


class DoNothingSecretManager(BaseSecrets):
    """
    Does nothing secret manager
    """

    service_name: str = "do-nothing"

    def get(self, name: str, **kwargs) -> str:
        """
        If a name is provided, return None else return empty dict.

        Args:
            name (str): The name of the secret to retrieve

        Raises:
            exceptions.SecretNotFoundError: Secret not found in the secrets manager.

        Returns:
            [str]: The value of the secret
        """
        return ""


class EnvSecretsManager(BaseSecrets):
    """
    A secret manager which uses environment variables for secrets.
    """

    service_name: str = "env-secrets"

    def get(self, name: str, **kwargs) -> str:
        """
        If a name is provided, return None else return empty dict.

        Args:
            name (str): The name of the secret to retrieve

        Raises:
            exceptions.SecretNotFoundError: Secret not found in the secrets manager.

        Returns:
            [str]: The value of the secret
        """
        try:
            return os.environ[name]
        except KeyError:
            raise exceptions.SecretNotFoundError(
                secret_name=name, secret_setting="environment variables"
            )
