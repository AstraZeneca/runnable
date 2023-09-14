import logging
from abc import ABC, abstractmethod
from typing import Union

from pydantic import BaseModel, ConfigDict

import magnus.context as context
from magnus import defaults

logger = logging.getLogger(defaults.LOGGER_NAME)


# --8<-- [start:docs]
class BaseSecrets(ABC, BaseModel):
    """
    A base class for Secrets Handler.
    All implementations should extend this class.

    Note: As a general guideline, do not extract anything from the config to set class level attributes.
          Integration patterns modify the config after init to change behaviors.
          Access config properties using getters/property of the class.

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


# --8<-- [end:docs]


class DoNothingSecretManager(BaseSecrets):
    """
    Does nothing secret manager
    """

    service_name: str = "do-nothing"

    def get(self, name: str = None, **kwargs) -> Union[str, dict]:
        """
        If a name is provided, return None else return empty dict.

        Args:
            name (str): The name of the secret to retrieve

        Raises:
            Exception: If the secret by the name is not found.

        Returns:
            [type]: [description]
        """
        if name:
            return ""
        return {}
