import logging

from dotenv import dotenv_values

from runnable import defaults, exceptions
from runnable.secrets import BaseSecrets

logger = logging.getLogger(defaults.LOGGER_NAME)


class DotEnvSecrets(BaseSecrets):
    """
    A secret manager which uses .env files for secrets.

    We recommend this secrets manager only for local development and should not be used for anything close to
    production.
    """

    service_name: str = "dotenv"
    location: str = defaults.DOTENV_FILE_LOCATION
    secrets: dict = {}

    @property
    def secrets_location(self):
        """
        Return the location of the .env file.
        If the user has not over-ridden it, it defaults to .env file in the project root.

        Returns:
            str: The location of the secrets file
        """
        return self.location

    def _load_secrets(self):
        """
        Use dotenv to load the secrets
        """
        self.secrets = dotenv_values(self.secrets_location)

    def get(self, name: str = "", **kwargs) -> str:
        """
        Get a secret of name from the secrets file.


        Args:
            name (str): The name of the secret to retrieve

        Raises:
            Exception: If the secret by the name is not found.

        Returns:
            str: The value of the secret
        """
        if not self.secrets:
            self._load_secrets()

        if name in self.secrets:
            return self.secrets[name]

        raise exceptions.SecretNotFoundError(
            secret_name=name, secret_setting=self.secrets_location
        )
