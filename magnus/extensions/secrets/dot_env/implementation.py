import logging
from typing import Union

from magnus import defaults, exceptions, utils
from magnus.secrets import BaseSecrets

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
        We assume that a dotenv file is of format,
            key=value  -> secrets[key]='value'
            key1=value1# comment  -> secrets[key1]='value1'
            key2=value2 # comment. -> secrets[key2]='value2'

        We strip the secret value of any empty spaces at the start and end.

        Raises:
            Exception: If the file at secrets_location is not found.
            Exception: If the secrets are not formatted correctly.
        """
        # It was loaded in the previous call and need not to be reloaded
        if self.secrets:
            return

        secrets_location = self.secrets_location
        if not utils.does_file_exist(secrets_location):
            raise Exception(f"Did not find the secrets file in {secrets_location}")

        with open(secrets_location, "r") as fr:
            for secret_line in fr:
                secret_line = secret_line.split("#")[0]  #  To remove any comments the user might have put
                data = secret_line.split("=")
                if len(data) != 2:
                    raise Exception("A secret should be of format, secret_name=secret_value[# any comment]")
                key, value = data
                self.secrets[key] = value.strip("\n")

    def get(self, name: str = "", **kwargs) -> Union[str, dict]:
        """
        Get a secret of name from the secrets file.

        If no name is provided, we return all

        Args:
            name (str): The name of the secret to retrieve

        Raises:
            Exception: If the secret by the name is not found.

        Returns:
            [type]: [description]
        """
        self._load_secrets()
        if not name:
            return self.secrets

        if name in self.secrets:
            return self.secrets[name]

        secrets_location = self.secrets_location
        raise exceptions.SecretNotFoundError(secret_name=name, secret_setting=secrets_location)
