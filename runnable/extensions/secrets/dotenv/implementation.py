import logging
import os

from runnable import defaults, exceptions, utils
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
        We assume that a dotenv file is of format,
            key=value  -> secrets[key]='value'
            key=value# comment  -> secrets[key1]='value1'
            key=value2 # comment. -> secrets[key2]='value2'

        Any of the above formats with export or set in front of them.

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
                # The order of removing fluff around the expression
                # the new line
                # the comment
                # the white space
                # Any export or set in front of the key any spaces after that.

                secret_line = secret_line.strip(os.linesep).split("#")[0].strip()

                if secret_line == "":
                    continue

                secret_line = utils.remove_prefix(secret_line, prefix="export").strip()
                secret_line = utils.remove_prefix(secret_line, prefix="EXPORT").strip()
                secret_line = utils.remove_prefix(secret_line, prefix="set").strip()
                secret_line = utils.remove_prefix(secret_line, prefix="SET").strip()

                data = secret_line.split("=")
                if len(data) != 2:
                    raise Exception("A secret should be of format, secret_name=secret_value[# any comment]")

                key, value = data
                self.secrets[key] = value.strip().strip('"').strip(os.linesep)

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
        self._load_secrets()

        if name in self.secrets:
            return self.secrets[name]

        raise exceptions.SecretNotFoundError(secret_name=name, secret_setting=self.secrets_location)
