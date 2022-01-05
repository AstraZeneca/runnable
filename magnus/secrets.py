import logging
from functools import lru_cache

from magnus import defaults
from magnus import utils
from magnus import exceptions

logger = logging.getLogger(defaults.NAME)


class BaseSecrets:
    """
    A base class for Secrets Handler.
    All implementations should extend this class.

    Do not use __init__ in the chlid class to store configuration parameters,
    as a general design guideline and use get_from_config methods to get values as required.

    This would ensure that we can change the config when appropriate.

    Raises:
        NotImplementedError: Base class and not implemented
    """
    service_name = None

    def __init__(self, config: dict, **kwargs):  # pylint: disable=unused-argument
        self.config = config or {}

    def get(self, name: str = None, **kwargs):
        """
        Return the secret by name.
        If no name is give, return all the secrets.

        Args:
            name (str): The name of the secret to return.

        Raises:
            NotImplementedError: Base class and hence not implemented.
        """
        raise NotImplementedError


def get_secrets_handler(secrets_config: dict):
    """
    Given a config, return the appropriate secrets handler

    Args:
        secrets_config (dict): The config as per the definition

    Raises:
        Exception: If no secrets handler of the type could be found in implementations

    Returns:
       BaseSecrets : The secrets handler as per the dag definiont
    """
    if secrets_config:
        secrets_type = secrets_config.get('type', None)
        if not secrets_type:
            raise Exception('Please provide a secrets type')

        for sub_class in BaseSecrets.__subclasses__():
            if secrets_type == sub_class.service_name:
                return sub_class(config=secrets_config.get('config', {}))

    logger.info('No secrets handler found')
    return DummySecretManager(config={})


class DummySecretManager(BaseSecrets):
    """
    Does nothing secret manager
    """

    service_name = 'dummy'

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.secrets = {}

    def get(self, name: str = None, **kwargs) -> dict:
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
        if name:
            return None
        return {}


class DotEnvSecrets(BaseSecrets):
    """
    A secret manager which uses .env files for secrets.

    We recommend this secrets manager only for local development and should not be used for anything close to
    production.
    """

    service_name = 'dotenv'

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.secrets = {}

    def get_secrets_location(self):
        """
        Return the location of the .env file.
        If the user has not over-ridden it, it defaults to .env file in the project root.

        Returns:
            str: The location of the secrets file
        """
        secrets_location = defaults.DOTENV_FILE_LOCATION
        if self.config and 'location' in self.config:
            secrets_location = self.config['location']

        return secrets_location

    def load_secrets(self):
        """
        We assume that a dotenv file is of format,
            key=value  -> secrets[key]='value'
            key1=value1# comment  -> secrets[key1]='value1'
            key2=value2 # comment. -> secrets[key2]='value2 '

        The key/value pair is read as present in the file with spaces included.
        Please be mindful of spaces.

        Raises:
            Exception: If the file at secrets_location is not found.
            Exception: If the secrets are not formatted correctly.
        """
        secrets_location = self.get_secrets_location()
        if not utils.does_file_exist(secrets_location):
            raise Exception(f'Did not find the secrets file in {secrets_location}')

        with open(secrets_location, 'r') as fr:
            for secret_line in fr:
                secret_line = secret_line.split('#')[0]  #  To remove any comments the user might have put
                data = secret_line.split('=')
                if len(data) != 2:
                    raise Exception('A secret should be of format, secret_name=secret_value[# any comment]')
                key, value = data
                self.secrets[key] = value.strip('\n')

    def get(self, name: str = None, **kwargs) -> dict:
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
        self.load_secrets()
        if not name:
            return self.secrets

        if name in self.secrets:
            return self.secrets[name]

        secrets_location = self.get_secrets_location()
        raise exceptions.SecretNotFoundError(secret_name=name, secret_setting=secrets_location)
