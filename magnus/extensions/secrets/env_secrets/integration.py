import logging

from magnus import defaults
from magnus.integration import BaseIntegration

logger = logging.getLogger(defaults.LOGGER_NAME)


class LocalContainerComputeEnvSecretsManager(BaseIntegration):
    """
    Integration between local container and env secrets manager
    """

    executor_type = "local-container"
    service_type = "secrets"  # One of secret, catalog, datastore
    service_provider = "env-secrets-manager"  # The actual implementation of the service

    def validate(self, **kwargs):
        msg = (
            "Local container executions cannot be used with environment secrets manager. "
            "Please use a supported secrets manager"
        )
        logger.exception(msg)
        raise Exception(msg)
