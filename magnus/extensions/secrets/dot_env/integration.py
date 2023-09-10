import logging
from pathlib import Path
from typing import cast

from magnus import defaults
from magnus.extensions.executor.local_container.implementation import LocalContainerExecutor
from magnus.integration import BaseIntegration

from .implementation import DotEnvSecrets

logger = logging.getLogger(defaults.LOGGER_NAME)


class LocalContainerComputeDotEnvSecrets(BaseIntegration):
    """
    Integration between local container and dot env secrets
    """

    executor_type = "local-container"
    service_type = "secrets"  # One of secret, catalog, datastore
    service_provider = "dotenv"  # The actual implementation of the service

    def validate(self, **kwargs):
        logger.warning("Using dot env for non local deployments is not ideal, consider options")

    def configure_for_traversal(self, **kwargs):
        self.executor = cast(LocalContainerExecutor, self.executor)
        self.service = cast(DotEnvSecrets, self.service)

        secrets_location = self.service.secrets_location
        self.executor._volumes[str(Path(secrets_location).resolve())] = {
            "bind": f"{self.executor._container_secrets_location}",
            "mode": "ro",
        }

    def configure_for_execution(self, **kwargs):
        self.executor = cast(LocalContainerExecutor, self.executor)
        self.service = cast(DotEnvSecrets, self.service)

        self.service.location = self.executor._container_secrets_location
