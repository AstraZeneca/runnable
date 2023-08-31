import logging
from pathlib import Path
from typing import cast

from stevedore import extension

from magnus import defaults
from magnus.catalog import FileSystemCatalog
from magnus.datastore import FileSystemRunLogstore
from magnus.executor import BaseExecutor, LocalContainerExecutor
from magnus.secrets import DotEnvSecrets

logger = logging.getLogger(defaults.NAME)
logging.getLogger("stevedore").setLevel(logging.CRITICAL)

# --8<-- [start:docs]


class BaseIntegration:
    """
    Base class for handling integration between Executor and one of Catalog, Secrets, RunLogStore.
    """

    executor_type = ""
    service_type = ""  # One of secret, catalog, datastore, experiment tracker
    service_provider = ""  # The actual implementation of the service

    def __init__(self, executor: "BaseExecutor", integration_service: object):
        self.executor = executor
        self.service = integration_service

    def validate(self, **kwargs):
        """
        Raise an exception if the executor_type is not compatible with service provider.

        By default, it is considered as compatible.
        """

    def configure_for_traversal(self, **kwargs):
        """
        Do any changes needed to both executor and service provider during traversal of the graph.

        By default, no change is required.
        """

    def configure_for_execution(self, **kwargs):
        """
        Do any changes needed to both executor and service provider during execution of a node.

        By default, no change is required.
        """


# --8<-- [end:docs]


def get_integration_handler(executor: "BaseExecutor", service: object) -> BaseIntegration:
    """
    Return the integration handler between executor and the service.

    If none found to be implemented, return the BaseIntegration which does nothing.

    Args:
        executor (BaseExecutor): The executor
        service (object): The service provider

    Returns:
        [BaseIntegration]: The implemented integration handler or BaseIntegration if none found

    Raises:
        Exception: If multiple integrations are found for the executor and service
    """
    service_type = service.service_type  # type: ignore
    service_name = getattr(service, "service_name")
    integrations = []

    # This should be this + subclasses of the BaseIntegration class
    # Not clear on the full idea on how to implement this.
    mgr = extension.ExtensionManager(
        namespace="integration",
        invoke_on_load=True,
        invoke_kwds={"executor": executor, "integration_service": service},
    )
    for _, kls in mgr.items():
        if (
            kls.obj.service_type == service_type
            and kls.obj.executor_type == executor.service_name
            and kls.obj.service_provider == service_name
        ):
            logger.info(f"Identified an integration pattern {kls.obj}")
            integrations.append(kls.obj)

    if len(integrations) > 1:
        msg = (
            f"Multiple integrations between {executor.service_name} and {service_name} of type {service_type} found. "
            "This is not correct. Please raise a bug report to fix this."
        )
        logger.exception(msg)
        raise Exception(msg)

    if not integrations:
        logger.warning(
            f"Could not find an integration pattern for {executor.service_name} and {service_name} for {service_type}"
        )
        return BaseIntegration(executor, service)

    return integrations[0]


def validate(executor: "BaseExecutor", service: object, **kwargs):
    """
    Helper function to resolve the Integration class and validate the compatibility between executor and service

    Args:
        executor (BaseExecutor) : The executor
        service (object): The service provider
    """
    integration_handler = get_integration_handler(executor, service)
    integration_handler.validate(**kwargs)


def configure_for_traversal(executor: "BaseExecutor", service: object, **kwargs):
    """
    Helper function to resolve the Integration class and configure the executor and service for graph traversal

    Args:
        executor (BaseExecutor) : The executor
        service (object): The service provider
    """
    integration_handler = get_integration_handler(executor, service)
    integration_handler.configure_for_traversal(**kwargs)


def configure_for_execution(executor: "BaseExecutor", service: object, **kwargs):
    """
    Helper function to resolve the Integration class and configure the executor and service for execution

    Args:
        executor (BaseExecutor) : The executor
        service (object): The service provider
    """
    integration_handler = get_integration_handler(executor, service)
    integration_handler.configure_for_execution(**kwargs)


class LocalComputeBufferedRunLogStore(BaseIntegration):
    """
    Local compute and buffered
    """

    executor_type = "local"
    service_type = "run_log_store"  # One of secret, catalog, datastore
    service_provider = "buffered"  # The actual implementation of the service

    def validate(self, **kwargs):
        msg = (
            "Run log generated by buffered run log store are not persisted. "
            "Re-running this run, in case of a failure, is not possible"
        )
        logger.warning(msg)


class LocalComputeFileSystemRunLogStore(BaseIntegration):
    """
    Local compute and File system run log store
    """

    executor_type = "local"
    service_type = "run_log_store"  # One of secret, catalog, datastore
    service_provider = "file-system"  # The actual implementation of the service

    def validate(self, **kwargs):
        if self.executor._is_parallel_execution():
            msg = (
                "Run log generated by file-system run log store are not thread safe. "
                "Inconsistent results are possible because of race conditions to write to the same file.\n"
                "Consider using partitioned run log store like database for consistent results."
            )
            logger.warning(msg)


class LocalContainerComputeBufferedRunLogStore(BaseIntegration):
    """
    Only local executions is possible for Buffered Run Log store
    """

    executor_type = "local-container"
    service_type = "run_log_store"  # One of secret, catalog, datastore
    service_provider = "buffered"  # The actual implementation of the service

    def validate(self, **kwargs):
        raise Exception("Only local compute executions is possible for buffered run log store")


class LocalContainerComputeFileSystemRunLogstore(BaseIntegration):
    """
    Integration between local container and file system run log store
    """

    executor_type = "local-container"
    service_type = "run_log_store"  # One of secret, catalog, datastore
    service_provider = "file-system"  # The actual implementation of the service

    def validate(self, **kwargs):
        if self.executor._is_parallel_execution():
            msg = (
                "Run log generated by file-system run log store are not thread safe. "
                "Inconsistent results are possible because of race conditions to write to the same file.\n"
                "Consider using partitioned run log store like database for consistent results."
            )
            logger.warning(msg)

    def configure_for_traversal(self, **kwargs):
        self.executor = cast(LocalContainerExecutor, self.executor)
        self.service = cast(FileSystemRunLogstore, self.service)

        write_to = self.service.log_folder_name
        self.executor.volumes[str(Path(write_to).resolve())] = {
            "bind": f"{self.executor.container_log_location}",
            "mode": "rw",
        }

    def configure_for_execution(self, **kwargs):
        self.executor = cast(LocalContainerExecutor, self.executor)
        self.service = cast(FileSystemRunLogstore, self.service)

        self.service.config.log_folder = self.executor.container_log_location


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
        self.executor.volumes[str(Path(secrets_location).resolve())] = {
            "bind": f"{self.executor.container_secrets_location}",
            "mode": "ro",
        }

    def configure_for_execution(self, **kwargs):
        self.executor = cast(LocalContainerExecutor, self.executor)
        self.service = cast(DotEnvSecrets, self.service)

        self.service.config.location = self.executor.container_secrets_location


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


class LocalContainerDoNothingCatalog(BaseIntegration):
    """
    Integration between local container and do nothing catalog
    """

    executor_type = "local-container"
    service_type = "catalog"  # One of secret, catalog, datastore
    service_provider = "do-nothing"  # The actual implementation of the service

    def validate(self, **kwargs):
        msg = "A do-nothing catalog does not hold any data and therefore cannot pass data between nodes."
        logger.warning(msg)


class LocalDoNothingCatalog(BaseIntegration):
    """
    Integration between local and do nothing catalog
    """

    executor_type = "local"
    service_type = "catalog"  # One of secret, catalog, datastore
    service_provider = "do-nothing"  # The actual implementation of the service

    def validate(self, **kwargs):
        msg = "A do-nothing catalog does not hold any data and therefore cannot pass data between nodes."
        logger.warning(msg)


class LocalContainerComputeFileSystemCatalog(BaseIntegration):
    """
    Integration pattern between Local container and File System catalog
    """

    executor_type = "local-container"
    service_type = "catalog"  # One of secret, catalog, datastore
    service_provider = "file-system"  # The actual implementation of the service

    def configure_for_traversal(self, **kwargs):
        self.executor = cast(LocalContainerExecutor, self.executor)
        self.service = cast(FileSystemCatalog, self.service)

        catalog_location = self.service.catalog_location
        self.executor.volumes[str(Path(catalog_location).resolve())] = {
            "bind": f"{self.executor.container_catalog_location}",
            "mode": "rw",
        }

    def configure_for_execution(self, **kwargs):
        self.executor = cast(LocalContainerExecutor, self.executor)
        self.service = cast(FileSystemCatalog, self.service)

        self.service.config.catalog_location = self.executor.container_catalog_location


class DemoRenderBufferedRunLogStore(BaseIntegration):
    """
    Demo rendered and buffered
    """

    executor_type = "demo-renderer"
    service_type = "run_log_store"  # One of secret, catalog, datastore
    service_provider = "buffered"  # The actual implementation of the service

    def validate(self, **kwargs):
        msg = (
            "Run log generated by buffered run log store are not persisted. "
            "Demo renderer cannot use buffered as steps are executed as individual commands."
        )
        logger.exception(msg)
        raise Exception(msg)
