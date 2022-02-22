import logging
from pathlib import Path
from typing import TYPE_CHECKING

from magnus import defaults
from magnus.catalog import BaseCatalog
from magnus.datastore import BaseRunLogStore
from magnus.secrets import BaseSecrets

if TYPE_CHECKING:
    from magnus.executor import BaseExecutor

from stevedore import extension

logger = logging.getLogger(defaults.NAME)


class BaseIntegration:
    """
    Base class for handling integration between Executor and one of Catalog, Secrets, RunLogStore.
    """
    mode_type = ''
    service_type = ''  # One of secret, catalog, datastore
    service_provider = ''  # The actual implementation of the service

    def __init__(self, executor: 'BaseExecutor', integration_service: object):
        self.executor = executor
        self.service = integration_service

    def validate(self, **kwargs):
        """
        Raise an exception if the mode_type is not compatible with service provider.

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


def get_service_type(service_provider: object) -> str:
    """
    Given a service provider, identify the type of service.

    Args:
        service_provider (object): The service provider object

    Raises:
        Exception: If the service provider is not inherited from one of BaseSecret, BaseCatalog, BaseRunLogStore

    Returns:
        [str]: Returns either 'secret', 'catalog', 'run_log_store' according to the service provider.
    """
    if isinstance(service_provider, BaseSecrets):
        return 'secrets'
    if isinstance(service_provider, BaseCatalog):
        return 'catalog'
    if isinstance(service_provider, BaseRunLogStore):
        return 'run_log_store'

    raise Exception('Service Provider is not a inherited from any of the Base Service providers')


def get_integration_handler(executor: 'BaseExecutor', service: object) -> BaseIntegration:
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
    service_type = get_service_type(service)
    service_name = getattr(service, 'service_name')
    integrations = []

    mgr = extension.ExtensionManager(
        namespace="magnus.integration.BaseIntegration",
        invoke_on_load=True,
        invoke_kwds={'executor': executor, 'integration_service': service}
    )
    for name, kls in mgr.items():
        if (kls.obj.service_type == service_type and  # type: ignore
                kls.obj.mode_type == executor.service_name and  # type: ignore
                kls.obj.service_provider == service_name):
            logger.info(f'Identified an intergration pattern {kls.obj}')
            integrations.append(kls.obj)

    if len(integrations) > 1:
        msg = (
            f'Multiple integrations between {executor.service_name} and {service_name} of type {service_type} found. '
            'This is not correct. Please raise a bug report to fix this.'
        )
        logger.exception(msg)
        raise Exception(msg)

    if not integrations:
        logger.warning(
            f'Could not find an integration pattern for {executor.service_name} and {service_name} for {service_type}')  # type: ignore
        return BaseIntegration(executor, service)

    return integrations[0]


def validate(executor: 'BaseExecutor', service: object, **kwargs):
    """
    Helper function to resolve the Integration class and validate the compatibility between executor and service

    Args:
        executor (BaseExecutor) : The executor
        service (object): The service provider
    """
    integration_handler = get_integration_handler(executor, service)
    integration_handler.validate(**kwargs)


def configure_for_traversal(executor: 'BaseExecutor', service: object, **kwargs):
    """
    Helper function to resolve the Integration class and configure the executor and service for graph traversal

    Args:
        executor (BaseExecutor) : The executor
        service (object): The service provider
    """
    integration_handler = get_integration_handler(executor, service)
    integration_handler.configure_for_traversal(**kwargs)


def configure_for_execution(executor: 'BaseExecutor', service: object, **kwargs):
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
    mode_type = 'local'
    service_type = 'run_log_store'  # One of secret, catalog, datastore
    service_provider = 'buffered'  # The actual implementation of the service

    def validate(self, **kwargs):
        msg = (
            'Run log generated by buffered run log store are not persisted. '
            'Re-running this run, in case of a failure, is not possible'
        )
        logger.warning(msg)


class LocalComputeFileSystemRunLogStore(BaseIntegration):
    """
    Local compute and File system run log store
    """
    mode_type = 'local'
    service_type = 'run_log_store'  # One of secret, catalog, datastore
    service_provider = 'file-system'  # The actual implementation of the service

    def validate(self, **kwargs):
        if self.executor.is_parallel_execution():
            msg = (
                'Run log generated by file-system run log store are not thread safe. '
                'Inconsistent results are possible because of race conditions to write to the same file.\n'
                'Consider using partitioned run log store like database for consistent results.'

            )
            logger.warning(msg)


class LocalContainerComputeBufferedRunLogStore(BaseIntegration):
    """
    Only local execution mode is possible for Buffered Run Log store
    """
    mode_type = 'local-container'
    service_type = 'run_log_store'  # One of secret, catalog, datastore
    service_provider = 'buffered'  # The actual implementation of the service

    def validate(self, **kwargs):
        raise Exception('Only local compute mode is possible for buffered run log store')


class LocalContainerComputeFileSystemRunLogstore(BaseIntegration):
    """
    Integration between local container and file system run log store
    """
    mode_type = 'local-container'
    service_type = 'run_log_store'  # One of secret, catalog, datastore
    service_provider = 'file-system'  # The actual implementation of the service

    def validate(self, **kwargs):
        if self.executor.is_parallel_execution():
            msg = (
                'Run log generated by file-system run log store are not thread safe. '
                'Inconsistent results are possible because of race conditions to write to the same file.\n'
                'Consider using partitioned run log store like database for consistent results.'

            )
            logger.warning(msg)

    def configure_for_traversal(self, **kwargs):
        write_to = self.service.log_folder_name
        self.executor.volumes[str(Path(write_to).resolve())] = {
            'bind': f'{self.executor.container_log_location}',
            'mode': 'rw'
        }

    def configure_for_execution(self, **kwargs):
        self.service.config[self.service.CONFIG_KEY_LOG_FOLDER] = self.executor.container_log_location


class LocalContainerComputeDotEnvSecrets(BaseIntegration):
    """
    Integration between local container and dot env secrets
    """
    mode_type = 'local-container'
    service_type = 'secrets'  # One of secret, catalog, datastore
    service_provider = 'dotenv'  # The actual implementation of the service

    def validate(self, **kwargs):
        logger.warning('Using dot env for non local deployments is not ideal, consider options')

    def configure_for_traversal(self, **kwargs):
        secrets_location = self.service.get_secrets_location()
        self.executor.volumes[str(Path(secrets_location).resolve())] = {
            'bind': f'{self.executor.container_secrets_location}',
            'mode': 'ro'
        }

    def configure_for_execution(self, **kwargs):
        self.service.config['location'] = self.executor.container_secrets_location


class LocalContainerDoNothingCatalog(BaseIntegration):
    """
    Integration between local container and do nothing catalog
    """
    mode_type = 'local-container'
    service_type = 'catalog'  # One of secret, catalog, datastore
    service_provider = 'do-nothing'  # The actual implementation of the service

    def validate(self, **kwargs):
        msg = (
            'A do-nothing catalog does not hold any data and therefore cannot pass data between nodes.'
        )
        logger.warning(msg)


class LocalDoNothingCatalog(BaseIntegration):
    """
    Integration between local and do nothing catalog
    """
    mode_type = 'local'
    service_type = 'catalog'  # One of secret, catalog, datastore
    service_provider = 'do-nothing'  # The actual implementation of the service

    def validate(self, **kwargs):
        msg = (
            'A do-nothing catalog does not hold any data and therefore cannot pass data between nodes.'
        )
        logger.warning(msg)


class LocalContainerComputeFileSystemCatalog(BaseIntegration):
    """
    Integration pattern between Local container and File System catalog
    """
    mode_type = 'local-container'
    service_type = 'catalog'  # One of secret, catalog, datastore
    service_provider = 'file-system'  # The actual implementation of the service

    def configure_for_traversal(self, **kwargs):
        catalog_location = self.service.get_catalog_location()
        self.executor.volumes[str(Path(catalog_location).resolve())] = {
            'bind': f'{self.executor.container_catalog_location}',
            'mode': 'rw'
        }

    def configure_for_execution(self, **kwargs):
        self.service.config['catalog_location'] = self.executor.container_catalog_location


class DemoRenderBufferedRunLogStore(BaseIntegration):
    """
    Demo rendered and buffered
    """
    mode_type = 'demo-renderer'
    service_type = 'run_log_store'  # One of secret, catalog, datastore
    service_provider = 'buffered'  # The actual implementation of the service

    def validate(self, **kwargs):
        msg = (
            'Run log generated by buffered run log store are not persisted. '
            'Demo renderer cannot use buffered as steps are executed as individual commands.'
        )
        logger.exception(msg)
        raise Exception(msg)
