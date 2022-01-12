import logging
from pathlib import Path


from magnus import defaults
from magnus.secrets import BaseSecrets
from magnus.catalog import BaseCatalog
from magnus.datastore import BaseRunLogStore


logger = logging.getLogger(defaults.NAME)


class BaseIntegration:
    """
    Base class for handling integration between Executor and one of Catalog, Secrets, RunLogStore.
    """
    mode_type = None
    service_type = None  # One of secret, catalog, datastore
    service_provider = None  # The actual implementation of the service

    def __init__(self, executor, integration_service):
        self.executor = executor
        self.service = integration_service

    # TODO: Remove this as an API
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

    def configure_for_execution(self,  **kwargs):
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
        return 'run-log-store'

    raise Exception('Service Provider is not a inherited from any of the Base Service providers')


def get_integration_handler(executor: object, service: object) -> BaseIntegration:
    """
    Return the integration handler between executor and the service.

    If none found to be implemented, return the BaseIntegration which does nothing.

    Args:
        executor (BaseExecutor): The executor
        service (object): The service provider

    Returns:
        [BaseIntegration]: The implemented integration handler or BaseIntegration if none found
    """
    service_type = get_service_type(service)
    service_name = getattr(service, 'service_name')
    for sub_class in BaseIntegration.__subclasses__():
        if sub_class.service_type == service_type and \
                sub_class.mode_type == executor.service_name and \
                sub_class.service_provider == service_name:
            logger.info(f'Identified an intergration pattern {sub_class.__name__}')
            return sub_class(executor, service)

    logger.warning(f'Could not find an integration pattern for {executor.service_name} and {service_name}')
    return BaseIntegration(executor, service)


def validate(executor: object, service: object, **kwargs):
    """
    Helper function to resolve the Integration class and validate the compatibility between executor and service

    Args:
        executor (BaseExecutor) : The executor
        service (object): The service provider
    """
    integration_handler = get_integration_handler(executor, service)
    integration_handler.validate(**kwargs)


def configure_for_traversal(executor: object, service: object, **kwargs):
    """
    Helper function to resolve the Integration class and configure the executor and service for graph traversal

    Args:
        executor (BaseExecutor) : The executor
        service (object): The service provider
    """
    integration_handler = get_integration_handler(executor, service)
    integration_handler.configure_for_traversal(**kwargs)


def configure_for_execution(executor: object, service: object, **kwargs):
    """
    Helper function to resolve the Integration class and configure the executor and service for execution

    Args:
        executor (BaseExecutor) : The executor
        service (object): The service provider
    """
    integration_handler = get_integration_handler(executor, service)
    integration_handler.configure_for_execution(**kwargs)


class LocalContainerComputeBufferedRunLogStore(BaseIntegration):
    """
    Only local execution mode is possible for Buffered Run Log store
    """
    mode_type = 'local-container'
    service_type = 'run-log-store'  # One of secret, catalog, datastore
    service_provider = 'buffered'  # The actual implementation of the service

    def validate(self, **kwargs):
        raise Exception('Only Local compute mode is possible for Buffered Run Log store')


class LocalContainerComputeFileSystemRunLogstore(BaseIntegration):
    """
    Integration between local container and file system run log store
    """
    mode_type = 'local-container'
    service_type = 'run-log-store'  # One of secret, catalog, datastore
    service_provider = 'file-system'  # The actual implementation of the service

    def configure_for_traversal(self, **kwargs):
        write_to = self.service.get_folder_name()
        self.executor.volumes[str(Path(write_to).resolve())] = {
            'bind': f'{self.executor.container_log_location}',
            'mode': 'rw'
        }

    def configure_for_execution(self, **kwargs):
        self.service.config[self.service.write_to_key] = self.executor.container_log_location


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


# TODO: Write integration patterns for demo-renderer and other systems
