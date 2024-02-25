import logging

from stevedore import extension

from runnable import defaults
from runnable.executor import BaseExecutor

logger = logging.getLogger(defaults.LOGGER_NAME)
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

    # Get all the integrations defined by the 3rd party in their pyproject.toml
    mgr = extension.ExtensionManager(
        namespace="integration",
        invoke_on_load=True,
        invoke_kwds={"executor": executor, "integration_service": service},
    )
    for _, kls in mgr.items():
        if (
            kls.obj.executor_type == executor.service_name
            and kls.obj.service_type == service_type
            and kls.obj.service_provider == service_name
        ):
            logger.info(f"Identified an integration pattern {kls.obj}")
            integrations.append(kls.obj)

    # Get all the implementations defined by the runnable package
    for kls in BaseIntegration.__subclasses__():
        # Match the exact service type
        if kls.service_type == service_type and kls.service_provider == service_name:
            # Match either all executor or specific ones provided
            if kls.executor_type == "" or kls.executor_type == executor.service_name:
                integrations.append(kls(executor=executor, integration_service=service))

    if len(integrations) > 1:
        msg = (
            f"Multiple integrations between {executor.service_name} and {service_name} of type {service_type} found. "
            "If you defined an integration pattern, please ensure it is specific and does not conflict with runnable "
            " implementations."
        )
        logger.exception(msg)
        raise Exception(msg)

    if not integrations:
        logger.warning(
            f"Could not find an integration pattern for {executor.service_name} and {service_name} for {service_type}."
            " This implies that there is no need to change the configurations."
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


class BufferedRunLogStore(BaseIntegration):
    """
    Integration between any executor and buffered run log store
    """

    service_type = "run_log_store"  # One of secret, catalog, datastore
    service_provider = "buffered"  # The actual implementation of the service

    def validate(self, **kwargs):
        if not self.executor.service_name == "local":
            raise Exception("Buffered run log store is only supported for local executor")

        msg = (
            "Run log generated by buffered run log store are not persisted. "
            "Re-running this run, in case of a failure, is not possible"
        )
        logger.warning(msg)


class DoNothingCatalog(BaseIntegration):
    """
    Integration between any executor and do nothing catalog
    """

    service_type = "catalog"  # One of secret, catalog, datastore
    service_provider = "do-nothing"  # The actual implementation of the service

    def validate(self, **kwargs):
        msg = "A do-nothing catalog does not hold any data and therefore cannot pass data between nodes."
        logger.warning(msg)


class DoNothingSecrets(BaseIntegration):
    """
    Integration between any executor and do nothing secrets
    """

    service_type = "secrets"  # One of secret, catalog, datastore
    service_provider = "do-nothing"  # The actual implementation of the service

    def validate(self, **kwargs):
        msg = "A do-nothing secrets does not hold any secrets and therefore cannot return you any secrets."
        logger.warning(msg)


class DoNothingExperimentTracker(BaseIntegration):
    """
    Integration between any executor and do nothing experiment tracker
    """

    service_type = "experiment_tracker"  # One of secret, catalog, datastore
    service_provider = "do-nothing"  # The actual implementation of the service

    def validate(self, **kwargs):
        msg = "A do-nothing experiment tracker does nothing and therefore cannot track anything."
        logger.warning(msg)
