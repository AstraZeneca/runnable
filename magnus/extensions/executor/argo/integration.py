import logging

from magnus import defaults
from magnus.integration import BaseIntegration

logger = logging.getLogger(defaults.NAME)


class BufferedRunLogStore(BaseIntegration):
    """
    Only local execution mode is possible for Buffered Run Log store
    """

    executor_type = "argo"
    service_type = "run_log_store"  # One of secret, catalog, datastore
    service_provider = "buffered"  # The actual implementation of the service

    def validate(self, **kwargs):
        raise Exception("Argo cannot run work with buffered run log store")


class FileSystemRunLogStore(BaseIntegration):
    """
    Only local execution mode is possible for Buffered Run Log store
    """

    executor_type = "argo"
    service_type = "run_log_store"  # One of secret, catalog, datastore
    service_provider = "file-system"  # The actual implementation of the service

    def validate(self, **kwargs):
        msg = (
            "Argo cannot run work with file-system run log store. Unless you have made a mechanism to use volume mounts"
        )
        logger.warning(msg)


class FileSystemCatalog(BaseIntegration):
    """
    Only local execution mode is possible for Buffered Run Log store
    """

    executor_type = "argo"
    service_type = "catalog"  # One of secret, catalog, datastore
    service_provider = "file-system"  # The actual implementation of the service

    def validate(self, **kwargs):
        msg = (
            "Argo cannot run work with file-system run log store. Unless you have made a mechanism to use volume mounts"
        )
        logger.warning(msg)
