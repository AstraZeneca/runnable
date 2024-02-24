import logging

from runnable import defaults
from runnable.integration import BaseIntegration

logger = logging.getLogger(defaults.NAME)


class BufferedRunLogStore(BaseIntegration):
    """
    Only local execution mode is possible for Buffered Run Log store
    """

    executor_type = "k8s-job"
    service_type = "run_log_store"  # One of secret, catalog, datastore
    service_provider = "buffered"  # The actual implementation of the service

    def validate(self, **kwargs):
        raise Exception("K8s job cannot run work with buffered run log store")


class FileSystemRunLogStore(BaseIntegration):
    """
    Only local execution mode is possible for Buffered Run Log store
    """

    executor_type = "k8s-job"
    service_type = "run_log_store"  # One of secret, catalog, datastore
    service_provider = "file-system"  # The actual implementation of the service

    def validate(self, **kwargs):
        msg = (
            "K8s job cannot run work with file-system run log store."
            "Unless you have made a mechanism to use volume mounts"
        )
        logger.warning(msg)


class ChunkedFSRunLogStore(BaseIntegration):
    """
    Only local execution mode is possible for Buffered Run Log store
    """

    executor_type = "k8s-job"
    service_type = "run_log_store"  # One of secret, catalog, datastore
    service_provider = "chunked-fs"  # The actual implementation of the service

    def validate(self, **kwargs):
        msg = (
            "K8s job cannot run work with chunked-fs run log store."
            "Unless you have made a mechanism to use volume mounts"
        )
        logger.warning(msg)


class FileSystemCatalog(BaseIntegration):
    """
    Only local execution mode is possible for Buffered Run Log store
    """

    executor_type = "k8s-job"
    service_type = "catalog"  # One of secret, catalog, datastore
    service_provider = "file-system"  # The actual implementation of the service

    def validate(self, **kwargs):
        msg = (
            "K8s Job cannot run work with file-system catalog." "Unless you have made a mechanism to use volume mounts"
        )
        logger.warning(msg)
