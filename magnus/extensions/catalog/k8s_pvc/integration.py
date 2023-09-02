import logging

from magnus import defaults
from magnus.integration import BaseIntegration

logger = logging.getLogger(defaults.NAME)


class LocalCompute(BaseIntegration):
    """
    Integration between local and k8's pvc
    """

    executor_type = "local"
    service_type = "catalog"  # One of secret, catalog, datastore
    service_provider = "k8s-pvc"  # The actual implementation of the service

    def validate(self, **kwargs):
        msg = "We can't use the local compute k8s pvc store integration."
        raise Exception(msg)


class LocalContainerCompute(BaseIntegration):
    """
    Integration between local-container and k8's pvc
    """

    executor_type = "local-container"
    service_type = "catalog"  # One of secret, catalog, datastore
    service_provider = "k8s-pvc"  # The actual implementation of the service

    def validate(self, **kwargs):
        msg = "We can't use the local-container compute k8s pvc store integration."
        raise Exception(msg)


class ArgoCompute(BaseIntegration):
    """
    Integration between argo and k8's pvc
    """

    executor_type = "argo"
    service_type = "catalog"  # One of secret, catalog, datastore
    service_provider = "k8s-pvc"  # The actual implementation of the service

    def configure_for_traversal(self, **kwargs):
        self.executor.persistent_volumes["catalog"] = (
            self.service.config.persistent_volume_name,
            self.service.config.mount_path,
        )
