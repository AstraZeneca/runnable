import logging
from typing import cast

from runnable import defaults
from runnable.integration import BaseIntegration

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
        from runnable.extensions.catalog.k8s_pvc.implementation import K8sPVCatalog
        from runnable.extensions.executor.argo.implementation import ArgoExecutor, UserVolumeMounts

        self.executor = cast(ArgoExecutor, self.executor)
        self.service = cast(K8sPVCatalog, self.service)

        volume_mount = UserVolumeMounts(
            name=self.service.persistent_volume_name,
            mount_path=self.service.mount_path,
        )

        self.executor.persistent_volumes.append(volume_mount)
