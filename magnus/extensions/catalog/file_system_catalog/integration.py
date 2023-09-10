from pathlib import Path
from typing import cast

from magnus.extensions.executor.local_container.implementation import LocalContainerExecutor
from magnus.integration import BaseIntegration

from .implementation import FileSystemCatalog


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
        self.executor._volumes[str(Path(catalog_location).resolve())] = {
            "bind": f"{self.executor._container_catalog_location}",
            "mode": "rw",
        }

    def configure_for_execution(self, **kwargs):
        self.executor = cast(LocalContainerExecutor, self.executor)
        self.service = cast(FileSystemCatalog, self.service)

        self.service.catalog_location = self.executor._container_catalog_location
