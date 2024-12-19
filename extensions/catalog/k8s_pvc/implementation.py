import logging
from pathlib import Path

from runnable import defaults
from runnable.extensions.catalog.file_system.implementation import FileSystemCatalog

logger = logging.getLogger(defaults.LOGGER_NAME)


class K8sPVCatalog(FileSystemCatalog):
    service_name: str = "k8s-pvc"
    persistent_volume_name: str
    mount_path: str

    def get_catalog_location(self):
        return str(Path(self.mount_path) / self.catalog_location)
