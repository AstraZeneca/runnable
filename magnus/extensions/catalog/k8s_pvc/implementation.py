import logging
from pathlib import Path

from magnus import defaults
from magnus.catalog import FileSystemCatalog

logger = logging.getLogger(defaults.LOGGER_NAME)


class K8sPVCatalog(FileSystemCatalog):
    service_name = "k8s-pvc"

    class ContextConfig(FileSystemCatalog.ContextConfig):
        persistent_volume_name: str
        mount_path: str

    def __init__(self, config: dict, **kwargs):  # pylint: disable=unused-argument
        self.config = self.ContextConfig(**(config or {}))

    @property
    def catalog_location(self) -> str:
        """
        Get the catalog location from the config.
        If its not defined, use the magnus default

        Returns:
            str: The catalog location as defined by the config or magnus default '.catalog'
        """
        return str(Path(self.config.mount_path) / self.config.catalog_location)  # type: ignore
