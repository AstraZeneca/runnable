import logging
from pathlib import Path

from magnus import defaults
from magnus.datastore import FileSystemRunLogstore

logger = logging.getLogger(defaults.NAME)


class K8PersistentVolumeRunLogstore(FileSystemRunLogstore):
    """
    Uses the K8s Persistent Volumes to store run logs.
    """

    service_name = "k8s-pvc"

    class ContextConfig(FileSystemRunLogstore.ContextConfig):
        persistent_volume_name: str
        mount_path: str

    def __init__(self, config: dict, **kwargs):  # pylint: disable=unused-argument
        self.config = self.ContextConfig(**(config or {}))

    @property
    def log_folder_name(self) -> str:
        return str(Path(self.config.mount_path) / self.config.log_folder)
