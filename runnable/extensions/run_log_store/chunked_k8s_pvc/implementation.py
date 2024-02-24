import logging
from pathlib import Path

from runnable import defaults
from runnable.extensions.run_log_store.chunked_file_system.implementation import ChunkedFileSystemRunLogStore

logger = logging.getLogger(defaults.NAME)


class ChunkedK8PersistentVolumeRunLogstore(ChunkedFileSystemRunLogStore):
    """
    Uses the K8s Persistent Volumes to store run logs.
    """

    service_name: str = "chunked-k8s-pvc"
    persistent_volume_name: str
    mount_path: str

    @property
    def log_folder_name(self) -> str:
        return str(Path(self.mount_path) / self.log_folder)
