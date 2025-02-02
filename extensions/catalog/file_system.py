from pathlib import Path
from typing import Type

from extensions.catalog.any_path import AnyPathCatalog


class FileSystemCatalog(AnyPathCatalog):
    service_name: str = "file-system"

    def get_path_client(self) -> Type[Path]:
        return Path
