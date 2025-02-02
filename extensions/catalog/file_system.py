import logging
import shutil
from pathlib import Path
from typing import Any

from cloudpathlib import CloudPath
from pydantic import Field

from extensions.catalog.any_path import AnyPathCatalog
from runnable import defaults

logger = logging.getLogger(defaults.LOGGER_NAME)


class FileSystemCatalog(AnyPathCatalog):
    service_name: str = "file-system"

    catalog_location: str = Field(default=defaults.CATALOG_LOCATION_FOLDER)

    def get_summary(self) -> dict[str, Any]:
        return {
            "compute_data_folder": self.compute_data_folder,
            "catalog_location": self.catalog_location,
        }

    def get_catalog_location(self) -> Path:
        run_id = self._context.run_id
        return Path(self.catalog_location) / run_id / self.compute_data_folder

    def download_from_catalog(self, file: Path | CloudPath) -> None:
        assert isinstance(file, Path)

        run_catalog = self.get_catalog_location()
        relative_file_path = file.relative_to(run_catalog)

        copy_to = self.compute_data_folder
        # Make the directory in the data folder if required
        Path(copy_to / relative_file_path.parent).mkdir(parents=True, exist_ok=True)
        shutil.copy(file, copy_to / relative_file_path)

    def upload_to_catalog(self, file: Path) -> None:
        run_catalog = self.get_catalog_location()
        run_catalog.mkdir(parents=True, exist_ok=True)

        logger.debug(
            f"Copying objects from {self.compute_data_folder} to the run catalog location of {run_catalog}"
        )

        relative_file_path = file.relative_to(self.compute_data_folder)

        (run_catalog / relative_file_path.parent).mkdir(parents=True, exist_ok=True)
        shutil.copy(file, run_catalog / relative_file_path)
