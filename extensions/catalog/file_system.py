import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from runnable import defaults, utils
from runnable.catalog import BaseCatalog, is_catalog_out_of_sync
from runnable.datastore import DataCatalog

logger = logging.getLogger(defaults.LOGGER_NAME)


class FileSystemCatalog(BaseCatalog):
    """
    A Catalog handler that uses the local file system for cataloging.

    Note: Do not use this if the steps of the pipeline run on different compute environments.

    Example config:

    catalog:
      type: file-system
      config:
        catalog_location: The location to store the catalog.
        compute_data_folder: The folder to source the data from.

    """

    service_name: str = "file-system"
    catalog_location: str = defaults.CATALOG_LOCATION_FOLDER

    def get_catalog_location(self):
        return self.catalog_location

    def get_summary(self) -> Dict[str, Any]:
        summary = {
            "Catalog Location": self.get_catalog_location(),
        }

        return summary

    def get(
        self, name: str, run_id: str, compute_data_folder: str = "", **kwargs
    ) -> List[DataCatalog]:
        """
        Get the file by matching glob pattern to the name

        Args:
            name ([str]): A glob matching the file name
            run_id ([str]): The run id

        Raises:
            Exception: If the catalog location does not exist

        Returns:
            List(object) : A list of catalog objects
        """
        logger.info(
            f"Using the {self.service_name} catalog and trying to get {name} for run_id: {run_id}"
        )

        copy_to = self.compute_data_folder
        if compute_data_folder:
            copy_to = compute_data_folder

        copy_to = Path(copy_to)  # type: ignore

        catalog_location = self.get_catalog_location()
        run_catalog = Path(catalog_location) / run_id / copy_to

        logger.debug(
            f"Copying objects to {copy_to} from the run catalog location of {run_catalog}"
        )

        if not utils.does_dir_exist(run_catalog):
            msg = (
                f"Expected Catalog to be present at: {run_catalog} but not found.\n"
                "Note: Please make sure that some data was put in the catalog before trying to get from it.\n"
            )
            raise Exception(msg)

        # Iterate through the contents of the run_catalog and copy the files that fit the name pattern
        # We should also return a list of data hashes
        glob_files = run_catalog.glob(name)
        logger.debug(
            f"Glob identified {glob_files} as matches to from the catalog location: {run_catalog}"
        )

        data_catalogs = []
        run_log_store = self._context.run_log_store
        for file in glob_files:
            if file.is_dir():
                # Need not add a data catalog for the folder
                continue

            if str(file).endswith(".execution.log"):
                continue

            relative_file_path = file.relative_to(run_catalog)

            data_catalog = run_log_store.create_data_catalog(str(relative_file_path))
            data_catalog.catalog_handler_location = catalog_location
            data_catalog.catalog_relative_path = str(relative_file_path)
            data_catalog.data_hash = utils.get_data_hash(str(file))
            data_catalog.stage = "get"
            data_catalogs.append(data_catalog)

            # Make the directory in the data folder if required
            Path(copy_to / relative_file_path.parent).mkdir(parents=True, exist_ok=True)
            shutil.copy(file, copy_to / relative_file_path)

            logger.info(f"Copied {file} from {run_catalog} to {copy_to}")

        if not data_catalogs:
            raise Exception(f"Did not find any files matching {name} in {run_catalog}")

        return data_catalogs

    def put(
        self,
        name: str,
        run_id: str,
        compute_data_folder: str = "",
        synced_catalogs: Optional[List[DataCatalog]] = None,
        **kwargs,
    ) -> List[DataCatalog]:
        """
        Put the files matching the glob pattern into the catalog.

        If previously synced catalogs are provided, and no changes were observed, we do not sync them.

        Args:
            name (str): The glob pattern of the files to catalog
            run_id (str): The run id of the run
            compute_data_folder (str, optional): The compute data folder to sync from. Defaults to settings default.
            synced_catalogs (dict, optional): dictionary of previously synced catalogs. Defaults to None.

        Raises:
            Exception: If the compute data folder does not exist.

        Returns:
            List(object) : A list of catalog objects
        """
        logger.info(
            f"Using the {self.service_name} catalog and trying to put {name} for run_id: {run_id}"
        )

        copy_from = self.compute_data_folder
        if compute_data_folder:
            copy_from = compute_data_folder
        copy_from = Path(copy_from)  # type: ignore

        catalog_location = self.get_catalog_location()
        run_catalog = Path(catalog_location) / run_id
        utils.safe_make_dir(run_catalog)

        logger.debug(
            f"Copying objects from {copy_from} to the run catalog location of {run_catalog}"
        )

        if not utils.does_dir_exist(copy_from):
            msg = (
                f"Expected compute data folder to be present at: {compute_data_folder} but not found. \n"
                "Note: runnable does not create the compute data folder for you. Please ensure that the "
                "folder exists.\n"
            )
            raise Exception(msg)

        # Iterate through the contents of copy_from and if the name matches, we move them to the run_catalog
        # We should also return a list of datastore.DataCatalog items

        glob_files = copy_from.glob(name)  # type: ignore
        logger.debug(
            f"Glob identified {glob_files} as matches to from the compute data folder: {copy_from}"
        )

        data_catalogs = []
        run_log_store = self._context.run_log_store
        for file in glob_files:
            if file.is_dir():
                # Need not add a data catalog for the folder
                continue

            relative_file_path = file.relative_to(".")

            data_catalog = run_log_store.create_data_catalog(str(relative_file_path))
            data_catalog.catalog_handler_location = catalog_location
            data_catalog.catalog_relative_path = (
                run_id + os.sep + str(relative_file_path)
            )
            data_catalog.data_hash = utils.get_data_hash(str(file))
            data_catalog.stage = "put"
            data_catalogs.append(data_catalog)

            if is_catalog_out_of_sync(data_catalog, synced_catalogs):
                logger.info(f"{data_catalog.name} was found to be changed, syncing")

                # Make the directory in the catalog if required
                Path(run_catalog / relative_file_path.parent).mkdir(
                    parents=True, exist_ok=True
                )
                shutil.copy(file, run_catalog / relative_file_path)
            else:
                logger.info(
                    f"{data_catalog.name} was found to be unchanged, ignoring syncing"
                )

        if not data_catalogs:
            raise Exception(f"Did not find any files matching {name} in {copy_from}")

        return data_catalogs

    def sync_between_runs(self, previous_run_id: str, run_id: str):
        """
        Given the previous run id, sync the catalogs between the current one and previous

        Args:
            previous_run_id (str): The previous run id to sync the catalogs from
            run_id (str): The run_id to which the data catalogs should be synced to.

        Raises:
            Exception: If the previous run log does not exist in the catalog

        """
        logger.info(
            f"Using the {self.service_name} catalog and syncing catalogs"
            "between old: {previous_run_id} to new: {run_id}"
        )

        catalog_location = Path(self.get_catalog_location())
        run_catalog = catalog_location / run_id
        utils.safe_make_dir(run_catalog)

        if not utils.does_dir_exist(catalog_location / previous_run_id):
            msg = (
                f"Catalogs from previous run : {previous_run_id} are not found.\n"
                "Note: Please provision the catalog objects generated by previous run in the same catalog location"
                " as the current run, even if the catalog handler for the previous run was different"
            )
            raise Exception(msg)

        cataloged_files = list((catalog_location / previous_run_id).glob("*"))

        for cataloged_file in cataloged_files:
            if str(cataloged_file).endswith("execution.log"):
                continue

            if cataloged_file.is_file():
                shutil.copy(cataloged_file, run_catalog / cataloged_file.name)
            else:
                shutil.copytree(cataloged_file, run_catalog / cataloged_file.name)
            logger.info(f"Copied file from: {cataloged_file} to {run_catalog}")
