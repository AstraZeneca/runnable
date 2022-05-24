import logging
import os
import shutil
from pathlib import Path
from typing import List

from magnus import defaults, utils

logger = logging.getLogger(defaults.NAME)


def get_run_log_store():
    """
    This method should be called after the executor module has been populated with all the systems.

    This method retrieves the run log store from the global executor.

    Returns:
        object: The run log store
    """
    from magnus.pipeline import \
        global_executor  # pylint: disable=import-outside-toplevel
    return global_executor.run_log_store


def is_catalog_out_of_sync(catalog, synced_catalogs=None) -> bool:
    """
    Check if the catalog items are out of sync from already cataloged objects.
    If they are, return False.
    If the object does not exist or synced catalog does not exist, return True
    """
    if not synced_catalogs:
        return True  # If nothing has been synced in the past

    for synced_catalog in synced_catalogs:
        if synced_catalog.catalog_relative_path == catalog.catalog_relative_path:
            if synced_catalog.data_hash == catalog.data_hash:
                return False
            return True

    return True  # The object does not exist, sync it


class BaseCatalog:
    """
    Base Catalog class definition.

    All implementations of the catalog handler should inherit and extend this class.

    Note: As a general guideline, do not extract anything from the config to set class level attributes.
          Integration patterns modify the config after init to change behaviors.
          Access config properties using getters/property of the class.
    """
    service_name = ''

    def __init__(self, config, **kwargs):  # pylint: disable=unused-argument
        self.config = config or {}

    @property
    def compute_data_folder(self) -> str:
        """
        Returns the compute data folder defined as per the config of the catalog.

        Returns:
            [str]: The compute data folder as defined or defaults to magnus default 'data/'
        """
        return self.config.get('compute_data_folder', defaults.COMPUTE_DATA_FOLDER)

    def get(self, name: str, run_id: str, compute_data_folder=None, **kwargs) -> List[object]:
        # pylint: disable=unused-argument
        """
        Get the catalog item by 'name' for the 'run id' and store it in compute data folder.

        The catalog location should have been created before you can get from it.

        Args:
            name (str): The name of the catalog item
            run_id (str): The run_id of the run.
            compute_data_folder (str, optional): The compute data folder. Defaults to magnus default (data/)

        Raises:
            NotImplementedError: Base class, hence not implemented

        Returns:
            List(object) : A list of catalog objects
        """
        raise NotImplementedError

    def put(self, name: str, run_id: str, compute_data_folder=None, synced_catalogs=None, **kwargs) -> List[object]:
        # pylint: disable=unused-argument
        """
        Put the file by 'name' from the 'compute_data_folder' in the catalog for the run_id.

        If previous syncing has happened and the file has not been changed, we do not sync again.

        Args:
            name (str): The name of the catalog item.
            run_id (str): The run_id of the run.
            compute_data_folder (str, optional): The compute data folder. Defaults to magnus default (data/)
            synced_catalogs (dict, optional): Any previously synced catalogs. Defaults to None.

        Raises:
            NotImplementedError: Base class, hence not implemented

        Returns:
            List(object) : A list of catalog objects
        """
        raise NotImplementedError

    def sync_between_runs(self, previous_run_id: str, run_id: str):
        """
        Given run_id of a previous run, sync them to the catalog of the run given by run_id

        Args:
            previous_run_id (str): The run id of the previous run
            run_id (str): The run_id to which the data catalogs should be synced to.

        Raises:
            NotImplementedError: Base class, hence not implemented
        """
        raise NotImplementedError


class DoNothingCatalog(BaseCatalog):
    """
    A Catalog handler that does nothing.

    Example config:

    catalog:
      type: do-nothing

    """
    service_name = 'do-nothing'

    def get(self, name: str, run_id: str, compute_data_folder=None, **kwargs) -> List[object]:
        """
        Does nothing
        """
        logger.info('Using a do-nothing catalog, doing nothing in get')
        return []

    def put(self, name: str, run_id: str, compute_data_folder=None, synced_catalogs=None, **kwargs) -> List[object]:
        """
        Does nothing
        """
        logger.info('Using a do-nothing catalog, doing nothing in put')
        return []

    def sync_between_runs(self, previous_run_id: str, run_id: str):
        """
        Does nothing
        """
        logger.info('Using a do-nothing catalog, doing nothing while sync between runs')
        ...


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
    # TODO think of compression algorithms to save disk space, ZipFile
    service_name = 'file-system'

    def get_catalog_location(self) -> str:
        """
        Get the catalog location from the config.
        If its not defined, use the magnus default

        Returns:
            str: The catalog location as defined by the config or magnus default '.catalog'
        """
        if self.config:
            return self.config.get('catalog_location', defaults.CATALOG_LOCATION_FOLDER)

        return defaults.CATALOG_LOCATION_FOLDER

    def get(self, name: str, run_id: str, compute_data_folder=None, **kwargs) -> List[object]:
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
        logger.info(f'Using the {self.service_name} catalog and trying to get {name} for run_id: {run_id}')

        copy_to = self.compute_data_folder
        if compute_data_folder:
            copy_to = compute_data_folder

        copy_to = Path(copy_to)

        if not utils.does_dir_exist(copy_to):
            msg = (
                f'Expected compute data folder to be present at: {copy_to} but not found. \n'
                'Note: Magnus does not create the compute data folder for you. Please ensure that the folder exists.\n'
            )
            raise Exception(msg)

        catalog_location = self.get_catalog_location()
        run_catalog = Path(catalog_location) / run_id

        logger.debug(f'Copying objects to {copy_to} from the run catalog location of {run_catalog}')

        if not utils.does_dir_exist(run_catalog):
            msg = (
                f'Expected Catalog to be present at: {run_catalog} but not found.\n'
                'Note: Please make sure that some data was put in the catalog before trying to get from it.\n'
            )
            raise Exception(msg)

        # Iterate through the contents of the run_catalog and copy the files that fit the name pattern
        # We should also return a list of data hashes
        glob_files = run_catalog.glob(name)
        logger.debug(f'Glob identified {glob_files} as matches to from the catalog location: {run_catalog}')

        data_catalogs = []
        run_log_store = get_run_log_store()
        for file in glob_files:
            if file.is_dir():
                # Need not add a data catalog for the folder
                continue

            relative_file_path = file.relative_to(run_catalog)

            data_catalog = run_log_store.create_data_catalog(str(relative_file_path))
            data_catalog.catalog_handler_location = catalog_location
            data_catalog.catalog_relative_path = str(relative_file_path)
            data_catalog.data_hash = utils.get_data_hash(str(file))
            data_catalog.stage = 'get'
            data_catalogs.append(data_catalog)

            # Make the directory in the data folder if required
            Path(copy_to / relative_file_path.parent).mkdir(parents=True, exist_ok=True)
            shutil.copy(file, copy_to / relative_file_path)

            logger.info(f'Copied {file} from {run_catalog} to {copy_to}')

        return data_catalogs

    def put(self, name: str, run_id: str, compute_data_folder=None, synced_catalogs=None, **kwargs) -> List[object]:
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
        logger.info(f'Using the {self.service_name} catalog and trying to put {name} for run_id: {run_id}')

        copy_from = self.compute_data_folder
        if compute_data_folder:
            copy_from = compute_data_folder
        copy_from = Path(copy_from)

        catalog_location = self.get_catalog_location()
        run_catalog = Path(catalog_location) / run_id
        utils.safe_make_dir(run_catalog)

        logger.debug(f'Copying objects from {copy_from} to the run catalog location of {run_catalog}')

        if not utils.does_dir_exist(copy_from):
            msg = (
                f'Expected compute data folder to be present at: {compute_data_folder} but not found. \n'
                'Note: Magnus does not create the compute data folder for you. Please ensure that the folder exists.\n'
            )
            raise Exception(msg)

        # Iterate through the contents of copy_from and if the name matches, we move them to the run_catalog
        # We should also return a list of datastore.DataCatalog items

        glob_files = copy_from.glob(name)
        logger.debug(f'Glob identified {glob_files} as matches to from the compute data folder: {copy_from}')

        data_catalogs = []
        run_log_store = get_run_log_store()
        for file in glob_files:
            if file.is_dir():
                # Need not add a data catalog for the folder
                continue

            relative_file_path = file.relative_to('.')

            data_catalog = run_log_store.create_data_catalog(str(relative_file_path))
            data_catalog.catalog_handler_location = catalog_location
            data_catalog.catalog_relative_path = run_id + os.sep + str(relative_file_path)
            data_catalog.data_hash = utils.get_data_hash(str(file))
            data_catalog.stage = 'put'
            data_catalogs.append(data_catalog)

            if is_catalog_out_of_sync(data_catalog, synced_catalogs):
                logger.info(f'{data_catalog.name} was found to be changed, syncing')

                # Make the directory in the catalog if required
                Path(run_catalog / relative_file_path.parent).mkdir(parents=True, exist_ok=True)
                shutil.copy(file, run_catalog / relative_file_path)
            else:
                logger.info(f'{data_catalog.name} was found to be unchanged, ignoring syncing')
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
        logger.info(f'Using the {self.service_name} catalog and syncing catalogs'
                    'between old: {previous_run_id} to new: {run_id}')

        catalog_location = Path(self.get_catalog_location())
        run_catalog = catalog_location / run_id
        utils.safe_make_dir(run_catalog)

        if not utils.does_dir_exist(catalog_location / previous_run_id):
            msg = (
                f'Catalogs from previous run : {previous_run_id} are not found.\n'
                'Note: Please provision the catalog objects generated by previous run in the same catalog location'
                ' as the current run, even if the catalog handler for the previous run was different')
            raise Exception(msg)

        cataloged_files = (catalog_location / previous_run_id).glob('**/**')

        for cataloged_file in cataloged_files:
            shutil.copy(cataloged_file, run_catalog)
            logger.info(f'Copied file from: {cataloged_file} to {run_catalog}')
