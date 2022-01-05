
import logging
from pathlib import Path
import shutil
import os


from magnus import defaults
from magnus import utils


logger = logging.getLogger(defaults.NAME)


def get_run_log_store():
    """
    This method should be called after the executor module has been populated with all the systems.

    This method retrieves the run log store from the global executor.

    Returns:
        object: The run log store
    """
    from magnus.pipeline import global_executor  # pylint: disable=import-outside-toplevel
    return global_executor.run_log_store


def is_catalog_out_of_sync(catalog, synced_catalogs=None) -> bool:
    """
    Check if the catalog items are out of sync from alredy cataloged objects.
    If they are, return False.
    If the object does not exist or synced catalog does not exist, return True
    """
    if not synced_catalogs:
        synced_catalogs = []
    for synced_catalog in synced_catalogs:
        if synced_catalog.catalog_relative_path == catalog.catalog_relative_path:
            if synced_catalog.data_hash == catalog.data_hash:
                return False
            return True
    return True


class BaseCatalog:
    """
    A Base Catalog handler class

    All implementations of the catalog handler should inherit and extend this class.

    As a general design guideline, do not instantiate anything from config as they might change.
    Instead have getters methods to get variables from the config.
    """
    service_name = None

    def __init__(self, config, **kwargs):  # pylint: disable=unused-argument
        self.config = config or {}
        self.compute_data_folder = self.config.get('compute_data_folder', defaults.COMPUTE_DATA_FOLDER)

    def get(self, name: str, run_id: str, compute_data_folder=None, **kwargs):  # pylint: disable=unused-argument
        """
        Get the catalog item by 'name' for the run id, 'run_id' and store it in compute data folder

        Args:
            name (str): The name of the catalog item
            run_id (str): The run_id of the run.
            compute_data_folder (str, optional): The compute data folder. Defaults to magnus default (data/)

        Raises:
            NotImplementedError: Base class, hence not implemented
        """
        raise NotImplementedError

    def put(self, name: str, run_id: str, compute_data_folder=None, synced_catalogs=None, **kwargs):
        # pylint: disable=unused-argument
        """
        Put the file by 'name' from the 'compute_data_folder' in the catalog for the run_id.

        If previous syncing has happened and the file has not been changed, we do not sync again.

        Args:
            name (str): [description]
            run_id (str): [description]
            compute_data_folder (str, optional): The compute data folder. Defaults to magnus default (data/)
            synced_catalogs (dict, optional): Any previously synced catalogs. Defaults to None.

        Raises:
            NotImplementedError: Base class, hence not implemented
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


def get_catalog_handler(config: dict) -> BaseCatalog:
    """
    Get the implementation of the catalog handler as per the config.

    We give a file-system catalog if none is provided

    Args:
        config (dict): The implemented catalog handler class

    Raises:
        Exception: If config is provided with no type in defined.

    Returns:
        BaseCatalog: An implemented catalog handler
    """
    if config:
        catalog_type = config.get('type', None)
        if not catalog_type:
            raise Exception('Catalog type is necessary in catalog config')

        logger.info(f'Trying to get a Catalog handler of type: {catalog_type}')
        for sub_class in BaseCatalog.__subclasses__():
            if catalog_type == sub_class.service_name:
                return sub_class(config.get('config', {}))

    logger.warning('Getting a File System catalog type')
    return FileSystemCatalog(config={})


class DummyCatalog(BaseCatalog):
    """
    A Catalog handler that does nothing.

    Example config:

    catalog:
      type: dummy

    """
    service_name = 'dummy'

    def get(self, name: str, run_id: str, compute_data_folder=None, **kwargs):
        """ Get the file by matching glob pattern to the name

        Args:
            name ([str]): A glob matching the file name
            run_id ([str]): The run id

        Raises:
            Exception: If the catalog location does not exist
        """
        ...

    def put(self, name: str, run_id: str, compute_data_folder=None, synced_catalogs=None, **kwargs):
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
            [type]: [description]
        """
        ...

    def sync_between_runs(self, previous_run_id: str, run_id: str):
        """
        Given the previous run id, sync the catalogs between the current one and previous

        Args:
            previous_run_id (str): The previous run id to sync the catalogs from
            run_id (str): The run_id to which the data catalogs should be synced to.

        Raises:
            Exception: If the previous run log does not exist in the catalog

        """
        ...


class FileSystemCatalog(BaseCatalog):
    """
    A Catalog handler that uses the local file system for cataloging.

    Do not use this catalog if the compute is not local.

    Example config:

    catalog:
      type: file-system
      config:
        catalog_location: The location to store the catalog.
        compute_data_folder: The folder to source the data from.

    """
    # TODO think of compression algorithms to save disk space, ZipFile
    # TODO add map_variable as glob pattern
    # TODO: consider run_id as a glob pattern
    service_name = 'file-system'

    def get_catalog_location(self):
        """
        Get the catalog location from the config.
        If its not defined, use the magnus default

        Returns:
            str: The catalog location
        """
        catalog_location = self.config.get('catalog_location', defaults.CATALOG_LOCATION_FOLDER)

        return catalog_location

    def get(self, name: str, run_id: str, compute_data_folder=None, **kwargs):
        """ Get the file by matching glob pattern to the name

        Args:
            name ([str]): A glob matching the file name
            run_id ([str]): The run id

        Raises:
            Exception: If the catalog location does not exist
        """
        copy_to = self.compute_data_folder
        if compute_data_folder:
            copy_to = compute_data_folder

        catalog_location = self.get_catalog_location()
        utils.safe_make_dir(copy_to)
        run_catalog = Path(catalog_location) / run_id
        # Iterate through the contents of the run_catalog and copy the files that fit the name pattern
        # We should also return a list of data hashes
        if not utils.does_dir_exist(run_catalog):
            raise Exception(f'Expected Catalog to be present at: {run_catalog} but not found')

        glob_files = run_catalog.glob(name)
        data_catalogs = []
        run_log_store = get_run_log_store()
        for file in glob_files:
            data_catalog = run_log_store.create_data_catalog(str(file.name))
            data_catalog.catalog_handler_location = catalog_location
            data_catalog.catalog_relative_path = utils.remove_prefix(str(file), catalog_location)
            data_catalog.data_hash = utils.get_data_hash(str(file))
            data_catalog.stage = 'get'
            data_catalogs.append(data_catalog)

            shutil.copy(file, copy_to)

        return data_catalogs

    def put(self, name: str, run_id: str, compute_data_folder=None, synced_catalogs=None, **kwargs):
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
            [type]: [description]
        """
        copy_from = self.compute_data_folder
        if compute_data_folder:
            copy_from = compute_data_folder

        catalog_location = self.get_catalog_location()
        run_catalog = Path(catalog_location) / run_id
        utils.safe_make_dir(run_catalog)
        # Iterate through the contents of copy_from and if the name matches, we move them to the run_catalog
        # We should also return a list of datastore.DataCatalog items
        if not utils.does_dir_exist(copy_from):
            raise Exception(f'Expected compute data folder to be present at: {compute_data_folder} but not found')

        glob_files = Path(copy_from).glob(name)
        data_catalogs = []
        run_log_store = get_run_log_store()
        for file in glob_files:
            data_catalog = run_log_store.create_data_catalog(str(file.name))
            data_catalog.catalog_handler_location = catalog_location
            data_catalog.catalog_relative_path = run_id + os.sep + str(file.name)
            data_catalog.data_hash = utils.get_data_hash(str(file))
            data_catalog.stage = 'put'
            data_catalogs.append(data_catalog)

            if is_catalog_out_of_sync(data_catalog, synced_catalogs):
                logger.info(f'{data_catalog.name} was found to be changed, syncing')
                shutil.copy(file, run_catalog)
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
        catalog_location = Path(self.get_catalog_location())
        run_catalog = catalog_location / run_id
        utils.safe_make_dir(run_catalog)

        if not utils.does_dir_exist(catalog_location / previous_run_id):
            raise Exception(f'Previous run log by run_id: {previous_run_id} is not found.')

        cataloged_files = (catalog_location / previous_run_id).glob('**/**')

        for cataloged_file in cataloged_files:
            shutil.copy(cataloged_file, run_catalog)
