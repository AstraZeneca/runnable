import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict

import runnable.context as context
from runnable import defaults
from runnable.datastore import DataCatalog

logger = logging.getLogger(defaults.LOGGER_NAME)


def is_catalog_out_of_sync(
    catalog, synced_catalogs=Optional[List[DataCatalog]]
) -> bool:
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


# --8<-- [start:docs]


class BaseCatalog(ABC, BaseModel):
    """
    Base Catalog class definition.

    All implementations of the catalog handler should inherit and extend this class.
    """

    service_name: str = ""
    service_type: str = "catalog"
    model_config = ConfigDict(extra="forbid")

    @abstractmethod
    def get_summary(self) -> Dict[str, Any]: ...

    @property
    def _context(self):
        return context.run_context

    @property
    def compute_data_folder(self) -> str:
        return defaults.COMPUTE_DATA_FOLDER

    @abstractmethod
    def get(
        self, name: str, run_id: str, compute_data_folder: str = "", **kwargs
    ) -> List[DataCatalog]:
        """
        Get the catalog item by 'name' for the 'run id' and store it in compute data folder.

        The catalog location should have been created before you can get from it.

        Args:
            name (str): The name of the catalog item
            run_id (str): The run_id of the run.
            compute_data_folder (str, optional): The compute data folder. Defaults to runnable default (data/)

        Raises:
            NotImplementedError: Base class, hence not implemented

        Returns:
            List(object) : A list of catalog objects
        """
        raise NotImplementedError

    @abstractmethod
    def put(
        self,
        name: str,
        run_id: str,
        compute_data_folder: str = "",
        synced_catalogs: Optional[List[DataCatalog]] = None,
        **kwargs,
    ) -> List[DataCatalog]:
        """
        Put the file by 'name' from the 'compute_data_folder' in the catalog for the run_id.

        If previous syncing has happened and the file has not been changed, we do not sync again.

        Args:
            name (str): The name of the catalog item.
            run_id (str): The run_id of the run.
            compute_data_folder (str, optional): The compute data folder. Defaults to runnable default (data/)
            synced_catalogs (dict, optional): Any previously synced catalogs. Defaults to None.

        Raises:
            NotImplementedError: Base class, hence not implemented

        Returns:
            List(object) : A list of catalog objects
        """
        raise NotImplementedError

    @abstractmethod
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


# --8<-- [end:docs]


class DoNothingCatalog(BaseCatalog):
    """
    A Catalog handler that does nothing.

    Example config:

    catalog:
      type: do-nothing

    """

    service_name: str = "do-nothing"

    def get_summary(self) -> Dict[str, Any]:
        return {}

    def get(
        self, name: str, run_id: str, compute_data_folder: str = "", **kwargs
    ) -> List[DataCatalog]:
        """
        Does nothing
        """
        logger.info("Using a do-nothing catalog, doing nothing in get")
        return []

    def put(
        self,
        name: str,
        run_id: str,
        compute_data_folder: str = "",
        synced_catalogs: Optional[List[DataCatalog]] = None,
        **kwargs,
    ) -> List[DataCatalog]:
        """
        Does nothing
        """
        logger.info("Using a do-nothing catalog, doing nothing in put")
        return []

    def sync_between_runs(self, previous_run_id: str, run_id: str):
        """
        Does nothing
        """
        logger.info("Using a do-nothing catalog, doing nothing while sync between runs")
        logger.info("Using a do-nothing catalog, doing nothing while sync between runs")
