from typing import List, Optional

from runnable.datastore import DataCatalog


def is_catalog_out_of_sync(catalog, synced_catalogs=Optional[List[DataCatalog]]) -> bool:
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
