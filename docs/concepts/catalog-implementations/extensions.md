To extend and implement a custom catalog, you need to over-ride the appropriate methods of the ```Base``` class.

Some of the methods of the ```BaseCatalog``` have default implementations and need not be over-written.

Please refer to [*Guide to extensions* ](../../../extensions/extensions/) for a detailed explanation and the need for
implementing a *Integration* pattern along with the extension.

Extensions that are being actively worked on and listed to be released as part of ```magnus-extensions```

- s3 : Using s3 to store a catalog objects

```python
# You can find this in the source code at: magnus/catalog.py along with a few example
# implementations of do-nothing and file-system
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
```
