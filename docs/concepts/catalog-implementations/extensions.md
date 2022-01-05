# Extensions

To implement your own Catalog Extension, all you need to extend the the below Base class

```python
class BaseCatalog:
    """
    A Base Catalog handler class

    All implementations of the catalog handler should inherit and extend this class.

    As a general design guideline, do not instantiate anything from config as they might change.
    Instead have getters methods to get variables from the config.
    """
    catalog_type = None

    def __init__(self, config, **kwargs):  # pylint: disable=unused-argument
        self.config = config or {}
        self.compute_data_folder = self.config.get('compute_data_folder', defaults.COMPUTE_DATA_FOLDER)

    def accomodate_executor(self, executor, stage='execution'):  # pylint: disable=unused-argument,no-self-use
        """
        Use this method to change any of the mode (executor) settings.

        Raise an execption if this service provider is not compatible with the compute provider.

        This function would be called twice:

        * During the traversal of the graph with stage='traversal'.
        * During the execution of the node with stage='execution'.

        Most of the times, the method need not do anything and you can simply pass.

        Args:
            executor (magnus.executor.BaseExecutor): The compute mode
            stage (str, optional): The stage at which the function is called. Defaults to 'execution'.
        """
        raise NotImplementedError

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
```

Provide the path to the implementation in magnus config as described here.