# Extensions to Secrets

To implement your own extensions to Secrets, all you have to do is extend this Base class.

```python
class BaseSecrets:
    secrets_type = None

    def __init__(self, config: dict, **kwargs):
        self.config = config or {}

    def accomodate_executor(self, executor, stage='execution'):
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
        pass

    def get(self, name: str = None, **kwargs):
        """
        Return the secret by name.
        If no name is give, return all the secrets.

        Args:
            name (str): The name of the secret to return.

        Raises:
            NotImplementedError: Base class and hence not implemented.
        """
        raise NotImplementedError
```