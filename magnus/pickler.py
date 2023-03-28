import pickle
from typing import Any


class BasePickler:
    """
    The base class for all picklers.

    We are still in the process of hardening the design of this class.
    For now, we are just going to use pickle.
    """

    extension = ""
    service_name = ""

    def dump(self, data: Any, path: str):
        """
        Dump an object to the specified path.
        The path is the full path.

        To correctly identify the pickler from possible implementations, we use the extension.
        An extension is added automatically, if not provided.

        Args:
            data (Any): The object to pickle
            path (str): The path to save the pickle file

        Raises:
            NotImplementedError: Base class has no implementation
        """
        raise NotImplementedError

    def load(self, path: str) -> Any:
        """
        Load the object from the specified path.

        To correctly identify the pickler from possible implementations, we use the extension.
        An extension is added automatically, if not provided.

        Args:
            path (str): The path to load the pickled file from.

        Raises:
            NotImplementedError: Base class has no implementation.
        """
        raise NotImplementedError


class NativePickler(BasePickler):
    """
    Uses native python pickle to load and dump files
    """

    extension = ".pickle"
    service_name = "pickle"

    def dump(self, data: Any, path: str):
        """
        Dump an object to the specified path.
        The path is the full path.

        Args:
            data (Any): The data to pickle
            path (str): The path to save the pickle file
        """
        if not path.endswith(self.extension):
            path = path + self.extension

        with open(path, "wb") as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    def load(self, path: str) -> Any:
        """
        Load the object from the specified path.

        Args:
            path (str): The path to load the object from.

        Returns:
            Any: The data loaded from the file.
        """
        if not path.endswith(self.extension):
            path = path + self.extension

        data = None
        with open(path, "rb") as f:
            data = pickle.load(f)

        return data
