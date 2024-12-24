from abc import ABC, abstractmethod
from typing import Any

import dill as pickle
from pydantic import BaseModel, ConfigDict

import runnable.context as context


class BasePickler(ABC, BaseModel):
    """
    The base class for all pickler.

    We are still in the process of hardening the design of this class.
    For now, we are just going to use pickle.
    """

    extension: str = ""
    service_name: str = ""
    service_type: str = "pickler"
    model_config = ConfigDict(extra="forbid")

    @property
    def _context(self):
        return context.run_context

    @abstractmethod
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

    @abstractmethod
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

    extension: str = ".dill"
    service_name: str = "pickle"

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
