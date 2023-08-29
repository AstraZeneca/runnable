import contextlib
import logging
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

from magnus import defaults

logger = logging.getLogger(defaults.NAME)

# --8<-- [start:docs]


class BaseExperimentTracker(ABC):
    """
    Base Experiment tracker class definition.
    """

    service_name: str = ""
    service_type: str = "experiment_tracker"

    class Config(BaseModel):
        ...

    @property
    def client_context(self) -> Any:
        """
        Returns the client context.
        """
        return contextlib.nullcontext()

    @abstractmethod
    def log_metric(self, key: str, value: float, step: int = 0):
        """
        Sets the metric in the experiment tracking.

        Args:
            key (str): The key against you want to store the value
            value (float): The value of the metric
            step (int): Optional step at which it was recorded

        Raises:
            NotImplementedError: Base class, hence not implemented
        """
        raise NotImplementedError

    @abstractmethod
    def log_parameter(self, key: str, value: Any):
        """
        Logs a parameter in the experiment tracking.

        Args:
            key (str): The key against you want to store the value
            value (any): The value of the metric

        Raises:
            NotImplementedError: Base class, hence not implemented
        """
        pass


# --8<-- [end:docs]


class DoNothingTracker(BaseExperimentTracker):
    """
    A Do nothing tracker
    """

    service_name = "do-nothing"

    class ContextConfig(BaseExperimentTracker.Config):
        ...

    def __init__(self, config: dict) -> None:
        self.config = self.ContextConfig(**(config or {}))

    def log_metric(self, key: str, value: float, step: int = 0):
        """
        Sets the metric in the experiment tracking.

        Args:
            key (str): The key against you want to store the value
            value (float): The value of the metric
        """
        pass

    def log_parameter(self, key: str, value: Any):
        """
        Since this is a Do nothing tracker, we don't need to log anything.
        """
        pass
