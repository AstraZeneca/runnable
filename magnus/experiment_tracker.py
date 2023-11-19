import contextlib
import logging
from abc import ABC, abstractmethod
from typing import Any, ContextManager

from pydantic import BaseModel, ConfigDict

import magnus.context as context
from magnus import defaults

logger = logging.getLogger(defaults.LOGGER_NAME)

# --8<-- [start:docs]


class BaseExperimentTracker(ABC, BaseModel):
    """
    Base Experiment tracker class definition.
    """

    service_name: str = ""
    service_type: str = "experiment_tracker"

    @property
    def _context(self):
        return context.run_context

    model_config = ConfigDict(extra="forbid")

    @property
    def client_context(self) -> ContextManager:
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

        # TODO: Consider log_artifact


# --8<-- [end:docs]


class DoNothingTracker(BaseExperimentTracker):
    """
    A Do nothing tracker
    """

    service_name: str = "do-nothing"

    def log_metric(self, key: str, value: float, step: int = 0):
        """
        Sets the metric in the experiment tracking.

        Args:
            key (str): The key against you want to store the value
            value (float): The value of the metric
        """
        ...

    def log_parameter(self, key: str, value: Any):
        """
        Since this is a Do nothing tracker, we don't need to log anything.
        """
        ...
