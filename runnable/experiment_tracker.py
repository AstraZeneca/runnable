import contextlib
import json
import logging
import os
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, ContextManager, Dict, Tuple, Union

from pydantic import BaseModel, ConfigDict

import runnable.context as context
from runnable import defaults
from runnable.utils import remove_prefix

logger = logging.getLogger(defaults.LOGGER_NAME)


def retrieve_step_details(key: str) -> Tuple[str, int]:
    key = remove_prefix(key, defaults.TRACK_PREFIX)
    data = key.split(defaults.STEP_INDICATOR)

    key = data[0].lower()
    step = 0

    if len(data) > 1:
        step = int(data[1])

    return key, step


def get_tracked_data() -> Dict[str, Any]:
    tracked_data: Dict[str, Any] = defaultdict(dict)
    for env_var, value in os.environ.items():
        if env_var.startswith(defaults.TRACK_PREFIX):
            key, step = retrieve_step_details(env_var)

            # print(value, type(value))
            try:
                value = json.loads(value)
            except json.decoder.JSONDecodeError:
                logger.warning(f"Tracker {key} could not be JSON decoded, adding the literal value")

            tracked_data[key][step] = value
            del os.environ[env_var]

    for key, value in tracked_data.items():
        if len(value) == 1:
            tracked_data[key] = value[0]

    return tracked_data


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

    def publish_data(self, tracked_data: Dict[str, Any]):
        for key, value in tracked_data.items():
            if isinstance(value, dict):
                for key2, value2 in value.items():
                    self.log_metric(key, value2, step=key2)
                continue
            self.log_metric(key, value)

    @abstractmethod
    def log_metric(self, key: str, value: Union[int, float], step: int = 0):
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

    service_name: str = "do-nothing"

    def log_metric(self, key: str, value: Union[int, float], step: int = 0):
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
