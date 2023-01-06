import logging
from typing import Any

from magnus import defaults

logger = logging.getLogger(defaults.NAME)


class BaseExperimentTracker:
    """
    Base Experiment tracker class definition.
    """

    service_name = ''

    def __init__(self, config, **kwargs):  # pylint: disable=unused-argument
        self.config = config or {}

    def set_metric(self, key: str, value: Any):
        """
        Sets the metric in the experiment tracking.

        Args:
            key (str): The key against you want to store the value
            value (Any): The value of the metric

        Raises:
            NotImplementedError: Base class, hence not implemented
        """
        raise NotImplementedError

    def get_metric(self, key: str) -> Any:
        """
        Return the metric by the key

        Args:
            key (str): The metric you want to retrieve

        Raises:
            NotImplementedError: Base Class, hence not implemented

        Returns:
            Any: The value of the key
        """
        raise NotImplementedError


class DoNothingTracker(BaseExperimentTracker):
    """
    A Do nothing tracker
    """
    service_name = 'do-nothing'

    def set_metric(self, key: str, value: Any):
        """
        Sets the metric in the experiment tracking.

        Args:
            key (str): The key against you want to store the value
            value (Any): The value of the metric
        """
        pass

    def get_metric(self, key: str) -> Any:
        """
        Return the metric by the key

        Args:
            key (str): The metric you want to retrieve

        Returns:
            Any: The value of the key
        """
        return None
