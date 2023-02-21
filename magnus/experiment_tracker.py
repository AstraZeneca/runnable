import contextlib
import functools
import logging
from typing import Any

from pydantic import BaseModel

from magnus import defaults

logger = logging.getLogger(defaults.NAME)


class BaseExperimentTracker:
    """
    Base Experiment tracker class definition.
    """

    service_name = ''

    class Config(BaseModel):
        pass

    def __init__(self, config: dict = None, **kwargs):  # pylint: disable=unused-argument
        config = config or {}
        self.config = self.Config(**config)

    @property
    def client_context(self):
        return contextlib.nullcontext()

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


class DoNothingTracker(BaseExperimentTracker):
    """
    A Do nothing tracker
    """
    service_name = 'do-nothing'

    def set_metric(self, key: str, value: float, step: int = 0):
        """
        Sets the metric in the experiment tracking.

        Args:
            key (str): The key against you want to store the value
            value (float): The value of the metric
        """
        pass

    def log_parameter(self, key: str, value: Any):
        pass


# class MLFlowExperimentTracker(BaseExperimentTracker):
#     """
#     A MLFlow experiment tracker.

#     TODO: Need to set up credentials from secrets
#     """
#     service_name = 'mlflow'

#     class Config(BaseExperimentTracker.Config):
#         server_url: str
#         autolog: bool = False

#     def __init__(self, config, **kwargs):
#         super().__init__(config, **kwargs)
#         self.default_experiment_name = 'Default'
#         self.active_run_id = None

#         mlflow.set_tracking_uri(self.config.server_url)

#         if self.config.autolog:
#             mlflow.autolog(log_models=False)

#     @functools.cached_property
#     def experiment_id(self):
#         from magnus.pipeline import global_executor
#         experiment_name = self.default_experiment_name

#         # If a tag is provided, we should create that as our experiment
#         if global_executor.tag:
#             experiment_name = global_executor.tag

#         experiment = mlflow.get_experiment_by_name(experiment_name)
#         if not experiment:
#             # Create the experiment and get it.
#             experiment = mlflow.create_experiment(experiment_name)
#             experiment = mlflow.get_experiment(experiment)

#         return experiment.experiment_id

#     @functools.cached_property
#     def run_name(self):
#         from magnus.pipeline import global_executor
#         return global_executor.run_id

#     @property
#     def client_context(self):
#         if self.active_run_id:
#             return mlflow.start_run(run_id=self.active_run_id,
#                                     experiment_id=self.experiment_id, run_name=self.run_name)

#         active_run = mlflow.start_run(run_name=self.run_name, experiment_id=self.experiment_id)
#         self.active_run_id = active_run.info.run_id
#         return active_run

#     def set_metric(self, key: str, value: float, step: int = 0):
#         """
#         Sets the metric in the experiment tracking.

#         Args:
#             key (str): The key against you want to store the value
#             value (Any): The value of the metric
#         """
#         if not isinstance(value, float):
#             msg = (
#                 f'Only float values are accepted as metrics. Setting the metric {key} as parameter {key}_{step}'
#             )
#             logger.warning(msg)
#             self.log_parameter(key=key+f'_{step}', value=value)
#             return

#         with self.client_context as _:
#             mlflow.log_metric(key, value, step=step or None)

#     def get_metric(self, key: str) -> Any:
#         """
#         Return the metric by the key

#         Args:
#             key (str): The metric you want to retrieve

#         Returns:
#             Any: The value of the key
#         """
#         with self.client_context as client:
#             metrics = client.data.metrics
#             return metrics.get(key)

#     def log_parameter(self, key: str, value: Any):
#         with self.client_context as _:
#             mlflow.log_param(key, value)

#     def get_parameter(self, key: str) -> Any:
#         pass
