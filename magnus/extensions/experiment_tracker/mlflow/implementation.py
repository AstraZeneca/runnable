import functools
import logging
from typing import Any

from pydantic import Extra

from magnus import defaults
from magnus.experiment_tracker import BaseExperimentTracker

logger = logging.getLogger(defaults.NAME)

try:
    import mlflow
except ImportError as _e:
    msg = "You need to install mlflow to use MLFlowExperimentTracker. " "Try `pip install mlflow-skinny`."
    raise Exception(msg) from _e


class MLFlowExperimentTracker(BaseExperimentTracker):
    """
    A MLFlow experiment tracker.

    TODO: Need to set up credentials from secrets
    """

    service_name: str = "mlflow"
    server_url: str
    autolog: bool = False
    _default_experiment_name: str = "Default"
    _active_run_id: str = ""

    class Config:
        extra = Extra.forbid
        underscore_attrs_are_private = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        mlflow.set_tracking_uri(self.server_url)

        if self.autolog:
            mlflow.autolog(log_models=False)

    @functools.cached_property
    def experiment_id(self):
        from magnus import context

        experiment_name = self._default_experiment_name

        # If a tag is provided, we should create that as our experiment
        if context.tag:
            experiment_name = context.executor.tag

        experiment = mlflow.get_experiment_by_name(experiment_name)
        if not experiment:
            # Create the experiment and get it.
            experiment = mlflow.create_experiment(experiment_name)
            experiment = mlflow.get_experiment(experiment)

        return experiment.experiment_id

    @functools.cached_property
    def run_name(self):
        from magnus import context

        return context.executor.run_id

    @property
    def client_context(self):
        if self._active_run_id:
            return mlflow.start_run(
                run_id=self._active_run_id, experiment_id=self.experiment_id, run_name=self.run_name
            )

        active_run = mlflow.start_run(run_name=self.run_name, experiment_id=self.experiment_id)
        self._active_run_id = active_run.info.run_id
        return active_run

    def log_metric(self, key: str, value: float, step: int = 0):
        """
        Sets the metric in the experiment tracking.

        Args:
            key (str): The key against you want to store the value
            value (Any): The value of the metric
        """
        if not isinstance(value, float):
            msg = f"Only float values are accepted as metrics. Setting the metric {key} as parameter {key}_{step}"
            logger.warning(msg)
            self.log_parameter(key=key + f"_{step}", value=value)
            return

        with self.client_context as _:
            mlflow.log_metric(key, value, step=step or None)

    def log_parameter(self, key: str, value: Any):
        with self.client_context as _:
            mlflow.log_param(key, value)
