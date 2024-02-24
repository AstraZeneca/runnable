import functools
import logging
from typing import Any, Union

from pydantic import ConfigDict, PrivateAttr

from runnable import defaults
from runnable.experiment_tracker import BaseExperimentTracker

logger = logging.getLogger(defaults.NAME)


class MLFlowExperimentTracker(BaseExperimentTracker):
    """
    A MLFlow experiment tracker.

    TODO: Need to set up credentials from secrets
    """

    service_name: str = "mlflow"

    server_url: str
    autolog: bool = False

    _default_experiment_name: str = PrivateAttr(default="Default")
    _active_run_id: str = PrivateAttr(default="")
    _client: Any = PrivateAttr(default=None)

    model_config = ConfigDict(extra="forbid")

    def model_post_init(self, __context: Any) -> None:
        try:
            import mlflow
        except ImportError:
            raise Exception("You need to install mlflow to use MLFlowExperimentTracker.")

        self._client = mlflow

        self._client.set_tracking_uri(self.server_url)

        if self.autolog:
            self._client.autolog(log_models=False)

    @functools.cached_property
    def experiment_id(self):
        experiment_name = self._default_experiment_name

        # If a tag is provided, we should create that as our experiment
        if self._context.tag:
            experiment_name = self._context.tag

        experiment = self._client.get_experiment_by_name(experiment_name)
        if not experiment:
            # Create the experiment and get it.
            experiment = self._client.create_experiment(experiment_name)
            experiment = self._client.get_experiment(experiment)

        return experiment.experiment_id

    @functools.cached_property
    def run_name(self):
        return self._context.run_id

    @property
    def client_context(self):
        if self._active_run_id:
            return self._client.start_run(
                run_id=self._active_run_id, experiment_id=self.experiment_id, run_name=self.run_name
            )

        active_run = self._client.start_run(run_name=self.run_name, experiment_id=self.experiment_id)
        self._active_run_id = active_run.info.run_id
        return active_run

    def log_metric(self, key: str, value: Union[int, float], step: int = 0):
        """
        Sets the metric in the experiment tracking.

        Args:
            key (str): The key against you want to store the value
            value (Any): The value of the metric
        """
        if not isinstance(value, float) or isinstance(value, int):
            msg = f"Only float/int values are accepted as metrics. Setting the metric {key} as parameter {key}_{step}"
            logger.warning(msg)
            self.log_parameter(key=key, value=value, step=step)
            return

        with self.client_context as _:
            self._client.log_metric(key, float(value), step=step or None)

    def log_parameter(self, key: str, value: Any, step: int = 0):
        with self.client_context as _:
            self._client.log_param(key + f"_{str(step)}", value)
