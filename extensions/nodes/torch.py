import importlib
import logging
import os
from datetime import datetime
from typing import Any, Callable

from pydantic import ConfigDict, Field

from extensions.nodes.torch_config import TorchConfig
from runnable import PythonJob, datastore, defaults
from runnable.datastore import StepLog
from runnable.nodes import DistributedNode
from runnable.tasks import PythonTaskType, create_task
from runnable.utils import TypeMapVariable

logger = logging.getLogger(defaults.LOGGER_NAME)

try:
    from torch.distributed.launcher.api import LaunchConfig, elastic_launch
    from torch.distributed.run import config_from_args
except ImportError:
    raise ImportError("torch is not installed. Please install torch first.")

print("torch is installed")


def training_subprocess():
    command = os.environ.get("RUNNABLE_TORCH_COMMAND")
    run_id = os.environ.get("RUNNABLE_TORCH_RUN_ID", "")
    parameters_files = os.environ.get("RUNNABLE_TORCH_PARAMETERS_FILES", "")
    process_run_id = run_id + "-" + os.environ.get("RANK", "")

    delete_env_vars_with_prefix("RUNNABLE_")

    func = get_callable_from_dotted_path(command)
    job = PythonJob(function=func)

    job.execute(
        parameters_file=parameters_files,
        job_id=process_run_id,
    )


def get_callable_from_dotted_path(dotted_path) -> Callable:
    try:
        # Split the path into module path and callable object
        module_path, callable_name = dotted_path.rsplit(".", 1)

        # Import the module
        module = importlib.import_module(module_path)

        # Get the callable from the module
        callable_obj = getattr(module, callable_name)

        # Check if the object is callable
        if not callable(callable_obj):
            raise TypeError(f"The object {callable_name} is not callable.")

        return callable_obj

    except (ImportError, AttributeError, ValueError) as e:
        raise ImportError(f"Could not import '{dotted_path}'.") from e


def delete_env_vars_with_prefix(prefix):
    to_delete = []  # List to keep track of variables to delete

    # Iterate over a list of all environment variable keys
    for var in os.environ:
        if var.startswith(prefix):
            to_delete.append(var)

    # Delete each of the variables collected
    for var in to_delete:
        del os.environ[var]


class TorchNode(DistributedNode, TorchConfig):
    node_type: str = Field(default="torch", serialization_alias="type")
    executable: PythonTaskType = Field(exclude=True)

    # Similar to TaskNode
    model_config = ConfigDict(extra="allow")

    def get_summary(self) -> dict[str, Any]:
        summary = {
            "name": self.name,
            "type": self.node_type,
        }

        return summary

    @classmethod
    def parse_from_config(cls, config: dict[str, Any]) -> "TorchNode":
        task_config = {
            k: v for k, v in config.items() if k not in TorchNode.model_fields.keys()
        }
        node_config = {
            k: v for k, v in config.items() if k in TorchNode.model_fields.keys()
        }

        executable = create_task(task_config)

        assert isinstance(executable, PythonTaskType)
        return cls(executable=executable, **node_config, **task_config)

    def get_launch_config(self) -> LaunchConfig:
        config, _, _ = config_from_args(self)
        config.run_id = self._context.run_id
        return config

    def execute(
        self,
        mock=False,
        map_variable: TypeMapVariable = None,
        attempt_number: int = 1,
    ) -> StepLog:
        assert (
            map_variable is None or not map_variable
        ), "TorchNode does not support map_variable"

        step_log = self._context.run_log_store.get_step_log(
            self._get_step_log_name(map_variable), self._context.run_id
        )

        # Attempt to call the function or elastic launch
        launch_config = self.get_launch_config()
        logger.info(f"launch_config: {launch_config}")

        os.environ["RUNNABLE_TORCH_COMMAND"] = self.executable.command
        os.environ["RUNNABLE_TORCH_PARAMETERS_FILES"] = (
            self._context.parameters_file or ""
        )
        os.environ["RUNNABLE_TORCH_RUN_ID"] = self._context.run_id
        # retrieve the master address and port from the parameters
        # default to localhost and 29500
        launcher = elastic_launch(
            launch_config,
            training_subprocess,
        )
        try:
            launcher()
            attempt_log = datastore.StepAttempt(
                status=defaults.SUCCESS,
                start_time=str(datetime.now()),
                end_time=str(datetime.now()),
                attempt_number=attempt_number,
            )
        except Exception as e:
            attempt_log = datastore.StepAttempt(
                status=defaults.FAIL,
                start_time=str(datetime.now()),
                end_time=str(datetime.now()),
                attempt_number=attempt_number,
            )
            logger.error(f"Error executing TorchNode: {e}")

        delete_env_vars_with_prefix("RUNNABLE_TORCH")

        logger.info(f"attempt_log: {attempt_log}")
        logger.info(f"Step {self.name} completed with status: {attempt_log.status}")

        step_log.status = attempt_log.status
        step_log.attempts.append(attempt_log)

        return step_log

    def fan_in(self, map_variable: dict[str, str | int | float] | None = None):
        # Destroy the service
        # Destroy the statefulset
        assert (
            map_variable is None or not map_variable
        ), "TorchNode does not support map_variable"

    def fan_out(self, map_variable: dict[str, str | int | float] | None = None):
        # Create a service
        # Create a statefulset
        # Gather the IPs and set them as parameters downstream
        assert (
            map_variable is None or not map_variable
        ), "TorchNode does not support map_variable"
