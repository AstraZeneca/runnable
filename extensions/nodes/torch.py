import importlib
import logging
import os
from datetime import datetime
from typing import Any, Callable

from pydantic import ConfigDict, Field

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


class TorchNode(DistributedNode):
    node_type: str = Field(default="torch", serialization_alias="type")
    executable: PythonTaskType = Field(exclude=True)

    nnodes: str = Field(default="1:1")
    nproc_per_node: int = Field(default=4)

    rdzv_backend: str = Field(default="static")
    rdzv_endpoint: str = Field(default="")
    rdzv_id: str | None = Field(default=None)
    rdzv_conf: str = Field(default="")

    max_restarts: int = Field(default=3)
    monitor_interval: float = Field(default=0.1)
    start_method: str = Field(default="spawn")
    role: str = Field(default="default_role")
    log_dir: str = Field(default="torch_logs")
    redirects: str = Field(default="3")
    tee: str = Field(default="0")
    master_addr: str = Field(default="localhost")
    master_port: str = Field(default="29500")
    training_script: str = Field(default="dummy_training_script")
    training_script_args: str = Field(default="")

    # Optional fields
    local_ranks_filter: str = Field(default="")
    node_rank: int = Field(default=0)
    local_addr: str | None = Field(default=None)
    logs_specs: str | None = Field(default=None)
    standalone: bool = Field(default=False)
    module: bool = Field(default=False)
    no_python: bool = Field(default=False)
    run_path: bool = Field(default=False)

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
        return config

    def execute(
        self,
        mock=False,
        map_variable: TypeMapVariable = None,
        attempt_number: int = 1,
    ) -> StepLog:
        assert map_variable is None, "TorchNode does not support map_variable"
        print("Executing TorchNode")

        step_log = self._context.run_log_store.get_step_log(
            self._get_step_log_name(map_variable), self._context.run_id
        )

        # Attempt to call the function or elastic launch
        launch_config = self.get_launch_config()
        print(f"launch_config: {launch_config}")
        os.environ["RUNNABLE_TORCH_COMMAND"] = self.executable.command
        os.environ["RUNNABLE_TORCH_PARAMETERS_FILES"] = (
            self._context.parameters_file or ""
        )
        os.environ["RUNNABLE_TORCH_RUN_ID"] = self._context.run_id
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
            print(f"Error: {e}")

        delete_env_vars_with_prefix("RUNNABLE_TORCH")

        logger.info(f"attempt_log: {attempt_log}")
        logger.info(f"Step {self.name} completed with status: {attempt_log.status}")

        step_log.status = attempt_log.status
        step_log.attempts.append(attempt_log)

        return step_log

    # TODO: Not sure we need these methods
    def fan_in(self, map_variable: dict[str, str | int | float] | None = None):
        assert map_variable is None, "TorchNode does not support map_variable"

    def fan_out(self, map_variable: dict[str, str | int | float] | None = None):
        assert map_variable is None, "TorchNode does not support map_variable"
