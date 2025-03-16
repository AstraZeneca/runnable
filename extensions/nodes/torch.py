import importlib
import logging
import os
import random
import string
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from pydantic import BaseModel, ConfigDict, Field, field_serializer

from extensions.nodes.torch_config import EasyTorchConfig, TorchConfig
from runnable import PythonJob, datastore, defaults
from runnable.datastore import StepLog
from runnable.nodes import DistributedNode
from runnable.tasks import PythonTaskType, create_task
from runnable.utils import TypeMapVariable

logger = logging.getLogger(defaults.LOGGER_NAME)

try:
    from torch.distributed.elastic.multiprocessing.api import DefaultLogsSpecs, Std
    from torch.distributed.launcher.api import LaunchConfig, elastic_launch

except ImportError:
    raise ImportError("torch is not installed. Please install torch first.")

print("torch is installed")


def training_subprocess():
    """
    This function is called by the torch.distributed.launcher.api.elastic_launch
    It happens in a subprocess and is responsible for executing the user's function

    It is unrelated to the actual node execution, so any cataloging, run_log_store should be
    handled to match to main process.

    We have these variables to use:

    os.environ["RUNNABLE_TORCH_COMMAND"] = self.executable.command
    os.environ["RUNNABLE_TORCH_PARAMETERS_FILES"] = (
        self._context.parameters_file or ""
    )
    os.environ["RUNNABLE_TORCH_RUN_ID"] = self._context.run_id
    os.environ["RUNNABLE_TORCH_COPY_CONTENTS_TO"] = (
        self._context.catalog_handler.compute_data_folder
    )
    os.environ["RUNNABLE_TORCH_TORCH_LOGS"] = self.log_dir or ""

    """
    command = os.environ.get("RUNNABLE_TORCH_COMMAND")
    run_id = os.environ.get("RUNNABLE_TORCH_RUN_ID", "")
    parameters_files = os.environ.get("RUNNABLE_TORCH_PARAMETERS_FILES", "")

    process_run_id = (
        run_id
        + "-"
        + os.environ.get("RANK", "")
        + "-"
        + "".join(random.choices(string.ascii_lowercase, k=3))
    )
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

    delete_env_vars_with_prefix("RUNNABLE_")

    func = get_callable_from_dotted_path(command)

    # The job runs with the default configuration
    # ALl the execution logs are stored in .catalog
    job = PythonJob(function=func)

    job.execute(
        parameters_file=parameters_files,
        job_id=process_run_id,
    )

    from runnable.context import run_context

    job_log = run_context.run_log_store.get_run_log_by_id(run_id=run_context.run_id)

    if job_log.status == defaults.FAIL:
        raise Exception(f"Job {process_run_id} failed")


# TODO: Can this be utils.get_module_and_attr_names
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


# TODO: The design of this class is not final
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
        internal_log_spec = InternalLogSpecs(**self.model_dump(exclude_none=True))
        log_spec: DefaultLogsSpecs = DefaultLogsSpecs(
            **internal_log_spec.model_dump(exclude_none=True)
        )
        easy_torch_config = EasyTorchConfig(
            **self.model_dump(
                exclude_none=True,
            )
        )

        launch_config = LaunchConfig(
            **easy_torch_config.model_dump(
                exclude_none=True,
            ),
            logs_specs=log_spec,
            run_id=self._context.run_id,
        )
        logger.info(f"launch_config: {launch_config}")
        return launch_config

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

        # ENV variables are shared with the subprocess, use that as communication
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
            logger.error(f"Error executing TorchNode: {e}")
        finally:
            # This can only come from the subprocess
            if Path(".catalog").exists():
                os.rename(".catalog", "proc_logs")
                # Move .catalog and torch_logs to the parent node's catalog location
                self._context.catalog_handler.put(
                    "proc_logs/**/*", allow_file_not_found_exc=True
                )

            # TODO: This is not working!!
            if self.log_dir:
                self._context.catalog_handler.put(
                    self.log_dir + "/**/*", allow_file_not_found_exc=True
                )

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


# This internal model makes it easier to extract the required fields
# of log specs from user specification.
# https://github.com/pytorch/pytorch/blob/main/torch/distributed/elastic/multiprocessing/api.py#L243
class InternalLogSpecs(BaseModel):
    log_dir: Optional[str] = Field(default="torch_logs")
    redirects: str = Field(default="0")  # Std.NONE
    tee: str = Field(default="0")  # Std.NONE
    local_ranks_filter: Optional[set[int]] = Field(default=None)

    model_config = ConfigDict(extra="ignore")

    @field_serializer("redirects")
    def convert_redirects(self, redirects: str) -> Std | dict[int, Std]:
        return Std.from_str(redirects)

    @field_serializer("tee")
    def convert_tee(self, tee: str) -> Std | dict[int, Std]:
        return Std.from_str(tee)
