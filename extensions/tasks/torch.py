import importlib
import logging
import os
import random
import string
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_serializer, model_validator
from ruamel.yaml import YAML

import runnable.context as context
from extensions.tasks.torch_config import EasyTorchConfig, TorchConfig
from runnable import Catalog, defaults
from runnable.datastore import StepAttempt
from runnable.tasks import BaseTaskType
from runnable.utils import get_module_and_attr_names

try:
    from torch.distributed.elastic.multiprocessing.api import DefaultLogsSpecs, Std
    from torch.distributed.launcher.api import LaunchConfig, elastic_launch

except ImportError:
    raise ImportError("torch is not installed. Please install torch first.")


logger = logging.getLogger(defaults.LOGGER_NAME)


class TorchTaskType(BaseTaskType, TorchConfig):
    task_type: str = Field(default="torch", serialization_alias="command_type")
    catalog: Optional[Catalog] = Field(default=None, alias="catalog")
    command: str

    @model_validator(mode="before")
    @classmethod
    def check_secrets_and_returns(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "secrets" in data and data["secrets"]:
                raise ValueError("'secrets' is not supported for torch")
            if "returns" in data and data["returns"]:
                raise ValueError("'secrets' is not supported for torch")
        return data

    def get_summary(self) -> dict[str, Any]:
        return self.model_dump(by_alias=True, exclude_none=True)

    @property
    def _context(self):
        return context.run_context

    def _get_launch_config(self) -> LaunchConfig:
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

    def execute_command(
        self,
        map_variable: defaults.TypeMapVariable = None,
    ):
        assert map_variable is None, "map_variable is not supported for torch"

        launch_config = self._get_launch_config()
        logger.info(f"launch_config: {launch_config}")

        # ENV variables are shared with the subprocess, use that as communication
        os.environ["RUNNABLE_TORCH_COMMAND"] = self.command
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
            attempt_log = StepAttempt(
                status=defaults.SUCCESS,
                start_time=str(datetime.now()),
                end_time=str(datetime.now()),
                attempt_number=1,
            )
        except Exception as e:
            attempt_log = StepAttempt(
                status=defaults.FAIL,
                start_time=str(datetime.now()),
                end_time=str(datetime.now()),
                attempt_number=1,
            )
            logger.error(f"Error executing TorchNode: {e}")
        finally:
            # This can only come from the subprocess
            if Path("proc_logs").exists():
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

        return attempt_log


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


def delete_env_vars_with_prefix(prefix):
    to_delete = []  # List to keep track of variables to delete

    # Iterate over a list of all environment variable keys
    for var in os.environ:
        if var.startswith(prefix):
            to_delete.append(var)

    # Delete each of the variables collected
    for var in to_delete:
        del os.environ[var]


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
    from runnable import PythonJob  # noqa: F401

    command = os.environ.get("RUNNABLE_TORCH_COMMAND")
    assert command, "Command is not provided"

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

    # In this subprocess there shoould not be any RUNNABLE environment variables
    delete_env_vars_with_prefix("RUNNABLE_")

    module_name, func_name = get_module_and_attr_names(command)
    module = importlib.import_module(module_name)

    callable_obj = getattr(module, func_name)

    # The job runs with the default configuration
    # ALl the execution logs are stored in .catalog
    job = PythonJob(function=callable_obj)

    config_content = {
        "catalog": {"type": "file-system", "config": {"catalog_location": "proc_logs"}}
    }

    temp_config_file = Path("runnable-config.yaml")
    with open(str(temp_config_file), "w", encoding="utf-8") as config_file:
        yaml = YAML(typ="safe", pure=True)
        yaml.dump(config_content, config_file)

    job.execute(
        parameters_file=parameters_files,
        job_id=process_run_id,
    )

    # delete the temp config file
    temp_config_file.unlink()

    from runnable.context import run_context

    job_log = run_context.run_log_store.get_run_log_by_id(run_id=run_context.run_id)

    if job_log.status == defaults.FAIL:
        raise Exception(f"Job {process_run_id} failed")
