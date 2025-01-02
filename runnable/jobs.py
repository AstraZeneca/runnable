# TODO: move this to extensions after it matures
import logging
from abc import abstractmethod
from typing import Any, Dict

from pydantic import BaseModel, ConfigDict, Field
from stevedore import driver

import runnable.context as context
from runnable import defaults
from runnable.datastore import StepAttempt
from runnable.tasks import BaseTaskType, create_task

logger = logging.getLogger(defaults.LOGGER_NAME)
logging.getLogger("stevedore").setLevel(logging.CRITICAL)


# TODO: Do we need this abstraction?
# Why not let the job executor deal with tasks directly?


class BaseJob(BaseModel):
    name: str
    executable: BaseTaskType = Field(exclude=True)
    model_config = ConfigDict(extra="ignore")

    def __str__(self):
        return f"{self.name}"

    @property
    def _context(self):
        return context.run_context

    @classmethod
    @abstractmethod
    def parse_from_config(cls, config: Dict[str, Any]) -> "BaseJob":
        """
        Parse the config from the user and create the corresponding node.

        Args:
            config (Dict[str, Any]): The config of the node from the yaml or from the sdk.

        Returns:
            BaseNode: The corresponding node.
        """

    @abstractmethod
    def execute(
        self,
        mock: bool = False,
        attempt_number: int = 1,
        **kwargs,
    ) -> StepAttempt: ...

    @abstractmethod
    def prepare_for_job_execution(self): ...

    @abstractmethod
    def tear_down_after_job_execution(self): ...


class PythonJob(BaseJob):
    @classmethod
    def parse_from_config(cls, config: Dict[str, Any]) -> "PythonJob":
        # separate task config from node config
        task_config = {
            k: v for k, v in config.items() if k not in PythonJob.model_fields.keys()
        }
        node_config = {
            k: v for k, v in config.items() if k in PythonJob.model_fields.keys()
        }

        task_config["command_type"] = "python"
        executable = create_task(task_config)
        return cls(executable=executable, **node_config, **task_config)

    def execute(
        self,
        mock: bool = False,
        attempt_number: int = 1,
        **kwargs,
    ) -> StepAttempt:
        attempt_log = self.executable.execute_command(
            mock=mock,
            attempt_number=attempt_number,
            **kwargs,
        )
        logger.info(f"attempt_log: {attempt_log}")
        logger.info(f"Step {self.name} completed with status: {attempt_log.status}")

        return attempt_log

    def prepare_for_job_execution(self):
        pass

    def tear_down_after_job_execution(self):
        pass


class NotebookJob(BaseJob):
    @classmethod
    def parse_from_config(cls, config: Dict[str, Any]) -> "NotebookJob":
        # separate task config from node config
        task_config = {
            k: v for k, v in config.items() if k not in PythonJob.model_fields.keys()
        }
        node_config = {
            k: v for k, v in config.items() if k in PythonJob.model_fields.keys()
        }

        task_config["command_type"] = "notebook"
        executable = create_task(task_config)
        return cls(executable=executable, **node_config, **task_config)

    def execute(
        self,
        mock: bool = False,
        attempt_number: int = 1,
        **kwargs,
    ) -> StepAttempt:
        attempt_log = self.executable.execute_command(
            mock=mock,
            attempt_number=attempt_number,
            **kwargs,
        )

        logger.info(f"attempt_log: {attempt_log}")
        logger.info(f"Step {self.name} completed with status: {attempt_log.status}")
        return attempt_log

    def prepare_for_job_execution(self):
        pass

    def tear_down_after_job_execution(self):
        pass


def create_job(name: str, job_config: dict):
    """
    Creates a job object based on the configuration.
    """
    try:
        job_type = job_config.pop(
            "type"
        )  # Remove the type as it is not used in job creation.
        job_mgr: BaseJob = driver.DriverManager(namespace="jobs", name=job_type).driver

        job_config["command_type"] = job_type
        invoke_kwds = {
            "name": name,
            **job_config,
        }
        job = job_mgr.parse_from_config(config=invoke_kwds)
        return job
    except KeyError:
        msg = f"The job configuration does not contain the required key {job_type}"
        logger.exception(job_config)
        raise Exception(msg)
    except Exception as _e:
        msg = (
            f"Could not find the job type {job_type}. Please ensure you have installed "
            "the extension that provides the job type"
            "\nCore supports: python, notebook, shell"
        )
        raise Exception(msg) from _e
