import logging
from datetime import datetime
from typing import Any, Dict

from pydantic import ConfigDict, Field

from runnable import datastore, defaults
from runnable.datastore import StepLog
from runnable.defaults import MapVariableType
from runnable.nodes import ExecutableNode
from runnable.tasks import BaseTaskType, create_task

logger = logging.getLogger(defaults.LOGGER_NAME)


class TaskNode(ExecutableNode):
    """
    A node of type Task.

    This node does the actual function execution of the graph in all cases.
    """

    executable: BaseTaskType = Field(exclude=True)
    node_type: str = Field(default="task", serialization_alias="type")

    # It is technically not allowed as parse_from_config filters them.
    # This is just to get the task level configuration to be present during serialization.
    model_config = ConfigDict(extra="allow")

    @classmethod
    def parse_from_config(cls, config: Dict[str, Any]) -> "TaskNode":
        # separate task config from node config
        task_config = {
            k: v for k, v in config.items() if k not in TaskNode.model_fields.keys()
        }
        node_config = {
            k: v for k, v in config.items() if k in TaskNode.model_fields.keys()
        }

        executable = create_task(task_config)
        return cls(executable=executable, **node_config, **task_config)

    def get_summary(self) -> Dict[str, Any]:
        summary = {
            "name": self.name,
            "type": self.node_type,
            "executable": self.executable.get_summary(),
            "catalog": self._get_catalog_settings(),
        }

        return summary

    def execute(
        self,
        mock=False,
        map_variable: MapVariableType = None,
        attempt_number: int = 1,
    ) -> StepLog:
        """
        All that we do in runnable is to come to this point where we actually execute the command.

        Args:
            executor (_type_): The executor class
            mock (bool, optional): If we should just mock and not execute. Defaults to False.
            map_variable (dict, optional): If the node is part of internal branch. Defaults to None.

        Returns:
            StepAttempt: The attempt object
        """
        step_log = self._context.run_log_store.get_step_log(
            self._get_step_log_name(map_variable), self._context.run_id
        )

        if not mock:
            # Do not run if we are mocking the execution, could be useful for caching and dry runs
            attempt_log = self.executable.execute_command(map_variable=map_variable)
            attempt_log.attempt_number = attempt_number
        else:
            attempt_log = datastore.StepAttempt(
                status=defaults.SUCCESS,
                start_time=str(datetime.now()),
                end_time=str(datetime.now()),
                attempt_number=attempt_number,
            )

        logger.info(f"attempt_log: {attempt_log}")
        logger.info(f"Step {self.name} completed with status: {attempt_log.status}")

        step_log.status = attempt_log.status
        step_log.attempts.append(attempt_log)

        return step_log
