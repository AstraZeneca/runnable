import logging
from datetime import datetime
from typing import Any, Dict

from pydantic import ConfigDict, Field

from runnable import datastore, defaults
from runnable.datastore import StepLog
from runnable.defaults import MapVariableType
from runnable.nodes import ExecutableNode

logger = logging.getLogger(defaults.LOGGER_NAME)


class StubNode(ExecutableNode):
    """
    Stub is a convenience design node.
    It always returns success in the attempt log and does nothing.

    This node is very similar to pass state in Step functions.

    This node type could be handy when designing the pipeline and stubbing functions
    --8<-- [start:stub_reference]
    An stub execution node of the pipeline.
    Please refer to define pipeline/tasks/stub for more information.

    As part of the dag definition, a stub task is defined as follows:

    dag:
      steps:
        stub_task: # The name of the node
        type: stub
        on_failure: The name of the step to traverse in case of failure
        next: The next node to execute after this task, use "success" to terminate the pipeline successfully
          or "fail" to terminate the pipeline with an error.

    It can take arbritary number of parameters, which is handy to temporarily silence a task node.
    --8<-- [end:stub_reference]
    """

    node_type: str = Field(default="stub", serialization_alias="type")
    model_config = ConfigDict(extra="ignore")

    def get_summary(self) -> Dict[str, Any]:
        summary = {
            "name": self.name,
            "type": self.node_type,
        }

        return summary

    @classmethod
    def parse_from_config(cls, config: Dict[str, Any]) -> "StubNode":
        return cls(**config)

    def execute(
        self,
        mock=False,
        map_variable: MapVariableType = None,
        attempt_number: int = 1,
    ) -> StepLog:
        """
        Do Nothing node.
        We just send an success attempt log back to the caller

        Args:
            executor ([type]): [description]
            mock (bool, optional): [description]. Defaults to False.
            map_variable (str, optional): [description]. Defaults to ''.

        Returns:
            [type]: [description]
        """
        step_log = self._context.run_log_store.get_step_log(
            self._get_step_log_name(map_variable), self._context.run_id
        )

        attempt_log = datastore.StepAttempt(
            status=defaults.SUCCESS,
            start_time=str(datetime.now()),
            end_time=str(datetime.now()),
            attempt_number=attempt_number,
        )

        step_log.status = attempt_log.status

        step_log.attempts.append(attempt_log)

        return step_log
