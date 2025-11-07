from datetime import datetime
from typing import Any, Dict, cast

from pydantic import Field

from runnable import datastore, defaults
from runnable.datastore import StepLog
from runnable.defaults import MapVariableType
from runnable.nodes import TerminalNode


class SuccessNode(TerminalNode):
    """
    A leaf node of the graph that represents a success node
    """

    node_type: str = Field(default="success", serialization_alias="type")

    @classmethod
    def parse_from_config(cls, config: Dict[str, Any]) -> "SuccessNode":
        return cast("SuccessNode", super().parse_from_config(config))

    def get_summary(self) -> Dict[str, Any]:
        summary = {
            "name": self.name,
            "type": self.node_type,
        }

        return summary

    def execute(
        self,
        mock=False,
        map_variable: MapVariableType = None,
        attempt_number: int = 1,
    ) -> StepLog:
        """
        Execute the success node.
        Set the run or branch log status to success.

        Args:
            executor (_type_): The executor class
            mock (bool, optional): If we should just mock and not perform anything. Defaults to False.
            map_variable (dict, optional): If the node belongs to an internal branch. Defaults to None.

        Returns:
            StepAttempt: The step attempt object
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

        run_or_branch_log = self._context.run_log_store.get_branch_log(
            self._get_branch_log_name(map_variable), self._context.run_id
        )
        run_or_branch_log.status = defaults.SUCCESS
        self._context.run_log_store.add_branch_log(
            run_or_branch_log, self._context.run_id
        )

        step_log.status = attempt_log.status

        step_log.attempts.append(attempt_log)

        return step_log
