import logging
from copy import deepcopy
from typing import Any, Dict, Optional, cast

from pydantic import Field, field_validator

from runnable import defaults
from runnable.defaults import IterableParameterModel, LOOP_PLACEHOLDER
from runnable.graph import Graph, create_graph
from runnable.nodes import CompositeNode

logger = logging.getLogger(defaults.LOGGER_NAME)


class LoopNode(CompositeNode):
    """
    A loop node that iterates over a branch until a break condition is met.

    The branch executes repeatedly until either:
    - parameters[break_on] == True
    - max_iterations is reached

    Each iteration gets its own branch log using LOOP_PLACEHOLDER pattern.
    """

    node_type: str = Field(default="loop", serialization_alias="type")

    # The sub-graph to execute repeatedly
    branch: Graph

    # Maximum iterations (safety limit)
    max_iterations: int

    # Boolean parameter name - when True, loop exits
    break_on: str

    # Environment variable name for iteration index (no prefix)
    index_as: str

    @field_validator("break_on", mode="after")
    @classmethod
    def check_break_on(cls, break_on: str) -> str:
        """Validate that the break_on parameter name is alphanumeric."""
        if not break_on.isalnum():
            raise ValueError(f"Parameter '{break_on}' must be alphanumeric.")
        return break_on

    @field_validator("index_as", mode="after")
    @classmethod
    def check_index_as(cls, index_as: str) -> str:
        """Validate that the index_as variable name is alphanumeric."""
        if not index_as.isalnum():
            raise ValueError(f"Variable '{index_as}' must be alphanumeric.")
        return index_as

    def get_summary(self) -> dict[str, Any]:
        summary = {
            "name": self.name,
            "type": self.node_type,
            "branch": self.branch.get_summary(),
            "max_iterations": self.max_iterations,
            "break_on": self.break_on,
            "index_as": self.index_as,
        }
        return summary

    def _get_iteration_branch_name(
        self, iter_variable: Optional[IterableParameterModel] = None
    ) -> str:
        """Get branch name for current iteration using placeholder resolution."""
        # Create branch name template with loop placeholder
        branch_template = f"{self.internal_name}.{LOOP_PLACEHOLDER}"

        # Resolve using the refactored method
        return self._resolve_iter_placeholders(branch_template, iter_variable)

    def fan_out(self, iter_variable: Optional[IterableParameterModel] = None):
        """Create branch log and set up parameters - implementation in next task."""
        pass

    def execute_as_graph(self, iter_variable: Optional[IterableParameterModel] = None):
        """Execute the loop locally - implementation in next task."""
        pass

    def fan_in(self, iter_variable: Optional[IterableParameterModel] = None) -> bool:
        """Check conditions and return should_exit flag - implementation in next task."""
        return True

    def _get_branch_by_name(self, branch_name: str) -> Graph:
        """
        Retrieve a branch by name.

        For a loop node, we always return the single branch.
        This method takes no responsibility in checking the validity of the naming.

        Args:
            branch_name (str): The name of the branch to retrieve

        Returns:
            Graph: The loop branch
        """
        return self.branch

    @classmethod
    def parse_from_config(cls, config: Dict[str, Any]) -> "LoopNode":
        """
        Parse LoopNode from configuration dictionary.

        Args:
            config: Configuration dictionary containing node settings

        Returns:
            LoopNode: Configured loop node instance
        """
        internal_name = cast(str, config.get("internal_name"))

        config_branch = config.pop("branch", {})
        if not config_branch:
            raise Exception("A loop node should have a branch")

        branch = create_graph(
            deepcopy(config_branch),
            internal_branch_name=internal_name + "." + LOOP_PLACEHOLDER,
        )
        return cls(branch=branch, **config)
