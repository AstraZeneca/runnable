import logging
from copy import deepcopy
from typing import Any, Dict, Optional, cast

from pydantic import Field, field_validator

from runnable import defaults
from runnable.datastore import Parameter
from runnable.defaults import IterableParameterModel, LoopIndexModel, LOOP_PLACEHOLDER
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

    def get_break_condition_value(
        self, iter_variable: Optional[IterableParameterModel] = None
    ) -> bool:
        """Get the break condition parameter value from current iteration branch."""
        # Get parameters from current iteration branch scope
        current_branch_name = self._get_iteration_branch_name(iter_variable)

        parameters: dict[str, Parameter] = self._context.run_log_store.get_parameters(
            run_id=self._context.run_id, internal_branch_name=current_branch_name
        )

        if self.break_on not in parameters:
            return False  # Default to continue if parameter doesn't exist

        condition_value = parameters[self.break_on].get_value()

        if not isinstance(condition_value, bool):
            raise ValueError(
                f"Break condition '{self.break_on}' must be boolean, "
                f"got {type(condition_value).__name__}"
            )

        return condition_value

    def _create_iteration_branch_log(
        self, iter_variable: Optional[IterableParameterModel] = None
    ):
        """Create branch log for the current iteration."""
        branch_name = self._get_iteration_branch_name(iter_variable)

        try:
            branch_log = self._context.run_log_store.get_branch_log(
                branch_name, self._context.run_id
            )
            logger.debug(f"Branch log already exists for {branch_name}")
        except Exception:  # BranchLogNotFoundError
            branch_log = self._context.run_log_store.create_branch_log(branch_name)
            logger.debug(f"Branch log created for {branch_name}")

        branch_log.status = defaults.PROCESSING
        self._context.run_log_store.add_branch_log(branch_log, self._context.run_id)
        return branch_log

    def _build_iteration_iter_variable(
        self, parent_iter_variable: Optional[IterableParameterModel], iteration: int
    ) -> IterableParameterModel:
        """Build iter_variable for current iteration."""
        if parent_iter_variable:
            iter_var = parent_iter_variable.model_copy(deep=True)
        else:
            iter_var = IterableParameterModel()

        # Initialize loop_variable if None
        if iter_var.loop_variable is None:
            iter_var.loop_variable = []

        # Add current iteration index
        iter_var.loop_variable.append(LoopIndexModel(value=iteration))

        return iter_var

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
