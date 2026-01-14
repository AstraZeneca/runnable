import logging
import os
from copy import deepcopy
from typing import Any, Dict, Optional, cast

from pydantic import Field, PrivateAttr

from runnable import defaults
from runnable.datastore import Parameter
from runnable.defaults import LOOP_PLACEHOLDER, IterableParameterModel, LoopIndexModel
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
    _should_exit: bool = PrivateAttr(default=False)

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
        """
        Create branch log for current iteration and copy parameters.

        For iteration 0: copy from parent scope
        For iteration N: copy from previous iteration (N-1) scope
        """
        # Create branch log for current iteration
        self._create_iteration_branch_log(iter_variable)

        # Determine current iteration from iter_variable
        current_iteration = 0
        if iter_variable and iter_variable.loop_variable:
            current_iteration = iter_variable.loop_variable[-1].value

        # Determine source of parameters
        if current_iteration == 0:
            # Copy from parent scope
            source_branch_name = self.internal_branch_name
        else:
            # Copy from previous iteration
            prev_iter_var = (
                iter_variable.model_copy(deep=True)
                if iter_variable
                else IterableParameterModel()
            )
            if prev_iter_var.loop_variable is None:
                prev_iter_var.loop_variable = []
            # Replace last loop index with previous iteration
            prev_iter_var.loop_variable[-1] = LoopIndexModel(
                value=current_iteration - 1
            )
            source_branch_name = self._get_iteration_branch_name(prev_iter_var)

        # Get source parameters
        source_params = self._context.run_log_store.get_parameters(
            run_id=self._context.run_id, internal_branch_name=source_branch_name
        )

        # Copy to current iteration branch
        target_branch_name = self._get_iteration_branch_name(iter_variable)
        self._context.run_log_store.set_parameters(
            parameters=source_params,
            run_id=self._context.run_id,
            internal_branch_name=target_branch_name,
        )

    def execute_as_graph(self, iter_variable: Optional[IterableParameterModel] = None):
        """
        Execute the loop locally.

        This function implements the main loop execution logic:
        1. Call fan_out() to set up iteration 0
        2. Loop until break condition or max_iterations
        3. For each iteration:
           - Set iteration index environment variable
           - Build iter_variable for current iteration
           - Execute branch graph
           - Check termination conditions with fan_in()
           - Create next iteration if continuing

        Args:
            iter_variable: Optional iteration context from parent composite nodes
        """
        # Initialize with iteration 0
        iteration = 0
        iteration_iter_variable = self._build_iteration_iter_variable(
            iter_variable, iteration
        )

        # Set up iteration 0
        self.fan_out(iter_variable=iteration_iter_variable)

        while True:
            # Set iteration index environment variable
            os.environ[self.index_as] = str(iteration)

            logger.debug(f"Executing loop iteration {iteration} for {self.name}")

            # Execute the branch for this iteration
            self._context.pipeline_executor.execute_graph(
                self.branch, iter_variable=iteration_iter_variable
            )

            # Check termination conditions
            self.fan_in(iter_variable=iteration_iter_variable)

            if self._should_exit:
                logger.debug(f"Loop {self.name} exiting after iteration {iteration}")
                break

            # Prepare for next iteration
            iteration += 1

            # Safety check - this should be caught by fan_in, but double-check
            if iteration >= self.max_iterations:
                logger.warning(
                    f"Loop {self.name} hit max_iterations safety limit: {self.max_iterations}"
                )
                break

            # Build iter_variable for next iteration and set it up
            iteration_iter_variable = self._build_iteration_iter_variable(
                iter_variable, iteration
            )
            self.fan_out(iter_variable=iteration_iter_variable)

    def fan_in(self, iter_variable: Optional[IterableParameterModel] = None) -> None:
        """
        Check termination conditions and handle loop completion.

        Checks in order:
        1. Branch execution failure - if current iteration failed, exit with fail status
        2. Break condition - if break_on parameter is True, exit with success status
        3. Max iterations - if reached limit, exit with current branch status

        Returns:
            None: Sets self._should_exit and handles status/parameter rollback
        """
        # Get current iteration from iter_variable
        current_iteration = 0
        if iter_variable and iter_variable.loop_variable:
            current_iteration = iter_variable.loop_variable[-1].value

        # FIRST: Check if current iteration's branch execution failed
        current_branch_name = self._get_iteration_branch_name(iter_variable)
        try:
            branch_log = self._context.run_log_store.get_branch_log(
                current_branch_name, self._context.run_id
            )

            # If branch execution failed, exit immediately with fail status
            if branch_log.status != defaults.SUCCESS:
                logger.debug(
                    f"Loop {self.name} exiting due to branch failure in iteration {current_iteration}"
                )
                self._rollback_parameters_to_parent(iter_variable)
                self._set_step_status_to_fail(iter_variable)
                self._should_exit = True
                return

        except Exception:
            # If we can't get branch log, assume failure
            logger.warning(
                f"Loop {self.name} could not get branch log for {current_branch_name}, assuming failure"
            )
            self._rollback_parameters_to_parent(iter_variable)
            self._set_step_status_to_fail(iter_variable)
            self._should_exit = True
            return

        # SECOND: Check break condition (only if branch succeeded)
        break_condition_met = False
        try:
            break_condition_met = self.get_break_condition_value(iter_variable)
        except (KeyError, ValueError):
            # If break parameter doesn't exist or invalid, continue
            break_condition_met = False

        # THIRD: Check max iterations (0-indexed, so iteration N means N+1 total iterations)
        max_iterations_reached = current_iteration >= (self.max_iterations - 1)

        should_exit = break_condition_met or max_iterations_reached

        if should_exit:
            # Roll back parameters to parent and set status based on branch success
            self._rollback_parameters_to_parent(iter_variable)
            self._set_final_step_status(iter_variable)

        self._should_exit = should_exit

    def _rollback_parameters_to_parent(
        self, iter_variable: Optional[IterableParameterModel] = None
    ):
        """Copy parameters from current iteration back to parent scope."""
        current_branch_name = self._get_iteration_branch_name(iter_variable)

        current_params = self._context.run_log_store.get_parameters(
            run_id=self._context.run_id, internal_branch_name=current_branch_name
        )

        # Copy back to parent
        self._context.run_log_store.set_parameters(
            parameters=current_params,
            run_id=self._context.run_id,
            internal_branch_name=self.internal_branch_name,
        )

    def _set_final_step_status(
        self, iter_variable: Optional[IterableParameterModel] = None
    ):
        """Set the loop node's final status based on branch execution."""
        effective_internal_name = self._resolve_iter_placeholders(
            self.internal_name, iter_variable=iter_variable
        )

        step_log = self._context.run_log_store.get_step_log(
            effective_internal_name, self._context.run_id
        )

        # Check current iteration branch status
        current_branch_name = self._get_iteration_branch_name(iter_variable)
        try:
            current_branch_log = self._context.run_log_store.get_branch_log(
                current_branch_name, self._context.run_id
            )

            if current_branch_log.status == defaults.SUCCESS:
                step_log.status = defaults.SUCCESS
            else:
                step_log.status = defaults.FAIL

        except Exception:
            # If branch log not found, mark as failed
            step_log.status = defaults.FAIL

        self._context.run_log_store.add_step_log(step_log, self._context.run_id)

    def _set_step_status_to_fail(
        self, iter_variable: Optional[IterableParameterModel] = None
    ):
        """Set the loop node's status to FAIL when branch execution fails."""
        effective_internal_name = self._resolve_iter_placeholders(
            self.internal_name, iter_variable=iter_variable
        )

        step_log = self._context.run_log_store.get_step_log(
            effective_internal_name, self._context.run_id
        )

        step_log.status = defaults.FAIL
        self._context.run_log_store.add_step_log(step_log, self._context.run_id)

    def _get_branch_by_name(self, branch_name: str) -> Graph:  # noqa: ARG002
        """
        Retrieve a branch by name.

        For a loop node, we always return the single branch.
        This method takes no responsibility in checking the validity of the naming.

        Args:
            branch_name (str): The name of the branch to retrieve (unused, interface compatibility)

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
