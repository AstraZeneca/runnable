import logging
from copy import deepcopy
from typing import Any, cast

from pydantic import Field, field_serializer, field_validator

from runnable import console, defaults
from runnable.datastore import Parameter
from runnable.graph import Graph, create_graph
from runnable.nodes import CompositeNode, MapVariableType

logger = logging.getLogger(defaults.LOGGER_NAME)


class ConditionalNode(CompositeNode):
    """
    parameter: name -> the parameter which is used for evaluation
    default: Optional[branch] = branch to execute if nothing is matched.
    branches: {
        "case1" : branch1,
        "case2: branch2,
    }

    Conceptually this is equal to:
    match parameter:
        case "case1":
            branch1
        case "case2":
            branch2
        case _:
            default

    """

    node_type: str = Field(default="conditional", serialization_alias="type")

    parameter: str  # the name of the parameter should be isalnum
    default: Graph | None = Field(default=None)  # TODO: Think about the design of this
    branches: dict[str, Graph]
    # The keys of the branches should be isalnum()

    @field_validator("parameter", mode="after")
    @classmethod
    def check_parameter(cls, parameter: str) -> str:
        """
        Validate that the parameter name is alphanumeric.

        Args:
            parameter (str): The parameter name to validate.

        Raises:
            ValueError: If the parameter name is not alphanumeric.

        Returns:
            str: The validated parameter name.
        """
        if not parameter.isalnum():
            raise ValueError(f"Parameter '{parameter}' must be alphanumeric.")
        return parameter

    def get_parameter_value(self) -> str | int | bool | float:
        """
        Get the parameter value from the context.

        Returns:
            Any: The value of the parameter.
        """
        parameters: dict[str, Parameter] = self._context.run_log_store.get_parameters(
            run_id=self._context.run_id
        )

        if self.parameter not in parameters:
            raise Exception(f"Parameter {self.parameter} not found in parameters")

        chosen_parameter_value = parameters[self.parameter].get_value()

        assert isinstance(chosen_parameter_value, (int, float, bool, str)), (
            f"Parameter '{self.parameter}' must be of type int, float, bool, or str, "
            f"but got {type(chosen_parameter_value).__name__}."
        )

        return chosen_parameter_value

    def get_summary(self) -> dict[str, Any]:
        summary = {
            "name": self.name,
            "type": self.node_type,
            "branches": [branch.get_summary() for branch in self.branches.values()],
            "parameter": self.parameter,
            "default": self.default.get_summary() if self.default else None,
        }

        return summary

    @field_serializer("branches")
    def ser_branches(self, branches: dict[str, Graph]) -> dict[str, Graph]:
        ret: dict[str, Graph] = {}

        for branch_name, branch in branches.items():
            ret[branch_name.split(".")[-1]] = branch

        return ret

    @classmethod
    def parse_from_config(cls, config: dict[str, Any]) -> "ConditionalNode":
        internal_name = cast(str, config.get("internal_name"))

        config_branches = config.pop("branches", {})
        branches = {}
        for branch_name, branch_config in config_branches.items():
            sub_graph = create_graph(
                deepcopy(branch_config),
                internal_branch_name=internal_name + "." + branch_name,
            )
            branches[internal_name + "." + branch_name] = sub_graph

        if not branches:
            raise Exception("A parallel node should have branches")
        return cls(branches=branches, **config)

    def _get_branch_by_name(self, branch_name: str) -> Graph:
        if branch_name in self.branches:
            return self.branches[branch_name]

        raise Exception(f"Branch {branch_name} does not exist")

    def fan_out(self, map_variable: MapVariableType = None):
        """
        This method is restricted to creating branch logs.
        """
        parameter_value = self.get_parameter_value()

        hit_once = False

        for internal_branch_name, _ in self.branches.items():
            # the match is done on the last part of the branch name
            result = str(parameter_value) == internal_branch_name.split(".")[-1]

            if not result:
                # Need not create a branch log for this branch
                continue

            effective_branch_name = self._resolve_map_placeholders(
                internal_branch_name, map_variable=map_variable
            )

            hit_once = True
            branch_log = self._context.run_log_store.create_branch_log(
                effective_branch_name
            )

            console.print(
                f"Branch log created for {effective_branch_name}: {branch_log}"
            )
            branch_log.status = defaults.PROCESSING
            self._context.run_log_store.add_branch_log(branch_log, self._context.run_id)

        if not hit_once:
            raise Exception(
                "None of the branches were true. Please check your evaluate statements"
            )

    def execute_as_graph(self, map_variable: MapVariableType = None):
        """
        This function does the actual execution of the sub-branches of the parallel node.

        From a design perspective, this function should not be called if the execution is 3rd party orchestrated.

        The modes that render the job specifications, do not need to interact with this node at all as they have their
        own internal mechanisms of handing parallel states.
        If they do not, you can find a way using as-is nodes as hack nodes.

        The execution of a dag, could result in
            * The dag being completely executed with a definite (fail, success) state in case of
                local or local-container execution
            * The dag being in a processing state with PROCESSING status in case of local-aws-batch

        Only fail state is considered failure during this phase of execution.

        Args:
            executor (Executor): The Executor as per the use config
            **kwargs: Optional kwargs passed around
        """
        self.fan_out(map_variable=map_variable)
        parameter_value = self.get_parameter_value()

        for internal_branch_name, branch in self.branches.items():
            result = str(parameter_value) == internal_branch_name.split(".")[-1]

            if result:
                # if the condition is met, execute the graph
                logger.debug(f"Executing graph for {branch}")
                self._context.pipeline_executor.execute_graph(
                    branch, map_variable=map_variable
                )

        self.fan_in(map_variable=map_variable)

    def fan_in(self, map_variable: MapVariableType = None):
        """
        The general fan in method for a node of type Parallel.

        3rd party orchestrators should use this method to find the status of the composite step.

        Args:
            executor (BaseExecutor): The executor class as defined by the config
            map_variable (dict, optional): If the node is part of a map. Defaults to None.
        """
        effective_internal_name = self._resolve_map_placeholders(
            self.internal_name, map_variable=map_variable
        )

        step_success_bool: bool = True
        parameter_value = self.get_parameter_value()

        for internal_branch_name, _ in self.branches.items():
            result = str(parameter_value) == internal_branch_name.split(".")[-1]

            if not result:
                # The branch would not have been executed
                continue

            effective_branch_name = self._resolve_map_placeholders(
                internal_branch_name, map_variable=map_variable
            )

            branch_log = self._context.run_log_store.get_branch_log(
                effective_branch_name, self._context.run_id
            )

            if branch_log.status != defaults.SUCCESS:
                step_success_bool = False

        step_log = self._context.run_log_store.get_step_log(
            effective_internal_name, self._context.run_id
        )

        if step_success_bool:  #  If none failed
            step_log.status = defaults.SUCCESS
        else:
            step_log.status = defaults.FAIL

        self._context.run_log_store.add_step_log(step_log, self._context.run_id)
