import logging
from copy import deepcopy
from string import Template
from typing import Any, cast

from pydantic import BaseModel, Field, field_serializer

from runnable import console, defaults
from runnable.datastore import JSONType
from runnable.graph import Graph, create_graph
from runnable.nodes import CompositeNode, TypeMapVariable

logger = logging.getLogger(defaults.LOGGER_NAME)


class ConditionalBranch(BaseModel):
    """
    Branch of a conditional node.

    The graph is only executed if the evaluation is successful.
    The string evaluate should be a string template:
    eg: "$parameter1 <= 5 and $parameter2 > 10"
    we replace the parameters with their values
    and evaluate the expression.

    """

    evaluate: str
    graph: Graph

    def get_summary(self) -> dict[str, Any]:
        summary = {
            "evaluate": self.evaluate,
            "graph": self.graph.get_summary(),
        }

        return summary


class ConditionalNode(CompositeNode):
    node_type: str = Field(default="conditional", serialization_alias="type")
    branches: dict[str, ConditionalBranch]

    evaluations: list[bool] = Field(default_factory=list, exclude=True)

    def get_summary(self) -> dict[str, Any]:
        summary = {
            "name": self.name,
            "type": self.node_type,
            "branches": [branch.get_summary() for branch in self.branches.values()],
        }

        return summary

    @field_serializer("branches")
    def ser_branches(
        self, branches: dict[str, ConditionalBranch]
    ) -> dict[str, ConditionalBranch]:
        ret: dict[str, ConditionalBranch] = {}

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
                deepcopy(branch_config["graph"]),
                internal_branch_name=internal_name + "." + branch_name,
            )
            conditional = ConditionalBranch(
                evaluate=branch_config["evaluate"],
                graph=sub_graph,
            )
            branches[internal_name + "." + branch_name] = conditional

        if not branches:
            raise Exception("A parallel node should have branches")
        return cls(branches=branches, **config)

    def _get_branch_by_name(self, branch_name: str) -> Graph:
        if branch_name in self.branches:
            return self.branches[branch_name].graph

        raise Exception(f"Branch {branch_name} does not exist")

    def evaluate_condition(
        self, condition: str, parameters: dict[str, JSONType]
    ) -> bool:
        """
        Evaluate the condition with the parameters.

        Args:
            condition (str): The condition to evaluate.
            parameters (JSONType): The parameters to use in the evaluation.

        Returns:
            bool: True if the condition is met, False otherwise.
        """
        str_template = Template(condition)
        evaluate = str_template.substitute(**parameters)

        assert "$" not in evaluate, "Not all required parameters were provided"

        result = eval(evaluate)  # pylint: disable=eval-used
        logger.debug(
            f"Evaluating condition: {condition} with parameters: {parameters} => {result}"
        )
        return result

    def fan_out(self, map_variable: TypeMapVariable = None):
        """
        This method is restricted to creating branch logs.
        """
        parameters: dict[str, JSONType] = {}
        for key, value in self._context.run_log_store.get_parameters(
            run_id=self._context.run_id
        ).items():
            if value.kind in ["json", "metric"]:
                parameters[key] = value.value

        parameters.update(map_variable or {})
        hit_once = False

        for internal_branch_name, branch in self.branches.items():
            effective_branch_name = self._resolve_map_placeholders(
                internal_branch_name, map_variable=map_variable
            )

            result = self.evaluate_condition(branch.evaluate, parameters)
            self.evaluations.append(result)

            if not result:
                # Need not create a branch log for this branch
                continue

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
                f"None of the branches were true. Please check your evaluate statements: {self.branches}"
            )

    def execute_as_graph(self, map_variable: TypeMapVariable = None):
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

        for index, (_, branch) in enumerate(self.branches.items()):
            if self.evaluations[index]:
                # if the condition is met, execute the graph
                logger.debug(f"Executing graph for {branch.graph}")
                self._context.executor.execute_graph(
                    branch.graph, map_variable=map_variable
                )

        self.fan_in(map_variable=map_variable)

    def fan_in(self, map_variable: TypeMapVariable = None):
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

        parameters: dict[str, JSONType] = {}
        for key, value in self._context.run_log_store.get_parameters(
            run_id=self._context.run_id
        ).items():
            if value.kind in ["json", "metric"]:
                parameters[key] = value.value

        parameters.update(map_variable or {})
        step_success_bool: bool = True

        for internal_branch_name, branch in self.branches.items():
            effective_branch_name = self._resolve_map_placeholders(
                internal_branch_name, map_variable=map_variable
            )

            result = self.evaluate_condition(branch.evaluate, parameters)

            if not result:
                # The branch would not have been executed
                continue

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
