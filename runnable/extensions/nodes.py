import logging
from collections import OrderedDict
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, cast

from pydantic import ConfigDict, Field, ValidationInfo, field_serializer, field_validator
from typing_extensions import Annotated

from runnable import datastore, defaults, utils
from runnable.datastore import StepLog
from runnable.defaults import TypeMapVariable
from runnable.graph import Graph, create_graph
from runnable.nodes import CompositeNode, ExecutableNode, TerminalNode
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
        task_config = {k: v for k, v in config.items() if k not in TaskNode.model_fields.keys()}
        node_config = {k: v for k, v in config.items() if k in TaskNode.model_fields.keys()}

        task_config["node_name"] = config.get("name")

        executable = create_task(task_config)
        return cls(executable=executable, **node_config, **task_config)

    def execute(
        self,
        mock=False,
        map_variable: TypeMapVariable = None,
        attempt_number: int = 1,
        **kwargs,
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
        print("Executing task:", self._context.executor._context_node)

        step_log = self._context.run_log_store.get_step_log(self._get_step_log_name(map_variable), self._context.run_id)
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

        step_log.status = attempt_log.status

        step_log.attempts.append(attempt_log)

        return step_log


class FailNode(TerminalNode):
    """
    A leaf node of the graph that represents a failure node
    """

    node_type: str = Field(default="fail", serialization_alias="type")

    @classmethod
    def parse_from_config(cls, config: Dict[str, Any]) -> "FailNode":
        return cast("FailNode", super().parse_from_config(config))

    def execute(
        self,
        mock=False,
        map_variable: TypeMapVariable = None,
        attempt_number: int = 1,
        **kwargs,
    ) -> StepLog:
        """
        Execute the failure node.
        Set the run or branch log status to failure.

        Args:
            executor (_type_): the executor class
            mock (bool, optional): If we should just mock and not do the actual execution. Defaults to False.
            map_variable (dict, optional): If the node belongs to internal branches. Defaults to None.

        Returns:
            StepAttempt: The step attempt object
        """
        step_log = self._context.run_log_store.get_step_log(self._get_step_log_name(map_variable), self._context.run_id)

        attempt_log = datastore.StepAttempt(
            status=defaults.FAIL,
            start_time=str(datetime.now()),
            end_time=str(datetime.now()),
            attempt_number=attempt_number,
        )

        run_or_branch_log = self._context.run_log_store.get_branch_log(
            self._get_branch_log_name(map_variable), self._context.run_id
        )
        run_or_branch_log.status = defaults.FAIL
        self._context.run_log_store.add_branch_log(run_or_branch_log, self._context.run_id)

        step_log.status = attempt_log.status

        step_log.attempts.append(attempt_log)

        return step_log


class SuccessNode(TerminalNode):
    """
    A leaf node of the graph that represents a success node
    """

    node_type: str = Field(default="success", serialization_alias="type")

    @classmethod
    def parse_from_config(cls, config: Dict[str, Any]) -> "SuccessNode":
        return cast("SuccessNode", super().parse_from_config(config))

    def execute(
        self,
        mock=False,
        map_variable: TypeMapVariable = None,
        attempt_number: int = 1,
        **kwargs,
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
        step_log = self._context.run_log_store.get_step_log(self._get_step_log_name(map_variable), self._context.run_id)

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
        self._context.run_log_store.add_branch_log(run_or_branch_log, self._context.run_id)

        step_log.status = attempt_log.status

        step_log.attempts.append(attempt_log)

        return step_log


class ParallelNode(CompositeNode):
    """
    A composite node containing many graph objects within itself.

    The structure is generally:
        ParallelNode:
            Branch A:
                Sub graph definition
            Branch B:
                Sub graph definition
            . . .

    """

    node_type: str = Field(default="parallel", serialization_alias="type")
    branches: Dict[str, Graph]
    is_composite: bool = Field(default=True, exclude=True)

    @field_serializer("branches")
    def ser_branches(self, branches: Dict[str, Graph]) -> Dict[str, Graph]:
        ret: Dict[str, Graph] = {}

        for branch_name, branch in branches.items():
            ret[branch_name.split(".")[-1]] = branch

        return ret

    @classmethod
    def parse_from_config(cls, config: Dict[str, Any]) -> "ParallelNode":
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

    def fan_out(self, map_variable: TypeMapVariable = None, **kwargs):
        """
        The general fan out method for a node of type Parallel.
        This method assumes that the step log has already been created.

        3rd party orchestrators should create the step log and use this method to create the branch logs.

        Args:
            executor (BaseExecutor): The executor class as defined by the config
            map_variable (dict, optional): If the node is part of a map node. Defaults to None.
        """
        # Prepare the branch logs
        for internal_branch_name, _ in self.branches.items():
            effective_branch_name = self._resolve_map_placeholders(internal_branch_name, map_variable=map_variable)

            branch_log = self._context.run_log_store.create_branch_log(effective_branch_name)
            branch_log.status = defaults.PROCESSING
            self._context.run_log_store.add_branch_log(branch_log, self._context.run_id)

    def execute_as_graph(self, map_variable: TypeMapVariable = None, **kwargs):
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
        self.fan_out(map_variable=map_variable, **kwargs)

        for _, branch in self.branches.items():
            self._context.executor.execute_graph(branch, map_variable=map_variable, **kwargs)

        self.fan_in(map_variable=map_variable, **kwargs)

    def fan_in(self, map_variable: TypeMapVariable = None, **kwargs):
        """
        The general fan in method for a node of type Parallel.

        3rd party orchestrators should use this method to find the status of the composite step.

        Args:
            executor (BaseExecutor): The executor class as defined by the config
            map_variable (dict, optional): If the node is part of a map. Defaults to None.
        """
        step_success_bool = True
        for internal_branch_name, _ in self.branches.items():
            effective_branch_name = self._resolve_map_placeholders(internal_branch_name, map_variable=map_variable)
            branch_log = self._context.run_log_store.get_branch_log(effective_branch_name, self._context.run_id)
            if branch_log.status != defaults.SUCCESS:
                step_success_bool = False

        # Collate all the results and update the status of the step
        effective_internal_name = self._resolve_map_placeholders(self.internal_name, map_variable=map_variable)
        step_log = self._context.run_log_store.get_step_log(effective_internal_name, self._context.run_id)

        if step_success_bool:  #  If none failed
            step_log.status = defaults.SUCCESS
        else:
            step_log.status = defaults.FAIL

        self._context.run_log_store.add_step_log(step_log, self._context.run_id)


class MapNode(CompositeNode):
    """
    A composite node that contains ONE graph object within itself that has to be executed with an iterable.

    The structure is generally:
        MapNode:
            branch

        The config is expected to have a variable 'iterate_on' and iterate_as which are looked for in the parameters.
        for iter_variable in parameters['iterate_on']:
            Execute the Branch by sending {'iterate_as': iter_variable}

    The internal naming convention creates branches dynamically based on the iteration value
    """

    node_type: str = Field(default="map", serialization_alias="type")
    iterate_on: str
    iterate_as: str
    branch: Graph
    is_composite: bool = True

    @classmethod
    def parse_from_config(cls, config: Dict[str, Any]) -> "MapNode":
        internal_name = cast(str, config.get("internal_name"))

        config_branch = config.pop("branch", {})
        if not config_branch:
            raise Exception("A map node should have a branch")

        branch = create_graph(
            deepcopy(config_branch),
            internal_branch_name=internal_name + "." + defaults.MAP_PLACEHOLDER,
        )
        return cls(branch=branch, **config)

    def _get_branch_by_name(self, branch_name: str) -> Graph:
        """
        Retrieve a branch by name.

        In the case of a Map Object, the branch naming is dynamic as it is parameterized on iterable.
        This method takes no responsibility in checking the validity of the naming.

        Returns a Graph Object

        Args:
            branch_name (str): The name of the branch to retrieve

        Raises:
            Exception: If the branch by that name does not exist
        """
        return self.branch

    def fan_out(self, map_variable: TypeMapVariable = None, **kwargs):
        """
        The general method to fan out for a node of type map.
        This method assumes that the step log has already been created.

        3rd party orchestrators should call this method to create the individual branch logs.

        Args:
            executor (BaseExecutor): The executor class as defined by the config
            map_variable (dict, optional): If the node is part of map. Defaults to None.
        """
        iterate_on = self._context.run_log_store.get_parameters(self._context.run_id)[self.iterate_on].get_value()

        # Prepare the branch logs
        for iter_variable in iterate_on:
            effective_branch_name = self._resolve_map_placeholders(
                self.internal_name + "." + str(iter_variable), map_variable=map_variable
            )
            branch_log = self._context.run_log_store.create_branch_log(effective_branch_name)
            branch_log.status = defaults.PROCESSING
            self._context.run_log_store.add_branch_log(branch_log, self._context.run_id)

    def execute_as_graph(self, map_variable: TypeMapVariable = None, **kwargs):
        """
        This function does the actual execution of the branch of the map node.

        From a design perspective, this function should not be called if the execution is 3rd party orchestrated.

        The modes that render the job specifications, do not need to interact with this node at all as
        they have their own internal mechanisms of handing map states or dynamic parallel states.
        If they do not, you can find a way using as-is nodes as hack nodes.

        The actual logic is :
            * We iterate over the iterable as mentioned in the config
            * For every value in the iterable we call the executor.execute_graph(branch, iterate_as: iter_variable)

        The execution of a dag, could result in
            * The dag being completely executed with a definite (fail, success) state in case of local
                or local-container execution
            * The dag being in a processing state with PROCESSING status in case of local-aws-batch

        Only fail state is considered failure during this phase of execution.

        Args:
            executor (Executor): The Executor as per the use config
            map_variable (dict): The map variables the graph belongs to
            **kwargs: Optional kwargs passed around
        """

        iterate_on = None
        try:
            iterate_on = self._context.run_log_store.get_parameters(self._context.run_id)[self.iterate_on].get_value()
        except KeyError:
            raise Exception(
                f"Expected parameter {self.iterate_on} not present in Run Log parameters, was it ever set before?"
            )

        if not isinstance(iterate_on, list):
            raise Exception("Only list is allowed as a valid iterator type")

        self.fan_out(map_variable=map_variable, **kwargs)

        for iter_variable in iterate_on:
            effective_map_variable = map_variable or OrderedDict()
            effective_map_variable[self.iterate_as] = iter_variable

            self._context.executor.execute_graph(self.branch, map_variable=effective_map_variable, **kwargs)

        self.fan_in(map_variable=map_variable, **kwargs)

    def fan_in(self, map_variable: TypeMapVariable = None, **kwargs):
        """
        The general method to fan in for a node of type map.

        3rd  party orchestrators should call this method to find the status of the step log.

        Args:
            executor (BaseExecutor): The executor class as defined by the config
            map_variable (dict, optional): If the node is part of map node. Defaults to None.
        """
        iterate_on = self._context.run_log_store.get_parameters(self._context.run_id)[self.iterate_on].get_value()
        # # Find status of the branches
        step_success_bool = True

        for iter_variable in iterate_on:
            effective_branch_name = self._resolve_map_placeholders(
                self.internal_name + "." + str(iter_variable), map_variable=map_variable
            )
            branch_log = self._context.run_log_store.get_branch_log(effective_branch_name, self._context.run_id)
            if branch_log.status != defaults.SUCCESS:
                step_success_bool = False

        # Collate all the results and update the status of the step
        effective_internal_name = self._resolve_map_placeholders(self.internal_name, map_variable=map_variable)
        step_log = self._context.run_log_store.get_step_log(effective_internal_name, self._context.run_id)

        if step_success_bool:  #  If none failed and nothing is waiting
            step_log.status = defaults.SUCCESS
        else:
            step_log.status = defaults.FAIL

        self._context.run_log_store.add_step_log(step_log, self._context.run_id)


class DagNode(CompositeNode):
    """
    A composite node that internally holds a dag.

    The structure is generally:
        DagNode:
            dag_definition: A YAML file that holds the dag in 'dag' block

        The config is expected to have a variable 'dag_definition'.
    """

    node_type: str = Field(default="dag", serialization_alias="type")
    dag_definition: str
    branch: Graph
    is_composite: bool = True
    internal_branch_name: Annotated[str, Field(validate_default=True)] = ""

    @field_validator("internal_branch_name")
    @classmethod
    def validate_internal_branch_name(cls, internal_branch_name: str, info: ValidationInfo):
        internal_name = info.data["internal_name"]
        return internal_name + "." + defaults.DAG_BRANCH_NAME

    @field_validator("dag_definition")
    @classmethod
    def validate_dag_definition(cls, value):
        if not value.endswith(".yaml"):  # TODO: Might have a problem with the SDK
            raise ValueError("dag_definition must be a YAML file")
        return value

    @classmethod
    def parse_from_config(cls, config: Dict[str, Any]) -> "DagNode":
        internal_name = cast(str, config.get("internal_name"))

        if "dag_definition" not in config:
            raise Exception(f"No dag definition found in {config}")

        dag_config = utils.load_yaml(config["dag_definition"])
        if "dag" not in dag_config:
            raise Exception("No DAG found in dag_definition, please provide it in dag block")

        branch = create_graph(dag_config["dag"], internal_branch_name=internal_name + "." + defaults.DAG_BRANCH_NAME)

        return cls(branch=branch, **config)

    def _get_branch_by_name(self, branch_name: str):
        """
        Retrieve a branch by name.
        The name is expected to follow a dot path convention.

        Returns a Graph Object

        Args:
            branch_name (str): The name of the branch to retrieve

        Raises:
            Exception: If the branch_name is not 'dag'
        """
        if branch_name != self.internal_branch_name:
            raise Exception(f"Node of type {self.node_type} only allows a branch of name {defaults.DAG_BRANCH_NAME}")

        return self.branch

    def fan_out(self, map_variable: TypeMapVariable = None, **kwargs):
        """
        The general method to fan out for a node of type dag.
        The method assumes that the step log has already been created.

        Args:
            executor (BaseExecutor): The executor class as defined by the config
            map_variable (dict, optional): _description_. Defaults to None.
        """
        effective_branch_name = self._resolve_map_placeholders(self.internal_branch_name, map_variable=map_variable)

        branch_log = self._context.run_log_store.create_branch_log(effective_branch_name)
        branch_log.status = defaults.PROCESSING
        self._context.run_log_store.add_branch_log(branch_log, self._context.run_id)

    def execute_as_graph(self, map_variable: TypeMapVariable = None, **kwargs):
        """
        This function does the actual execution of the branch of the dag node.

        From a design perspective, this function should not be called if the execution is 3rd party orchestrated.

        The modes that render the job specifications, do not need to interact with this node at all
        as they have their own internal mechanisms of handling sub dags.
        If they do not, you can find a way using as-is nodes as hack nodes.

        The actual logic is :
            * We just execute the branch as with any other composite nodes
            * The branch name is called 'dag'

        The execution of a dag, could result in
            * The dag being completely executed with a definite (fail, success) state in case of
                local or local-container execution
            * The dag being in a processing state with PROCESSING status in case of local-aws-batch

        Only fail state is considered failure during this phase of execution.

        Args:
            executor (Executor): The Executor as per the use config
            **kwargs: Optional kwargs passed around
        """
        self.fan_out(map_variable=map_variable, **kwargs)
        self._context.executor.execute_graph(self.branch, map_variable=map_variable, **kwargs)
        self.fan_in(map_variable=map_variable, **kwargs)

    def fan_in(self, map_variable: TypeMapVariable = None, **kwargs):
        """
        The general method to fan in for a node of type dag.

        3rd party orchestrators should call this method to find the status of the step log.

        Args:
            executor (BaseExecutor): The executor class as defined by the config
            map_variable (dict, optional): If the node is part of type dag. Defaults to None.
        """
        step_success_bool = True
        effective_branch_name = self._resolve_map_placeholders(self.internal_branch_name, map_variable=map_variable)
        effective_internal_name = self._resolve_map_placeholders(self.internal_name, map_variable=map_variable)

        branch_log = self._context.run_log_store.get_branch_log(effective_branch_name, self._context.run_id)
        if branch_log.status != defaults.SUCCESS:
            step_success_bool = False

        step_log = self._context.run_log_store.get_step_log(effective_internal_name, self._context.run_id)
        step_log.status = defaults.PROCESSING

        if step_success_bool:  #  If none failed and nothing is waiting
            step_log.status = defaults.SUCCESS
        else:
            step_log.status = defaults.FAIL

        self._context.run_log_store.add_step_log(step_log, self._context.run_id)


class StubNode(ExecutableNode):
    """
    Stub is a convenience design node.

    It always returns success in the attempt log and does nothing.

    This node is very similar to pass state in Step functions.

    This node type could be handy when designing the pipeline and stubbing functions
    """

    node_type: str = Field(default="stub", serialization_alias="type")
    model_config = ConfigDict(extra="allow")

    @classmethod
    def parse_from_config(cls, config: Dict[str, Any]) -> "StubNode":
        return cls(**config)

    def execute(
        self,
        mock=False,
        map_variable: TypeMapVariable = None,
        attempt_number: int = 1,
        **kwargs,
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
        step_log = self._context.run_log_store.get_step_log(self._get_step_log_name(map_variable), self._context.run_id)

        attempt_log = datastore.StepAttempt(
            status=defaults.SUCCESS,
            start_time=str(datetime.now()),
            end_time=str(datetime.now()),
            attempt_number=attempt_number,
        )

        step_log.status = attempt_log.status

        step_log.attempts.append(attempt_log)

        return step_log
