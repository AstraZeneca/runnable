import json
import logging
import multiprocessing
from collections import OrderedDict
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, Optional, cast

from pydantic import ConfigDict, Field, FieldValidationInfo, field_validator
from typing_extensions import Annotated

import magnus
from magnus import defaults, utils
from magnus.datastore import StepAttempt
from magnus.graph import Graph, create_graph
from magnus.nodes import CompositeNode, ExecutableNode, TerminalNode
from magnus.tasks import BaseTaskType, create_task

logger = logging.getLogger(defaults.LOGGER_NAME)


class TaskNode(ExecutableNode):
    """
    A node of type Task.

    This node does the actual function execution of the graph in all cases.
    """

    executable: BaseTaskType
    node_type: str = Field(default="task", serialization_alias="type")

    @classmethod
    def parse_from_config(cls, config: Dict[str, Any], parent_step: Optional[str] = None) -> "TaskNode":
        # separate task config from node config
        task_config = {k: v for k, v in config.items() if k not in TaskNode.model_fields.keys()}
        node_config = {k: v for k, v in config.items() if k in TaskNode.model_fields.keys()}

        task_config["node_name"] = config.get("name")

        executable = create_task(task_config)
        return cls(executable=executable, **node_config)

    def execute(self, mock=False, map_variable: Optional[Dict[str, str]] = None, **kwargs) -> StepAttempt:
        """
        All that we do in magnus is to come to this point where we actually execute the command.

        Args:
            executor (_type_): The executor class
            mock (bool, optional): If we should just mock and not execute. Defaults to False.
            map_variable (dict, optional): If the node is part of internal branch. Defaults to None.

        Returns:
            StepAttempt: The attempt object
        """
        # Here is where the juice is
        attempt_log = self._context.run_log_store.create_attempt_log()
        try:
            attempt_log.start_time = str(datetime.now())
            attempt_log.status = defaults.SUCCESS
            if not mock:
                # Do not run if we are mocking the execution, could be useful for caching and dry runs
                self.executable.execute_command(map_variable=map_variable)
        except Exception as _e:  # pylint: disable=W0703
            logger.exception("Task failed")
            attempt_log.status = defaults.FAIL
            attempt_log.message = str(_e)
        finally:
            attempt_log.end_time = str(datetime.now())
            attempt_log.duration = utils.get_duration_between_datetime_strings(
                attempt_log.start_time, attempt_log.end_time
            )
        return attempt_log


class FailNode(TerminalNode):
    """
    A leaf node of the graph that represents a failure node
    """

    node_type: str = Field(default="fail", serialization_alias="type")

    @classmethod
    def parse_from_config(cls, config: Dict[str, Any], parent_step: Optional[str] = None) -> "FailNode":
        return cast("FailNode", super().parse_from_config(config, parent_step))

    def execute(self, mock=False, map_variable: Optional[Dict[str, str]] = None, **kwargs) -> StepAttempt:
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
        attempt_log = self._context.run_log_store.create_attempt_log()
        try:
            attempt_log.start_time = str(datetime.now())
            attempt_log.status = defaults.SUCCESS
            #  could be a branch or run log
            run_or_branch_log = self._context.run_log_store.get_branch_log(
                self._get_branch_log_name(map_variable), self._context.run_id
            )
            run_or_branch_log.status = defaults.FAIL
            self._context.run_log_store.add_branch_log(run_or_branch_log, self._context.run_id)
        except BaseException:  # pylint: disable=W0703
            logger.exception("Fail node execution failed")
        finally:
            attempt_log.status = defaults.SUCCESS  # This is a dummy node, so we ignore errors and mark SUCCESS
            attempt_log.end_time = str(datetime.now())
            attempt_log.duration = utils.get_duration_between_datetime_strings(
                attempt_log.start_time, attempt_log.end_time
            )
        return attempt_log


class SuccessNode(TerminalNode):
    """
    A leaf node of the graph that represents a success node
    """

    node_type: str = Field(default="success", serialization_alias="type")

    @classmethod
    def parse_from_config(cls, config: Dict[str, Any], parent_step: Optional[str] = None) -> "SuccessNode":
        return cast("SuccessNode", super().parse_from_config(config, parent_step))

    def execute(self, mock=False, map_variable: Optional[Dict[str, str]] = None, **kwargs) -> StepAttempt:
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
        attempt_log = self._context.run_log_store.create_attempt_log()
        try:
            attempt_log.start_time = str(datetime.now())
            attempt_log.status = defaults.SUCCESS
            #  could be a branch or run log
            run_or_branch_log = self._context.run_log_store.get_branch_log(
                self._get_branch_log_name(map_variable), self._context.run_id
            )
            run_or_branch_log.status = defaults.SUCCESS
            self._context.run_log_store.add_branch_log(run_or_branch_log, self._context.run_id)
        except BaseException:  # pylint: disable=W0703
            logger.exception("Success node execution failed")
        finally:
            attempt_log.status = defaults.SUCCESS  # This is a dummy node and we make sure we mark it as success
            attempt_log.end_time = str(datetime.now())
            attempt_log.duration = utils.get_duration_between_datetime_strings(
                attempt_log.start_time, attempt_log.end_time
            )
        return attempt_log


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
    is_composite: bool = True

    @classmethod
    def parse_from_config(cls, config: Dict[str, Any], parent_step: Optional[str]) -> "ParallelNode":
        if not parent_step:
            raise Exception("A parallel node should have a parent step during parsing")

        config_branches = config.pop("branches", {})
        branches = {}
        for branch_name, branch_config in config_branches.items():
            sub_graph = create_graph(
                deepcopy(branch_config),
                internal_branch_name=parent_step + "." + branch_name,
            )
            branches[parent_step + "." + branch_name] = sub_graph

        if not branches:
            raise Exception("A parallel node should have branches")
        return cls(branches=branches, **config)

    def _get_branch_by_name(self, branch_name: str) -> Graph:
        if branch_name in self.branches:
            return self.branches[branch_name]

        raise Exception(f"Branch {branch_name} does not exist")

    def add_parent(self, parent: str):
        self.internal_name = parent + "." + self.internal_name

        for branch in self.branches.values():
            for node in branch.nodes.values():
                node.add_parent(parent)

    def fan_out(self, map_variable: Optional[Dict[str, str]] = None, **kwargs):
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

    def execute_as_graph(self, map_variable: Optional[Dict[str, str]] = None, **kwargs):
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

        jobs = []
        # Given that we can have nesting and complex graphs, controlling the number of processes is hard.
        # A better way is to actually submit the job to some process scheduler which does resource management
        for internal_branch_name, branch in self.branches.items():
            if self._context.executor._is_parallel_execution():
                # Trigger parallel jobs
                action = magnus.entrypoints.execute_single_brach
                kwargs = {
                    "configuration_file": self._context.configuration_file,
                    "pipeline_file": self._context.pipeline_file,
                    "branch_name": internal_branch_name.replace(" ", defaults.COMMAND_FRIENDLY_CHARACTER),
                    "run_id": self._context.run_id,
                    "map_variable": json.dumps(map_variable),
                    "tag": self._context.tag,
                }
                process = multiprocessing.Process(target=action, kwargs=kwargs)
                jobs.append(process)
                process.start()

            else:
                # If parallel is not enabled, execute them sequentially
                self._context.executor.execute_graph(branch, map_variable=map_variable, **kwargs)

        for job in jobs:
            job.join()  # Find status of the branches

        self.fan_in(map_variable=map_variable, **kwargs)

    def fan_in(self, map_variable: Optional[Dict[str, str]] = None, **kwargs):
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
    def parse_from_config(cls, config: Dict[str, Any], parent_step: Optional[str]) -> "MapNode":
        if not parent_step:
            raise Exception("A map node should have an parent step during parsing")

        config_branch = config.pop("branch", {})
        if not config_branch:
            raise Exception("A map node should have a branch")

        branch = create_graph(
            deepcopy(config_branch),
            internal_branch_name=parent_step + "." + defaults.MAP_PLACEHOLDER,
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

    def add_parent(self, parent: str):
        self.internal_name = parent + "." + self.internal_name

        for node in self.branch.nodes.values():
            node.add_parent(parent)

    def fan_out(self, map_variable: Optional[Dict[str, str]] = None, **kwargs):
        """
        The general method to fan out for a node of type map.
        This method assumes that the step log has already been created.

        3rd party orchestrators should call this method to create the individual branch logs.

        Args:
            executor (BaseExecutor): The executor class as defined by the config
            map_variable (dict, optional): If the node is part of map. Defaults to None.
        """
        iterate_on = self._context.run_log_store.get_parameters(self._context.run_id)[self.iterate_on]

        # Prepare the branch logs
        for iter_variable in iterate_on:
            effective_branch_name = self._resolve_map_placeholders(
                self.internal_name + "." + str(iter_variable), map_variable=map_variable
            )
            branch_log = self._context.run_log_store.create_branch_log(effective_branch_name)
            branch_log.status = defaults.PROCESSING
            self._context.run_log_store.add_branch_log(branch_log, self._context.run_id)

    def execute_as_graph(self, map_variable: Optional[Dict[str, str]] = None, **kwargs):
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
            iterate_on = self._context.run_log_store.get_parameters(self._context.run_id)[self.iterate_on]
        except KeyError:
            raise Exception(
                f"Expected parameter {self.iterate_on} not present in Run Log parameters, was it ever set before?"
            )

        if not isinstance(iterate_on, list):
            raise Exception("Only list is allowed as a valid iterator type")

        self.fan_out(map_variable=map_variable, **kwargs)

        jobs = []
        # Given that we can have nesting and complex graphs, controlling the number of processess is hard.
        # A better way is to actually submit the job to some process scheduler which does resource management
        for iter_variable in iterate_on:
            effective_map_variable = map_variable or OrderedDict()
            effective_map_variable[self.iterate_as] = iter_variable

            if self._context.executor._is_parallel_execution():
                # Trigger parallel jobs
                action = magnus.entrypoints.execute_single_brach
                kwargs = {
                    "configuration_file": self._context.configuration_file,
                    "pipeline_file": self._context.pipeline_file,
                    "branch_name": self.branch.internal_branch_name.replace(" ", defaults.COMMAND_FRIENDLY_CHARACTER),
                    "run_id": self._context.run_id,
                    "map_variable": json.dumps(effective_map_variable),
                    "tag": self._context.tag,
                }
                process = multiprocessing.Process(target=action, kwargs=kwargs)
                jobs.append(process)
                process.start()

            else:
                # If parallel is not enabled, execute them sequentially
                self._context.executor.execute_graph(self.branch, map_variable=effective_map_variable, **kwargs)

        for job in jobs:
            job.join()

        self.fan_in(map_variable=map_variable, **kwargs)

    def fan_in(self, map_variable: Optional[Dict[str, str]] = None, **kwargs):
        """
        The general method to fan in for a node of type map.

        3rd  party orchestrators should call this method to find the status of the step log.

        Args:
            executor (BaseExecutor): The executor class as defined by the config
            map_variable (dict, optional): If the node is part of map node. Defaults to None.
        """
        iterate_on = self._context.run_log_store.get_parameters(self._context.run_id)[self.iterate_on]
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
    def validate_internal_branch_name(cls, internal_branch_name: str, info: FieldValidationInfo):
        internal_name = info.data["internal_name"]
        return internal_name + "." + defaults.DAG_BRANCH_NAME

    @field_validator("dag_definition")
    @classmethod
    def validate_dag_definition(cls, value):
        if not value.endswith(".yaml"):  # TODO: Might have a problem with the SDK
            raise ValueError("dag_definition must be a YAML file")
        return value

    @classmethod
    def parse_from_config(cls, config: Dict[str, Any], parent_step: Optional[str] = None) -> "DagNode":
        if not parent_step:
            raise Exception("A dag node should have a parent step during parsing")

        if "dag_definition" not in config:
            raise Exception(f"No dag definition found in {config}")

        dag_config = utils.load_yaml(config["dag_definition"])
        if "dag" not in dag_config:
            raise Exception("No DAG found in dag_definition, please provide it in dag block")

        branch = create_graph(dag_config["dag"], internal_branch_name=parent_step + "." + defaults.DAG_BRANCH_NAME)

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

    def add_parent(self, parent: str):
        self.internal_name = parent + "." + self.internal_name

        for node in self.branch.nodes.values():
            node.add_parent(parent)

    def fan_out(self, map_variable: Optional[Dict[str, str]] = None, **kwargs):
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

    def execute_as_graph(self, map_variable: Optional[Dict[str, str]] = None, **kwargs):
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

    def fan_in(self, map_variable: Optional[Dict[str, str]] = None, **kwargs):
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

    It always returns success in the attempt log and does nothing during interactive compute.

    The command given to execute is ignored but it does do the syncing of the catalog.
    This node is very similar to pass state in Step functions.

    This node type could be handy when designing the pipeline and stubbing functions

    But in render mode for job specification of a 3rd party orchestrator, this node comes handy.
    """

    node_type: str = Field(default="stub", serialization_alias="type")
    model_config = ConfigDict(extra="allow")

    @classmethod
    def parse_from_config(cls, config: Dict[str, Any], parent_step: Optional[str] = None) -> "StubNode":
        return cls(**config)

    def execute(self, mock=False, map_variable: Optional[Dict[str, str]] = None, **kwargs) -> StepAttempt:
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
        attempt_log = self._context.run_log_store.create_attempt_log()

        attempt_log.start_time = str(datetime.now())
        attempt_log.status = defaults.SUCCESS  # This is a dummy node and always will be success

        attempt_log.end_time = str(datetime.now())
        attempt_log.duration = utils.get_duration_between_datetime_strings(attempt_log.start_time, attempt_log.end_time)
        return attempt_log
