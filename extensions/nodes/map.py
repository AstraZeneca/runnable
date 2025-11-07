import importlib
import logging
import os
import sys
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

from pydantic import Field

from extensions.nodes.task import TaskNode
from runnable import console, defaults, utils
from runnable.datastore import (
    JsonParameter,
    MetricParameter,
    ObjectParameter,
    Parameter,
)
from runnable.defaults import MapVariableType
from runnable.graph import Graph, create_graph
from runnable.nodes import CompositeNode

logger = logging.getLogger(defaults.LOGGER_NAME)


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

    # TODO: Should it be one function or a dict of functions indexed by the return name

    node_type: str = Field(default="map", serialization_alias="type")
    iterate_on: str
    iterate_as: str
    reducer: Optional[str] = Field(default=None)
    branch: Graph

    def get_summary(self) -> Dict[str, Any]:
        summary = {
            "name": self.name,
            "type": self.node_type,
            "branch": self.branch.get_summary(),
            "iterate_on": self.iterate_on,
            "iterate_as": self.iterate_as,
            "reducer": self.reducer,
        }

        return summary

    def get_reducer_function(self):
        if not self.reducer:
            return lambda *x: list(x)  # returns a list of the args

        # try a lambda function
        try:
            f = eval(self.reducer)
            if callable(f):
                return f
        except SyntaxError:
            logger.info(f"{self.reducer} is not a lambda function")

        # Load the reducer function from dotted path
        mod, func = utils.get_module_and_attr_names(self.reducer)
        sys.path.insert(0, os.getcwd())  # Need to add the current directory to path
        imported_module = importlib.import_module(mod)
        f = getattr(imported_module, func)

        return f

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

    @property
    def branch_returns(self):
        branch_returns: List[
            Tuple[str, Union[ObjectParameter, MetricParameter, JsonParameter]]
        ] = []
        for _, node in self.branch.nodes.items():
            if isinstance(node, TaskNode):
                for task_return in node.executable.returns:
                    if task_return.kind == "json":
                        branch_returns.append(
                            (
                                task_return.name,
                                JsonParameter(kind="json", value="", reduced=False),
                            )
                        )
                    elif task_return.kind == "object":
                        branch_returns.append(
                            (
                                task_return.name,
                                ObjectParameter(
                                    kind="object",
                                    value="Will be reduced",
                                    reduced=False,
                                ),
                            )
                        )
                    elif task_return.kind == "metric":
                        branch_returns.append(
                            (
                                task_return.name,
                                MetricParameter(kind="metric", value="", reduced=False),
                            )
                        )
                    else:
                        raise Exception("kind should be either json or object")

        return branch_returns

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

    def fan_out(self, map_variable: MapVariableType = None):
        """
        The general method to fan out for a node of type map.
        This method assumes that the step log has already been created.

        3rd party orchestrators should call this method to create the individual branch logs.

        Args:
            executor (BaseExecutor): The executor class as defined by the config
            map_variable (dict, optional): If the node is part of map. Defaults to None.
        """
        iterate_on = self._context.run_log_store.get_parameters(self._context.run_id)[
            self.iterate_on
        ].get_value()

        # Prepare the branch logs
        for iter_variable in iterate_on:
            effective_branch_name = self._resolve_map_placeholders(
                self.internal_name + "." + str(iter_variable), map_variable=map_variable
            )
            branch_log = self._context.run_log_store.create_branch_log(
                effective_branch_name
            )

            console.print(
                f"Branch log created for {effective_branch_name}: {branch_log}"
            )
            branch_log.status = defaults.PROCESSING
            self._context.run_log_store.add_branch_log(branch_log, self._context.run_id)

        # Gather all the returns of the task nodes and create parameters in reduced=False state.
        # TODO: Why are we preemptively creating the parameters?
        raw_parameters = {}
        if map_variable:
            # If we are in a map state already, the param should have an index of the map variable.
            for _, v in map_variable.items():
                for branch_return in self.branch_returns:
                    param_name, param_type = branch_return
                    raw_parameters[f"{v}_{param_name}"] = param_type.copy()
        else:
            for branch_return in self.branch_returns:
                param_name, param_type = branch_return
                raw_parameters[f"{param_name}"] = param_type.copy()

        self._context.run_log_store.set_parameters(
            parameters=raw_parameters, run_id=self._context.run_id
        )

    def execute_as_graph(self, map_variable: MapVariableType = None):
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
            iterate_on = self._context.run_log_store.get_parameters(
                self._context.run_id
            )[self.iterate_on].get_value()
        except KeyError as e:
            raise Exception(
                (
                    f"Expected parameter {self.iterate_on}",
                    "not present in Run Log parameters",
                    "was it ever set before?",
                )
            ) from e

        if not isinstance(iterate_on, list):
            raise Exception("Only list is allowed as a valid iterator type")

        self.fan_out(map_variable=map_variable)

        for iter_variable in iterate_on:
            effective_map_variable = map_variable or OrderedDict()
            effective_map_variable[self.iterate_as] = iter_variable

            self._context.pipeline_executor.execute_graph(
                self.branch, map_variable=effective_map_variable
            )

        self.fan_in(map_variable=map_variable)

    def fan_in(self, map_variable: MapVariableType = None):
        """
        The general method to fan in for a node of type map.

        3rd  party orchestrators should call this method to find the status of the step log.

        Args:
            executor (BaseExecutor): The executor class as defined by the config
            map_variable (dict, optional): If the node is part of map node. Defaults to None.
        """
        params = self._context.run_log_store.get_parameters(self._context.run_id)
        iterate_on = params[self.iterate_on].get_value()
        # # Find status of the branches
        step_success_bool = True
        effective_internal_name = self._resolve_map_placeholders(
            self.internal_name, map_variable=map_variable
        )

        for iter_variable in iterate_on:
            effective_branch_name = self._resolve_map_placeholders(
                self.internal_name + "." + str(iter_variable), map_variable=map_variable
            )
            branch_log = self._context.run_log_store.get_branch_log(
                effective_branch_name, self._context.run_id
            )
            # console.print(f"Branch log for {effective_branch_name}: {branch_log}")

            if branch_log.status != defaults.SUCCESS:
                step_success_bool = False

        # Collate all the results and update the status of the step
        step_log = self._context.run_log_store.get_step_log(
            effective_internal_name, self._context.run_id
        )

        if step_success_bool:  #  If none failed and nothing is waiting
            step_log.status = defaults.SUCCESS
        else:
            step_log.status = defaults.FAIL

        self._context.run_log_store.add_step_log(step_log, self._context.run_id)

        # If we failed, we return without any collection
        if not step_log.status == defaults.SUCCESS:
            return

        # Apply the reduce function and reduce the returns of the task nodes.
        # The final value of the parameter is the result of the reduce function.
        reducer_f = self.get_reducer_function()

        def update_param(
            params: Dict[str, Parameter], reducer_f: Callable, map_prefix: str = ""
        ):
            for branch_return in self.branch_returns:
                param_name, _ = branch_return

                to_reduce = []
                for iter_variable in iterate_on:
                    try:
                        to_reduce.append(
                            params[f"{iter_variable}_{param_name}"].get_value()
                        )
                    except KeyError as e:
                        from extensions.pipeline_executor.mocked import MockedExecutor

                        if isinstance(self._context.pipeline_executor, MockedExecutor):
                            pass
                        else:
                            raise Exception(
                                (
                                    f"Expected parameter {iter_variable}_{param_name}",
                                    "not present in Run Log parameters",
                                    "was it ever set before?",
                                )
                            ) from e

                param_name = f"{map_prefix}{param_name}"
                if to_reduce:
                    params[param_name].value = reducer_f(*to_reduce)
                else:
                    params[param_name].value = ""
                params[param_name].reduced = True

        if map_variable:
            # If we are in a map state already, the param should have an index of the map variable.
            for _, v in map_variable.items():
                update_param(params, reducer_f, map_prefix=f"{v}_")
        else:
            update_param(params, reducer_f)

        self._context.run_log_store.set_parameters(
            parameters=params, run_id=self._context.run_id
        )

        self._context.run_log_store.set_parameters(
            parameters=params, run_id=self._context.run_id
        )
