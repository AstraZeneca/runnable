import importlib
import logging
import os
import sys
from collections import OrderedDict
from copy import deepcopy
from multiprocessing import Pool
from typing import Any, Dict, List, Optional, Tuple, Union, cast

from pydantic import Field

from extensions.nodes.task import TaskNode
from runnable import console, defaults, exceptions, utils
from runnable.datastore import JsonParameter, MetricParameter, ObjectParameter
from runnable.defaults import IterableParameterModel, MapVariableModel
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
                                JsonParameter(kind="json", value=""),
                            )
                        )
                    elif task_return.kind == "object":
                        branch_returns.append(
                            (
                                task_return.name,
                                ObjectParameter(
                                    kind="object",
                                    value="",
                                ),
                            )
                        )
                    elif task_return.kind == "metric":
                        branch_returns.append(
                            (
                                task_return.name,
                                MetricParameter(kind="metric", value=""),
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

    def fan_out(
        self,
        iter_variable: Optional[IterableParameterModel] = None,
    ):
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

        assert iterate_on
        assert isinstance(iterate_on, list)

        # Prepare the branch logs
        for iteration_variable in iterate_on:
            effective_branch_name = self._resolve_map_placeholders(
                self.internal_name + "." + str(iteration_variable),
                iter_variable=iter_variable,
            )
            try:
                branch_log = self._context.run_log_store.get_branch_log(
                    effective_branch_name, self._context.run_id
                )
                console.print(f"Branch log already exists for {effective_branch_name}")
            except exceptions.BranchLogNotFoundError:
                branch_log = self._context.run_log_store.create_branch_log(
                    effective_branch_name
                )
                console.print(f"Branch log created for {effective_branch_name}")

            branch_log.status = defaults.PROCESSING
            self._context.run_log_store.add_branch_log(branch_log, self._context.run_id)

    def execute_as_graph(
        self,
        iter_variable: Optional[IterableParameterModel] = None,
    ):
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

        self.fan_out(iter_variable=iter_variable)

        # Check if parallel execution is enabled and supported
        enable_parallel = getattr(
            self._context.pipeline_executor, "enable_parallel", False
        )
        supports_parallel_writes = getattr(
            self._context.run_log_store, "supports_parallel_writes", False
        )

        # Check if we're using a local executor (local or local-container)
        executor_service_name = getattr(
            self._context.pipeline_executor, "service_name", ""
        )
        is_local_executor = executor_service_name in ["local", "local-container"]

        if enable_parallel and is_local_executor:
            if not supports_parallel_writes:
                logger.warning(
                    "Parallel execution was requested but the run log store does not support parallel writes. "
                    "Falling back to sequential execution. Consider using a run log store with "
                    "supports_parallel_writes=True for parallel execution."
                )
                self._execute_map_sequentially(iterate_on, iter_variable)
            else:
                logger.info("Executing map iterations in parallel")
                self._execute_map_in_parallel(iterate_on, iter_variable)
        else:
            self._execute_map_sequentially(iterate_on, iter_variable)

        self.fan_in(iter_variable=iter_variable)

    def _execute_map_sequentially(
        self,
        iterate_on: List,
        iter_variable: Optional[IterableParameterModel] = None,
    ):
        """Execute map iterations sequentially (original behavior)."""
        for iteration_variable in iterate_on:
            # Build effective map variable from existing iter_variable
            effective_map_variable = OrderedDict()
            if iter_variable and iter_variable.map_variable:
                effective_map_variable.update(
                    {k: v.value for k, v in iter_variable.map_variable.items()}
                )
            effective_map_variable[self.iterate_as] = iteration_variable

            converted_map: OrderedDict = OrderedDict(
                (k, MapVariableModel(value=v))
                for k, v in effective_map_variable.items()
            )
            effective_iter_variable = IterableParameterModel(map_variable=converted_map)

            self._context.pipeline_executor.execute_graph(
                self.branch, iter_variable=effective_iter_variable
            )

    def _execute_map_in_parallel(
        self,
        iterate_on: List,
        iter_variable: Optional[IterableParameterModel] = None,
    ):
        """Execute map iterations in parallel using multiprocessing."""
        from runnable.entrypoints import execute_single_branch

        if not isinstance(iterate_on, list):
            raise Exception("Only list is allowed as a valid iterator type")

        # Prepare arguments for each iteration
        iteration_args = []
        iteration_variables_map = []  # Keep track of iteration variables for failure reporting

        for iteration_variable in iterate_on:
            effective_iter_variable = (
                iter_variable.model_copy(deep=True)
                if iter_variable
                else IterableParameterModel()
            )

            effective_map_variable = (
                effective_iter_variable.map_variable or OrderedDict()
            )
            effective_map_variable[self.iterate_as] = MapVariableModel(
                value=iteration_variable
            )
            effective_iter_variable.map_variable = effective_map_variable

            branch_name = f"{self.internal_name}.{iteration_variable}"
            iteration_args.append(
                (
                    branch_name,
                    self.branch,
                    self._context,
                    effective_iter_variable.model_dump_json(),
                )
            )
            iteration_variables_map.append(iteration_variable)

        # Use multiprocessing Pool to execute iterations in parallel
        with Pool() as pool:
            results = pool.starmap(execute_single_branch, iteration_args)

        # Check if any iteration failed
        if not all(results):
            failed_iterations = [
                iteration_var
                for iteration_var, result in zip(iteration_variables_map, results)
                if not result
            ]
            logger.error(f"The following map iterations failed: {failed_iterations}")
            # Note: The actual failure handling and status update will be done in fan_in()

    def fan_in(
        self,
        iter_variable: Optional[IterableParameterModel] = None,
    ):
        """
        The general method to fan in for a node of type map.

        3rd  party orchestrators should call this method to find the status of the step log.

        Args:
            executor (BaseExecutor): The executor class as defined by the config
            map_variable (dict, optional): If the node is part of map node. Defaults to None.
        """
        params = self._context.run_log_store.get_parameters(self._context.run_id)
        iterate_on = params[self.iterate_on].get_value()

        assert iterate_on
        assert isinstance(iterate_on, list)
        # # Find status of the branches
        step_success_bool = True
        effective_internal_name = self._resolve_map_placeholders(
            self.internal_name, iter_variable=iter_variable
        )

        for iteration_variable in iterate_on:
            effective_branch_name = self._resolve_map_placeholders(
                self.internal_name + "." + str(iteration_variable),
                iter_variable=iter_variable,
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

        # Get parent scope for setting collected parameters
        # The parent is where the map node itself lives (no placeholder resolution needed)
        parent_params = self._context.run_log_store.get_parameters(
            self._context.run_id, internal_branch_name=self.internal_branch_name
        )

        for branch_return in self.branch_returns:
            param_name, _ = branch_return

            to_reduce = []
            for iteration_variable in iterate_on:
                effective_branch_name = self._resolve_map_placeholders(
                    self.internal_name + "." + str(iteration_variable),
                    iter_variable=iter_variable,
                )
                branch_params = self._context.run_log_store.get_parameters(
                    self._context.run_id, internal_branch_name=effective_branch_name
                )

                # Only add to reduce list if parameter exists
                # (branch might have taken failure path and not set all returns)
                if param_name in branch_params:
                    to_reduce.append(branch_params[param_name].get_value())

            # Create or update the parameter in parent scope with collected value
            if to_reduce:
                value = reducer_f(*to_reduce)
            else:
                value = ""

            # Create parameter if it doesn't exist in parent
            if param_name not in parent_params:
                parent_params[param_name] = JsonParameter(kind="json", value=value)
            else:
                parent_params[param_name].value = value

        self._context.run_log_store.set_parameters(
            parameters=parent_params,
            run_id=self._context.run_id,
            internal_branch_name=self.internal_branch_name,
        )

    async def execute_as_graph_async(
        self,
        iter_variable: Optional[IterableParameterModel] = None,
    ):
        """Async map execution."""
        self.fan_out(iter_variable=iter_variable)  # sync

        iterate_on = self._context.run_log_store.get_parameters(self._context.run_id)[
            self.iterate_on
        ].get_value()
        assert iterate_on
        assert isinstance(iterate_on, list)

        for iteration_variable in iterate_on:
            # Build effective map variable from existing iter_variable
            effective_map_variable = OrderedDict()
            if iter_variable and iter_variable.map_variable:
                effective_map_variable.update(
                    {k: v.value for k, v in iter_variable.map_variable.items()}
                )
            effective_map_variable[self.iterate_as] = iteration_variable

            # Convert to IterableParameterModel
            converted_map: OrderedDict = OrderedDict(
                (k, MapVariableModel(value=v))
                for k, v in effective_map_variable.items()
            )
            effective_iter_variable = IterableParameterModel(map_variable=converted_map)

            await self._context.pipeline_executor.execute_graph_async(
                self.branch, iter_variable=effective_iter_variable
            )

        self.fan_in(iter_variable=iter_variable)  # sync
