import logging
from copy import deepcopy
from multiprocessing import Pool
from typing import Any, Dict, Optional, cast

from pydantic import Field, field_serializer

from runnable import console, defaults, exceptions
from runnable.defaults import IterableParameterModel
from runnable.graph import Graph, create_graph
from runnable.nodes import CompositeNode

logger = logging.getLogger(defaults.LOGGER_NAME)


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

    def get_summary(self) -> Dict[str, Any]:
        summary = {
            "name": self.name,
            "type": self.node_type,
            "branches": [branch.get_summary() for branch in self.branches.values()],
        }

        return summary

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

    def fan_out(
        self,
        iter_variable: Optional[IterableParameterModel] = None,
    ):
        """
        The general fan out method for a node of type Parallel.
        This method assumes that the step log has already been created.

        3rd party orchestrators should create the step log and use this method to create the branch logs.

        Args:
            executor (BaseExecutor): The executor class as defined by the config
            iter_variable (dict, optional): If the node is part of a map node. Defaults to None.
        """
        # Prepare the branch logs
        for internal_branch_name, _ in self.branches.items():
            effective_branch_name = self._resolve_map_placeholders(
                internal_branch_name, iter_variable=iter_variable
            )

            try:
                branch_log = self._context.run_log_store.get_branch_log(
                    effective_branch_name, self._context.run_id
                )
                console.print(f"Branch log already exists for {effective_branch_name}")
            except (exceptions.BranchLogNotFoundError, exceptions.EntityNotFoundError):
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
                self._execute_sequentially(iter_variable)
            else:
                logger.info("Executing branches in parallel")
                self._execute_in_parallel(iter_variable)
        else:
            self._execute_sequentially(iter_variable)

        self.fan_in(iter_variable=iter_variable)

    def _execute_sequentially(
        self,
        iter_variable: Optional[IterableParameterModel] = None,
    ):
        """Execute branches sequentially (original behavior)."""
        for _, branch in self.branches.items():
            self._context.pipeline_executor.execute_graph(
                branch, iter_variable=iter_variable
            )

    def _execute_in_parallel(
        self,
        iter_variable: Optional[IterableParameterModel] = None,
    ):
        """Execute branches in parallel using multiprocessing."""
        from runnable.entrypoints import execute_single_branch

        # Prepare arguments for each branch
        branch_args = []
        for branch_name, branch in self.branches.items():
            branch_args.append((branch_name, branch, self._context, iter_variable))

        # Use multiprocessing Pool to execute branches in parallel
        with Pool() as pool:
            results = pool.starmap(execute_single_branch, branch_args)

        # Check if any branch failed
        if not all(results):
            failed_branches = [
                branch_name
                for (branch_name, _, _, _), result in zip(branch_args, results)
                if not result
            ]
            logger.error(f"The following branches failed: {failed_branches}")
            # Note: The actual failure handling and status update will be done in fan_in()

    def fan_in(
        self,
        iter_variable: Optional[IterableParameterModel] = None,
    ):
        """
        The general fan in method for a node of type Parallel.

        3rd party orchestrators should use this method to find the status of the composite step.

        Args:
            executor (BaseExecutor): The executor class as defined by the config
            iter_variable (dict, optional): If the node is part of a map. Defaults to None.
        """
        effective_internal_name = self._resolve_map_placeholders(
            self.internal_name, iter_variable=iter_variable
        )
        step_success_bool = True
        for internal_branch_name, _ in self.branches.items():
            effective_branch_name = self._resolve_map_placeholders(
                internal_branch_name, iter_variable=iter_variable
            )
            branch_log = self._context.run_log_store.get_branch_log(
                effective_branch_name, self._context.run_id
            )

            if branch_log.status != defaults.SUCCESS:
                step_success_bool = False

        # Collate all the results and update the status of the step

        step_log = self._context.run_log_store.get_step_log(
            effective_internal_name, self._context.run_id
        )

        if step_success_bool:  #  If none failed
            step_log.status = defaults.SUCCESS
        else:
            step_log.status = defaults.FAIL

        self._context.run_log_store.add_step_log(step_log, self._context.run_id)

        # If we failed, return without parameter rollback
        if not step_log.status == defaults.SUCCESS:
            return

        # Roll back parameters from all branches to parent scope
        parent_params = self._context.run_log_store.get_parameters(
            self._context.run_id, internal_branch_name=self.internal_branch_name
        )

        for internal_branch_name, _ in self.branches.items():
            effective_branch_name = self._resolve_map_placeholders(
                internal_branch_name, iter_variable=iter_variable
            )

            branch_params = self._context.run_log_store.get_parameters(
                self._context.run_id, internal_branch_name=effective_branch_name
            )

            # Merge branch parameters into parent (overwrite with branch values)
            # If multiple branches set the same parameter, last one wins
            parent_params.update(branch_params)

        self._context.run_log_store.set_parameters(
            parameters=parent_params,
            run_id=self._context.run_id,
            internal_branch_name=self.internal_branch_name,
        )

    async def execute_as_graph_async(
        self,
        iter_variable: Optional[IterableParameterModel] = None,
    ):
        """Async parallel execution."""
        self.fan_out(iter_variable=iter_variable)  # sync - just creates branch logs

        for _, branch in self.branches.items():
            await self._context.pipeline_executor.execute_graph_async(
                branch, iter_variable=iter_variable
            )

        self.fan_in(iter_variable=iter_variable)  # sync - just collates status
