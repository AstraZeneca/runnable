import logging
import os
from typing import List, Optional

from pydantic import Field, PrivateAttr

from extensions.pipeline_executor import GenericPipelineExecutor
from runnable import console, defaults
from runnable.datastore import DataCatalog
from runnable.defaults import IterableParameterModel
from runnable.graph import Graph
from runnable.nodes import BaseNode

logger = logging.getLogger(defaults.LOGGER_NAME)


class LocalExecutor(GenericPipelineExecutor):
    """
    In the mode of local execution, we run everything on the local computer.

    This has some serious implications on the amount of time it would take to complete the run.
    Also ensure that the local compute is good enough for the compute to happen of all the steps.

    Example config:

    ```yaml
    pipeline-executor:
      type: local
      config:
        enable_parallel: false  # Enable parallel execution for parallel/map nodes
    ```

    """

    service_name: str = "local"
    enable_parallel: bool = Field(default=False)

    # TODO: Not fully done
    object_serialisation: bool = Field(default=True)

    _is_local: bool = PrivateAttr(default=True)

    def execute_from_graph(
        self,
        node: BaseNode,
        iter_variable: Optional[IterableParameterModel] = None,
    ):
        if not self.object_serialisation:
            self._context.object_serialisation = False

        super().execute_from_graph(node=node, iter_variable=iter_variable)

    def trigger_node_execution(
        self,
        node: BaseNode,
        iter_variable: Optional[IterableParameterModel] = None,
    ):
        """
        In this mode of execution, we prepare for the node execution and execute the node

        Args:
            node (BaseNode): [description]
            iter_variable (str, optional): [description]. Defaults to ''.
        """
        self.execute_node(node=node, iter_variable=iter_variable)

    def execute_node(
        self,
        node: BaseNode,
        iter_variable: Optional[IterableParameterModel] = None,
    ):
        """
        For local execution, we just execute the node.

        Args:
            node (BaseNode): _description_
            iter_variable (dict[str, str], optional): _description_. Defaults to None.
        """
        self._execute_node(node=node, iter_variable=iter_variable)

    # ═══════════════════════════════════════════════════════════════
    # Async Path - implement async methods for local execution
    # ═══════════════════════════════════════════════════════════════

    async def execute_graph_async(
        self,
        dag: Graph,
        iter_variable: Optional[IterableParameterModel] = None,
    ):
        """Async graph traversal."""
        current_node = dag.start_at
        previous_node = None
        logger.info(f"Running async execution with {current_node}")

        branch_task_name: str = ""
        if dag.internal_branch_name:
            branch_task_name = BaseNode._resolve_map_placeholders(
                dag.internal_branch_name or "Graph",
                iter_variable,
            )
            console.print(
                f":runner: Executing the branch {branch_task_name} ... ",
                style="bold color(208)",
            )

        while True:
            working_on = dag.get_node_by_name(current_node)
            task_name = working_on._resolve_map_placeholders(
                working_on.internal_name, iter_variable
            )

            if previous_node == current_node:
                raise Exception("Potentially running in an infinite loop")
            previous_node = current_node

            try:
                await self.execute_from_graph_async(
                    working_on, iter_variable=iter_variable
                )
                # Sync helper - no await needed
                status, next_node_name = self._get_status_and_next_node_name(
                    current_node=working_on, dag=dag, iter_variable=iter_variable
                )

                if status == defaults.SUCCESS:
                    console.print(f":white_check_mark: Node {task_name} succeeded")
                else:
                    console.print(f":x: Node {task_name} failed")
            except Exception as e:
                console.print(":x: Error during execution", style="bold red")
                console.print(e, style=defaults.error_style)
                logger.exception(e)
                raise

            console.rule(style="[dark orange]")

            if working_on.node_type in ["success", "fail"]:
                break
            current_node = next_node_name

        # Sync helper - no await needed
        self._finalize_graph_execution(working_on, dag, iter_variable)

    async def execute_from_graph_async(
        self,
        node: BaseNode,
        iter_variable: Optional[IterableParameterModel] = None,
    ):
        """Async node execution entry point."""
        if not self.object_serialisation:
            self._context.object_serialisation = False

        # Sync helper - no await needed
        step_log = self._prepare_node_for_execution(node, iter_variable)
        if step_log is None:
            return  # Skipped

        logger.info(f"Executing node: {node.get_summary()}")

        if node.node_type in ["success", "fail"]:
            await self._execute_node_async(node, iter_variable=iter_variable)
            return

        if node.is_composite:
            await node.execute_as_graph_async(iter_variable=iter_variable)
            return

        task_name = node._resolve_map_placeholders(node.internal_name, iter_variable)
        console.print(
            f":runner: Executing the node {task_name} ... ", style="bold color(208)"
        )
        await self.trigger_node_execution_async(node=node, iter_variable=iter_variable)

    async def trigger_node_execution_async(
        self,
        node: BaseNode,
        iter_variable: Optional[IterableParameterModel] = None,
    ):
        """Async trigger for node execution."""
        await self._execute_node_async(node=node, iter_variable=iter_variable)

    async def _execute_node_async(
        self,
        node: BaseNode,
        iter_variable: Optional[IterableParameterModel] = None,
        mock: bool = False,
    ):
        """Async node execution wrapper."""
        current_attempt_number = self._calculate_attempt_number(node, iter_variable)
        os.environ[defaults.ATTEMPT_NUMBER] = str(current_attempt_number)

        logger.info(
            f"Trying to execute node: {node.internal_name}, attempt: {current_attempt_number}"
        )

        self._context_node = node

        # Sync - catalog get
        data_catalogs_get: Optional[List[DataCatalog]] = self._sync_catalog(stage="get")
        logger.debug(f"data_catalogs_get: {data_catalogs_get}")

        # ASYNC - execute the node
        step_log = await node.execute_async(
            iter_variable=iter_variable,
            attempt_number=current_attempt_number,
            mock=mock,
        )

        # Sync - catalog put and finalization
        allow_file_not_found_exc = step_log.status != defaults.SUCCESS
        data_catalogs_put: Optional[List[DataCatalog]] = self._sync_catalog(
            stage="put", allow_file_no_found_exc=allow_file_not_found_exc
        )
        logger.debug(f"data_catalogs_put: {data_catalogs_put}")
        step_log.add_data_catalogs(data_catalogs_put or [])
        step_log.add_data_catalogs(data_catalogs_get or [])

        console.print(f"Summary of the step: {step_log.internal_name}")
        console.print(step_log.get_summary(), style=defaults.info_style)

        self.add_task_log_to_catalog(
            name=self._context_node.internal_name, iter_variable=iter_variable
        )
        self._context_node = None

        self._context.run_log_store.add_step_log(step_log, self._context.run_id)
