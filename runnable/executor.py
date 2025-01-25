from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, PrivateAttr

import runnable.context as context
from runnable import defaults
from runnable.datastore import DataCatalog, JobLog, StepLog
from runnable.defaults import TypeMapVariable

if TYPE_CHECKING:  # pragma: no cover
    from runnable.graph import Graph
    from runnable.nodes import BaseNode
    from runnable.tasks import BaseTaskType

logger = logging.getLogger(defaults.LOGGER_NAME)


class BaseExecutor(ABC, BaseModel):
    """
    The skeleton of an executor class.
    Any implementation of an executor should inherit this class and over-ride accordingly.

    There is a extension available in runnable/extensions/executor/__init__.py
    which implements the most common functionality which is easier to
    extend/override in most scenarios.

    """

    service_name: str = ""
    service_type: str = "executor"

    _is_local: bool = PrivateAttr(default=False)

    model_config = ConfigDict(extra="forbid")

    @property
    def _context(self):
        return context.run_context

    @abstractmethod
    def _get_parameters(self) -> Dict[str, Any]:
        """
        Get the parameters for the execution.
        The parameters can be defined in parameters file and can be overridden by environment variables.

        Returns:
            Dict[str, Any]: The parameters for the execution.
        """
        ...

    @abstractmethod
    def _set_up_run_log(self, exists_ok=False):
        """
        Create a run log and put that in the run log store

        If exists_ok, we allow the run log to be already present in the run log store.
        """
        ...

    # TODO: Make this attempt number
    @property
    def step_attempt_number(self) -> int:
        """
        The attempt number of the current step.
        Orchestrators should use this step to submit multiple attempts of the job.

        Returns:
            int: The attempt number of the current step. Defaults to 1.
        """
        return int(os.environ.get(defaults.ATTEMPT_NUMBER, 1))

    @abstractmethod
    def send_return_code(self, stage="traversal"):
        """
        Convenience function used by pipeline to send return code to the caller of the cli

        Raises:
            Exception: If the pipeline execution failed
        """
        ...

    @abstractmethod
    def add_task_log_to_catalog(
        self, name: str, map_variable: Optional[TypeMapVariable] = None
    ): ...


class BaseJobExecutor(BaseExecutor):
    service_type: str = "job_executor"

    @abstractmethod
    def submit_job(self, job: BaseTaskType, catalog_settings: Optional[List[str]]):
        """
        Local executors should
        - create the run log
        - and call an execute_job

        Non local executors should
        - transpile the job to the platform specific job spec
        - submit the job to call execute_job
        """
        ...

    @abstractmethod
    def add_code_identities(self, job_log: JobLog, **kwargs):
        """
        Add code identities specific to the implementation.

        The Base class has an implementation of adding git code identities.

        Args:
            step_log (object): The step log object
            node (BaseNode): The node we are adding the step log for
        """
        ...

    @abstractmethod
    def _sync_catalog(
        self,
        catalog_settings: Optional[List[str]],
    ) -> Optional[List[DataCatalog]]:
        """
        1). Identify the catalog settings by over-riding node settings with the global settings.
        2). For stage = get:
                Identify the catalog items that are being asked to get from the catalog
                And copy them to the local compute data folder
        3). For stage = put:
                Identify the catalog items that are being asked to put into the catalog
                Copy the items from local compute folder to the catalog
        4). Add the items onto the step log according to the stage

        Args:
            node (Node): The current node being processed
            step_log (StepLog): The step log corresponding to that node
            stage (str): One of get or put

        Raises:
            Exception: If the stage is not in one of get/put

        """
        ...

    @abstractmethod
    def execute_job(self, job: BaseTaskType, catalog_settings: Optional[List[str]]):
        """
        Focusses only on execution of the job.
        """
        ...


# TODO: Consolidate execute_node, trigger_node_execution, _execute_node
class BasePipelineExecutor(BaseExecutor):
    service_type: str = "pipeline_executor"
    overrides: dict = {}

    _context_node: Optional[BaseNode] = PrivateAttr(default=None)

    @abstractmethod
    def add_code_identities(self, node: BaseNode, step_log: StepLog, **kwargs):
        """
        Add code identities specific to the implementation.

        The Base class has an implementation of adding git code identities.

        Args:
            step_log (object): The step log object
            node (BaseNode): The node we are adding the step log for
        """
        ...

    @abstractmethod
    def get_effective_compute_data_folder(self) -> Optional[str]:
        """
        Get the effective compute data folder for the given stage.
        If there is nothing to catalog, we return None.

        The default is the compute data folder of the catalog but this can be over-ridden by the node.

        Args:
            stage (str): The stage we are in the process of cataloging


        Returns:
            Optional[str]: The compute data folder as defined by catalog handler or the node or None.
        """
        ...

    @abstractmethod
    def _sync_catalog(
        self, stage: str, synced_catalogs=None
    ) -> Optional[List[DataCatalog]]:
        """
        1). Identify the catalog settings by over-riding node settings with the global settings.
        2). For stage = get:
                Identify the catalog items that are being asked to get from the catalog
                And copy them to the local compute data folder
        3). For stage = put:
                Identify the catalog items that are being asked to put into the catalog
                Copy the items from local compute folder to the catalog
        4). Add the items onto the step log according to the stage

        Args:
            node (Node): The current node being processed
            step_log (StepLog): The step log corresponding to that node
            stage (str): One of get or put

        Raises:
            Exception: If the stage is not in one of get/put

        """
        ...

    @abstractmethod
    def _execute_node(
        self,
        node: BaseNode,
        map_variable: TypeMapVariable = None,
        mock: bool = False,
        **kwargs,
    ):
        """
        This is the entry point when we do the actual execution of the function.

        While in interactive execution, we just compute, in 3rd party interactive execution, we need to reach
        this function.

        In most cases,
            * We get the corresponding step_log of the node and the parameters.
            * We sync the catalog to GET any data sets that are in the catalog
            * We call the execute method of the node for the actual compute and retry it as many times as asked.
            * If the node succeeds, we get any of the user defined metrics provided by the user.
            * We sync the catalog to PUT any data sets that are in the catalog.

        Args:
            node (Node): The node to execute
            map_variable (dict, optional): If the node is of a map state, map_variable is the value of the iterable.
                        Defaults to None.
        """
        ...

    @abstractmethod
    def execute_node(
        self, node: BaseNode, map_variable: TypeMapVariable = None, **kwargs
    ):
        """
        The entry point for all executors apart from local.
        We have already prepared for node execution.

        Args:
            node (BaseNode): The node to execute
            map_variable (dict, optional): If the node is part of a map, send in the map dictionary. Defaults to None.

        Raises:
            NotImplementedError: _description_
        """
        ...

    @abstractmethod
    def execute_from_graph(
        self, node: BaseNode, map_variable: TypeMapVariable = None, **kwargs
    ):
        """
        This is the entry point to from the graph execution.

        While the self.execute_graph is responsible for traversing the graph, this function is responsible for
        actual execution of the node.

        If the node type is:
            * task : We can delegate to _execute_node after checking the eligibility for re-run in cases of a re-run
            * success: We can delegate to _execute_node
            * fail: We can delegate to _execute_node

        For nodes that are internally graphs:
            * parallel: Delegate the responsibility of execution to the node.execute_as_graph()
            * dag: Delegate the responsibility of execution to the node.execute_as_graph()
            * map: Delegate the responsibility of execution to the node.execute_as_graph()

        Transpilers will NEVER use this method and will NEVER call ths method.
        This method should only be used by interactive executors.

        Args:
            node (Node): The node to execute
            map_variable (dict, optional): If the node if of a map state, this corresponds to the value of iterable.
                    Defaults to None.
        """
        ...

    @abstractmethod
    def _get_status_and_next_node_name(
        self, current_node: BaseNode, dag: Graph, map_variable: TypeMapVariable = None
    ) -> tuple[str, str]:
        """
        Given the current node and the graph, returns the name of the next node to execute.

        The name is always relative the graph that the node resides in.

        If the current node succeeded, we return the next node as per the graph.
        If the current node failed, we return the on failure node of the node (if provided) or the global one.

        Args:
            current_node (BaseNode): The current node.
            dag (Graph): The dag we are traversing.
            map_variable (dict): If the node belongs to a map branch.
        """

        ...

    @abstractmethod
    def execute_graph(self, dag: Graph, map_variable: TypeMapVariable = None, **kwargs):
        """
        The parallelization is controlled by the nodes and not by this function.

        Transpilers should over ride this method to do the translation of dag to the platform specific way.
        Interactive methods should use this to traverse and execute the dag.
            - Use execute_from_graph to handle sub-graphs

        Logically the method should:
            * Start at the dag.start_at of the dag.
            * Call the self.execute_from_graph(node)
            * depending upon the status of the execution, either move to the success node or failure node.

        Args:
            dag (Graph): The directed acyclic graph to traverse and execute.
            map_variable (dict, optional): If the node if of a map state, this corresponds to the value of the iterable.
                    Defaults to None.
        """
        ...

    @abstractmethod
    def _resolve_executor_config(self, node: BaseNode) -> Dict[str, Any]:
        """
        The overrides section can contain specific over-rides to an global executor config.
        To avoid too much clutter in the dag definition, we allow the configuration file to have overrides block.
        The nodes can over-ride the global config by referring to key in the overrides.

        For example:
        # configuration.yaml
        execution:
          type: cloud-implementation
          config:
            k1: v1
            k3: v3
            overrides:
              k2: v2 # Could be a mapping internally.

        # in pipeline definition.yaml
        dag:
          steps:
            step1:
              overrides:
                cloud-implementation:
                  k1: value_specific_to_node
                  k2:

        This method should resolve the node_config to {'k1': 'value_specific_to_node', 'k2': 'v2', 'k3': 'v3'}

        Args:
            node (BaseNode): The current node being processed.

        """
        ...

    @abstractmethod
    def fan_out(self, node: BaseNode, map_variable: TypeMapVariable = None):
        """
        This method is used to appropriately fan-out the execution of a composite node.
        This is only useful when we want to execute a composite node during 3rd party orchestrators.

        Reason: Transpilers typically try to run the leaf nodes but do not have any capacity to do anything for the
        step which is composite. By calling this fan-out before calling the leaf nodes, we have an opportunity to
        do the right set up (creating the step log, exposing the parameters, etc.) for the composite step.

        All 3rd party orchestrators should use this method to fan-out the execution of a composite node.
        This ensures:
            - The dot path notation is preserved, this method should create the step and call the node's fan out to
            create the branch logs and let the 3rd party do the actual step execution.
            - Gives 3rd party orchestrators an opportunity to set out the required for running a composite node.

        Args:
            node (BaseNode): The node to fan-out
            map_variable (dict, optional): If the node if of a map state,.Defaults to None.

        """
        ...

    @abstractmethod
    def fan_in(self, node: BaseNode, map_variable: TypeMapVariable = None):
        """
        This method is used to appropriately fan-in after the execution of a composite node.
        This is only useful when we want to execute a composite node during 3rd party orchestrators.

        Reason: Transpilers typically try to run the leaf nodes but do not have any capacity to do anything for the
        step which is composite. By calling this fan-in after calling the leaf nodes, we have an opportunity to
        act depending upon the status of the individual branches.

        All 3rd party orchestrators should use this method to fan-in the execution of a composite node.
        This ensures:
            - Gives the renderer's the control on where to go depending upon the state of the composite node.
            - The status of the step and its underlying branches are correctly updated.

        Args:
            node (BaseNode): The node to fan-in
            map_variable (dict, optional): If the node if of a map state,.Defaults to None.

        """
        ...

    @abstractmethod
    def trigger_node_execution(
        self, node: BaseNode, map_variable: TypeMapVariable = None, **kwargs
    ):
        """
        Executor specific way of triggering jobs when runnable does both traversal and execution

        Transpilers will NEVER use this method and will NEVER call them.
        Only interactive executors who need execute_from_graph will ever implement it.

        Args:
            node (BaseNode): The node to execute
            map_variable (str, optional): If the node if of a map state, this corresponds to the value of iterable.
                    Defaults to ''.

        NOTE: We do not raise an exception as this method is not required by many extensions
        """
        ...
