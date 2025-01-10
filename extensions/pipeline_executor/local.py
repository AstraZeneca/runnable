import logging

from pydantic import Field, PrivateAttr

from extensions.pipeline_executor import GenericPipelineExecutor
from runnable import defaults
from runnable.defaults import TypeMapVariable
from runnable.nodes import BaseNode

logger = logging.getLogger(defaults.LOGGER_NAME)


class LocalExecutor(GenericPipelineExecutor):
    """
    In the mode of local execution, we run everything on the local computer.

    This has some serious implications on the amount of time it would take to complete the run.
    Also ensure that the local compute is good enough for the compute to happen of all the steps.

    Example config:
    execution:
      type: local

    """

    service_name: str = "local"

    object_serialisation: bool = Field(default=True)

    _is_local: bool = PrivateAttr(default=True)

    def execute_from_graph(
        self, node: BaseNode, map_variable: TypeMapVariable = None, **kwargs
    ):
        if not self.object_serialisation:
            self._context.object_serialisation = False

        super().execute_from_graph(node=node, map_variable=map_variable, **kwargs)

    def trigger_node_execution(
        self, node: BaseNode, map_variable: TypeMapVariable = None, **kwargs
    ):
        """
        In this mode of execution, we prepare for the node execution and execute the node

        Args:
            node (BaseNode): [description]
            map_variable (str, optional): [description]. Defaults to ''.
        """
        self.execute_node(node=node, map_variable=map_variable, **kwargs)

    def execute_node(
        self, node: BaseNode, map_variable: TypeMapVariable = None, **kwargs
    ):
        """
        For local execution, we just execute the node.

        Args:
            node (BaseNode): _description_
            map_variable (dict[str, str], optional): _description_. Defaults to None.
        """
        self._execute_node(node=node, map_variable=map_variable, **kwargs)
