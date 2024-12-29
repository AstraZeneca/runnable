import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

import runnable.context as context
from runnable import defaults, exceptions
from runnable.datastore import StepLog
from runnable.defaults import TypeMapVariable

logger = logging.getLogger(defaults.LOGGER_NAME)

# --8<-- [start:docs]


class BaseNode(ABC, BaseModel):
    """
    Base class with common functionality provided for a Node of a graph.

    A node of a graph could be a
        * single execution node as task, success, fail.
        * Could be graph in itself as parallel, dag and map.
        * could be a convenience function like as-is.

    The name is relative to the DAG.
    The internal name of the node, is absolute name in dot path convention.
        This has one to one mapping to the name in the run log
    The internal name of a node, should always be odd when split against dot.

    The internal branch name, only applies for branched nodes, is the branch it belongs to.
    The internal branch name should always be even when split against dot.
    """

    node_type: str = Field(serialization_alias="type")
    name: str
    internal_name: str = Field(exclude=True)
    internal_branch_name: str = Field(default="", exclude=True)
    is_composite: bool = Field(default=False, exclude=True)

    @property
    def _context(self):
        return context.run_context

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=False)

    @field_validator("name")
    @classmethod
    def validate_name(cls, name: str):
        if "." in name or "%" in name:
            raise ValueError("Node names cannot have . or '%' in them")
        return name

    def _command_friendly_name(
        self, replace_with=defaults.COMMAND_FRIENDLY_CHARACTER
    ) -> str:
        """
        Replace spaces with special character for spaces.
        Spaces in the naming of the node is convenient for the user but causes issues when used programmatically.

        Returns:
            str: The command friendly name of the node
        """
        return self.internal_name.replace(" ", replace_with)

    @classmethod
    def _get_internal_name_from_command_name(cls, command_name: str) -> str:
        """
        Replace runnable specific character (%) with whitespace.
        The opposite of _command_friendly_name.

        Args:
            command_name (str): The command friendly node name

        Returns:
            str: The internal name of the step
        """
        return command_name.replace(defaults.COMMAND_FRIENDLY_CHARACTER, " ")

    @classmethod
    def _resolve_map_placeholders(
        cls, name: str, map_variable: TypeMapVariable = None
    ) -> str:
        """
        If there is no map step used, then we just return the name as we find it.

        If there is a map variable being used, replace every occurrence of the map variable placeholder with
        the value sequentially.

        For example:
        1). dag:
              start_at: step1
              steps:
                step1:
                    type: map
                    iterate_on: y
                    iterate_as: y_i
                    branch:
                        start_at: map_step1
                        steps:
                            map_step1: # internal_name step1.placeholder.map_step1
                                type: task
                                command: a.map_func
                                command_type: python
                                next: map_success
                            map_success:
                                type: success
                            map_failure:
                                type: fail

            and if y is ['a', 'b', 'c'].

            This method would be called 3 times with map_variable = {'y_i': 'a'}, map_variable = {'y_i': 'b'} and
            map_variable = {'y_i': 'c'} corresponding to the three branches.

        For nested map branches, we would get the map_variables ordered hierarchically.

        Args:
            name (str): The name to resolve
            map_variable (dict): The dictionary of map variables

        Returns:
            [str]: The resolved name
        """
        if not map_variable:
            return name

        for _, value in map_variable.items():
            name = name.replace(defaults.MAP_PLACEHOLDER, str(value), 1)

        return name

    def _get_step_log_name(self, map_variable: TypeMapVariable = None) -> str:
        """
        For every step in the dag, there is a corresponding step log name.
        This method returns the step log name in dot path convention.

        All node types except a map state has a "static" defined step_log names and are equivalent to internal_name.
        For nodes belonging to map state, the internal name has a placeholder that is replaced at runtime.

        Args:
            map_variable (dict): If the node is of type map, the names are based on the current iteration state of the
            parameter.

        Returns:
            str: The dot path name of the step log name
        """
        return self._resolve_map_placeholders(
            self.internal_name, map_variable=map_variable
        )

    def _get_branch_log_name(self, map_variable: TypeMapVariable = None) -> str:
        """
        For nodes that are internally branches, this method returns the branch log name.
        The branch log name is in dot path convention.

        For nodes that are not map, the internal branch name is equivalent to the branch name.
        For map nodes, the internal branch name has a placeholder that is replaced at runtime.

        Args:
            map_variable (dict): If the node is of type map, the names are based on the current iteration state of the
            parameter.

        Returns:
            str: The dot path name of the branch log
        """
        return self._resolve_map_placeholders(
            self.internal_branch_name, map_variable=map_variable
        )

    def __str__(self) -> str:  # pragma: no cover
        """
        String representation of the node.

        Returns:
            str: The string representation of the node.
        """
        return f"Node of type {self.node_type} and name {self.internal_name}"

    @abstractmethod
    def _get_on_failure_node(self) -> str:
        """
        If the node defines a on_failure node in the config, return this or None.

        The naming is relative to the dag, the caller is supposed to resolve it to the correct graph

        Returns:
            str: The on_failure node defined by the dag or ''
        This is a base implementation which the BaseNode does not satisfy
        """

    @abstractmethod
    def _get_next_node(self) -> str:
        """
        Return the next node as defined by the config.

        Returns:
            str: The node name, relative to the dag, as defined by the config
        """

    @abstractmethod
    def _is_terminal_node(self) -> bool:
        """
        Returns whether a node has a next node

        Returns:
            bool: True or False of whether there is next node.
        """

    @abstractmethod
    def _get_catalog_settings(self) -> Dict[str, Any]:
        """
        If the node defines a catalog settings, return it or None

        Returns:
            dict: catalog settings defined as per the node or None
        """

    @abstractmethod
    def _get_branch_by_name(self, branch_name: str):
        """
        Retrieve a branch by name.

        The name is expected to follow a dot path convention.

        Args:
            branch_name (str): [description]

        Raises:
            Exception: [description]
        """

    def _get_neighbors(self) -> List[str]:
        """
        Gets the connecting neighbor nodes, either the "next" node or "on_failure" node.

        Returns:
            list: List of connected neighbors for a given node. Empty if terminal node.
        """
        neighbors = []
        try:
            next_node = self._get_next_node()
            neighbors += [next_node]
        except exceptions.TerminalNodeError:
            pass

        try:
            fail_node = self._get_on_failure_node()
            if fail_node:
                neighbors += [fail_node]
        except exceptions.TerminalNodeError:
            pass

        return neighbors

    @abstractmethod
    def _get_executor_config(self, executor_type: str) -> str:
        """
        Return the executor config of the node, if defined, or empty dict

        Args:
            executor_type (str): The executor type that the config refers to.

        Returns:
            dict: The executor config, if defined or an empty dict
        """

    @abstractmethod
    def _get_max_attempts(self) -> int:
        """
        The number of max attempts as defined by the config or 1.

        Returns:
            int: The number of maximum retries as defined by the config or 1.
        """

    @abstractmethod
    def execute(
        self,
        mock=False,
        map_variable: TypeMapVariable = None,
        attempt_number: int = 1,
        **kwargs,
    ) -> StepLog:
        """
        The actual function that does the execution of the command in the config.

        Should only be implemented for task, success, fail and as-is and never for
        composite nodes.

        Args:
            executor (runnable.executor.BaseExecutor): The executor class
            mock (bool, optional): Don't run, just pretend. Defaults to False.
            map_variable (str, optional): The value of the map iteration variable, if part of a map node.
                Defaults to ''.

        Raises:
            NotImplementedError: Base class, hence not implemented.
        """

    @abstractmethod
    def execute_as_graph(self, map_variable: TypeMapVariable = None, **kwargs):
        """
        This function would be called to set up the execution of the individual
        branches of a composite node.

        Function should only be implemented for composite nodes like dag, map, parallel.

        Args:
            executor (runnable.executor.BaseExecutor): The executor.

        Raises:
            NotImplementedError: Base class, hence not implemented.
        """

    @abstractmethod
    def fan_out(self, map_variable: TypeMapVariable = None, **kwargs):
        """
        This function would be called to set up the execution of the individual
        branches of a composite node.

        Function should only be implemented for composite nodes like dag, map, parallel.

        Args:
            executor (runnable.executor.BaseExecutor): The executor.
            map_variable (str, optional): The value of the map iteration variable, if part of a map node.

        Raises:
            Exception: If the node is not a composite node.
        """

    @abstractmethod
    def fan_in(self, map_variable: TypeMapVariable = None, **kwargs):
        """
        This function would be called to tear down the execution of the individual
        branches of a composite node.

        Function should only be implemented for composite nodes like dag, map, parallel.

        Args:
            executor (runnable.executor.BaseExecutor): The executor.
            map_variable (str, optional): The value of the map iteration variable, if part of a map node.

        Raises:
            Exception: If the node is not a composite node.
        """

    @classmethod
    @abstractmethod
    def parse_from_config(cls, config: Dict[str, Any]) -> "BaseNode":
        """
        Parse the config from the user and create the corresponding node.

        Args:
            config (Dict[str, Any]): The config of the node from the yaml or from the sdk.

        Returns:
            BaseNode: The corresponding node.
        """

    @abstractmethod
    def get_summary(self) -> Dict[str, Any]:
        """
        Return the summary of the node

        Returns:
            Dict[str, Any]: _description_
        """


# --8<-- [end:docs]
class TraversalNode(BaseNode):
    next_node: str = Field(serialization_alias="next")
    on_failure: str = Field(default="")
    overrides: Dict[str, str] = Field(default_factory=dict)

    def _get_on_failure_node(self) -> str:
        """
        If the node defines a on_failure node in the config, return this or None.

        The naming is relative to the dag, the caller is supposed to resolve it to the correct graph

        Returns:
            str: The on_failure node defined by the dag or ''
        This is a base implementation which the BaseNode does not satisfy
        """
        return self.on_failure

    def _get_next_node(self) -> str:
        """
        Return the next node as defined by the config.

        Returns:
            str: The node name, relative to the dag, as defined by the config
        """

        return self.next_node

    def _is_terminal_node(self) -> bool:
        """
        Returns whether a node has a next node

        Returns:
            bool: True or False of whether there is next node.
        """
        return False

    def _get_executor_config(self, executor_type) -> str:
        return self.overrides.get(executor_type) or ""


class CatalogStructure(BaseModel):
    model_config = ConfigDict(extra="forbid")  # Need to forbid

    get: List[str] = Field(default_factory=list)
    put: List[str] = Field(default_factory=list)


class ExecutableNode(TraversalNode):
    catalog: Optional[CatalogStructure] = Field(default=None)
    max_attempts: int = Field(default=1, ge=1)

    def _get_catalog_settings(self) -> Dict[str, Any]:
        """
        If the node defines a catalog settings, return it or None

        Returns:
            dict: catalog settings defined as per the node or None
        """
        if self.catalog:
            return self.catalog.model_dump()
        return {}

    def _get_max_attempts(self) -> int:
        return self.max_attempts

    def _get_branch_by_name(self, branch_name: str):
        raise exceptions.NodeMethodCallError(
            "This is an executable node and does not have branches"
        )

    def execute_as_graph(self, map_variable: TypeMapVariable = None, **kwargs):
        raise exceptions.NodeMethodCallError(
            "This is an executable node and does not have a graph"
        )

    def fan_in(self, map_variable: TypeMapVariable = None, **kwargs):
        raise exceptions.NodeMethodCallError(
            "This is an executable node and does not have a fan in"
        )

    def fan_out(self, map_variable: TypeMapVariable = None, **kwargs):
        raise exceptions.NodeMethodCallError(
            "This is an executable node and does not have a fan out"
        )

    def prepare_for_job_execution(self):
        raise exceptions.NodeMethodCallError(
            "This is an executable node and does not have a prepare_for_job_execution"
        )

    def tear_down_after_job_execution(self):
        raise exceptions.NodeMethodCallError(
            "This is an executable node and does not have a tear_down_after_job_execution",
        )


class CompositeNode(TraversalNode):
    def _get_catalog_settings(self) -> Dict[str, Any]:
        """
        If the node defines a catalog settings, return it or None

        Returns:
            dict: catalog settings defined as per the node or None
        """
        raise exceptions.NodeMethodCallError(
            "This is a composite node and does not have a catalog settings"
        )

    def _get_max_attempts(self) -> int:
        raise Exception("This is a composite node and does not have a max_attempts")

    def execute(
        self,
        mock=False,
        map_variable: TypeMapVariable = None,
        attempt_number: int = 1,
        **kwargs,
    ) -> StepLog:
        raise exceptions.NodeMethodCallError(
            "This is a composite node and does not have an execute function"
        )

    def prepare_for_job_execution(self):
        raise exceptions.NodeMethodCallError(
            "This is an executable node and does not have a prepare_for_job_execution"
        )

    def tear_down_after_job_execution(self):
        raise exceptions.NodeMethodCallError(
            "This is an executable node and does not have a tear_down_after_job_execution"
        )


class TerminalNode(BaseNode):
    def _get_on_failure_node(self) -> str:
        return ""

    def _get_next_node(self) -> str:
        raise exceptions.TerminalNodeError()

    def _is_terminal_node(self) -> bool:
        return True

    def _get_catalog_settings(self) -> Dict[str, Any]:
        raise exceptions.TerminalNodeError()

    def _get_branch_by_name(self, branch_name: str):
        raise exceptions.TerminalNodeError()

    def _get_executor_config(self, executor_type) -> str:
        raise exceptions.TerminalNodeError()

    def _get_max_attempts(self) -> int:
        return 1

    def execute_as_graph(self, map_variable: TypeMapVariable = None, **kwargs):
        raise exceptions.TerminalNodeError()

    def fan_in(self, map_variable: TypeMapVariable = None, **kwargs):
        raise exceptions.TerminalNodeError()

    def fan_out(self, map_variable: TypeMapVariable = None, **kwargs):
        raise exceptions.TerminalNodeError()

    @classmethod
    def parse_from_config(cls, config: Dict[str, Any]) -> "TerminalNode":
        return cls(**config)
