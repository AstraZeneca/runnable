from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, cast

from pydantic import BaseModel, Field, SerializeAsAny
from stevedore import driver

from runnable import defaults, exceptions

logger = logging.getLogger(defaults.LOGGER_NAME)
logging.getLogger("stevedore").setLevel(logging.CRITICAL)


class Graph(BaseModel):
    """
    A class representing a graph.

    The representation is similar to AWS step functions.
    We have nodes and traversal is based on start_at and on_failure definition of individual nodes of the graph
    """

    start_at: str
    name: str = ""
    description: Optional[str] = ""
    internal_branch_name: str = Field(default="", exclude=True)
    nodes: SerializeAsAny[Dict[str, "BaseNode"]] = Field(
        default_factory=dict, serialization_alias="steps"
    )

    def get_summary(self) -> Dict[str, Any]:
        """
        Return a summary of the graph
        """
        return {
            "name": self.name,
            "description": self.description,
            "start_at": self.start_at,
            "nodes": [node.get_summary() for node in list(self.nodes.values())],
        }

    def get_node_by_name(self, name: str) -> "BaseNode":
        """
        Return the Node object by the name
        The name is always relative to the graph

        Args:
            name (str): Name of the node

        Raises:
            NodeNotFoundError: If the node of name is not found in the graph

        Returns:
            Node: The Node object by name
        """
        for key, value in self.nodes.items():
            if key == name:
                return value
        raise exceptions.NodeNotFoundError(name)

    def get_node_by_internal_name(self, internal_name: str) -> "BaseNode":
        """
        Return the node by the internal name of the node.
        The internal name uses dot path convention.
        This method is only relative to the nodes of the graph and does not perform graph search of sub-graphs

        Args:
            internal_name (str): The internal name of the node, follows a dot path convention

        Raises:
            NodeNotFoundError: If the node of the internal name is not found in the graph

        Returns:
            Node: The Node object by the name
        """
        for _, value in self.nodes.items():
            if value.internal_name == internal_name:
                return value
        raise exceptions.NodeNotFoundError(internal_name)

    def __str__(self):  # pragma: no cover
        """
        Return a string representation of the graph
        """
        node_str = ", ".join([x.name for x in list(self.nodes.values())])
        return f"Starts at: {self.start_at} and {node_str}"

    def add_node(self, node: "BaseNode"):
        """
        Add a node to the nodes of the graph

        Args:
            node (object): The node to add
        """
        self.nodes[node.name] = node

    def check_graph(self):
        """
        Validate the graph to make sure,
        1). All the neighbors of nodes are present.
        2). Detection of cycles.
        3). Confirming that the start_at is actually a node present in graph
        4). Detection of one and only one success node.
        5). Detection of one and only one fail node.
        6). Missing nodes if referred by next or on_failure

        Raises:
            Exception: [description]
        """
        messages = []

        missing_nodes = self.missing_neighbors()
        if missing_nodes:
            message = "The graph has references to nodes (next, on_failure), these nodes are missing from the DAG:\n"
            message += f'{", ".join(missing_nodes)}'
            logger.error(message)
            messages.append(message)

        if not missing_nodes:
            if not self.is_dag():
                message = "The DAG is cyclic or does not reach an end state"
                logger.error(message)
                messages.append(message)

        if not self.is_start_node_present():
            message = "The start node is not part of the graph"
            logger.error(message)
            messages.append(message)

        if not self.success_node_validation():
            message = "There should be exactly one success node"
            logger.error(message)
            messages.append(message)

        if not self.fail_node_validation():
            message = "There should be exactly one fail node"
            logger.error(message)
            messages.append(message)

        if messages:
            raise Exception(",  ".join(messages))

    def get_success_node(self) -> "BaseNode":
        """
        Return the success node of the graph

        Raises:
            Exception: If no success node is present in the graph

        Returns:
            object: The success node
        """
        for _, value in self.nodes.items():
            if value.node_type == "success":
                return value
        raise Exception("No success node defined")

    def get_fail_node(self) -> "BaseNode":
        """
        Returns the fail node of the graph

        Raises:
            Exception: If no fail  node is present in the graph

        Returns:
            object: The fail node of the graph
        """
        for _, value in self.nodes.items():
            if value.node_type == "fail":
                return value
        raise Exception("No fail node defined")

    def is_start_node_present(self) -> bool:
        """
        A check to ensure the start_at is part of the graph

        Returns:
            bool: True if start_at is one of the nodes, false otherwise
        """
        try:
            self.get_node_by_name(self.start_at)
            return True
        except exceptions.NodeNotFoundError:
            logger.exception("Could not find the node")
            return False

    def success_node_validation(self) -> bool:
        """
        Check to ensure there is one and only one success node in the graph

        Returns:
            bool: True if there is only one, false otherwise
        """
        node_count = 0
        for _, value in self.nodes.items():
            if value.node_type == "success":
                node_count += 1
        if node_count == 1:
            return True
        return False

    def fail_node_validation(self) -> bool:
        """
        Check to make sure there is one and only one fail node in the graph

        Returns:
            bool: true if there is one and only one fail node, false otherwise
        """
        node_count = 0
        for _, value in self.nodes.items():
            if value.node_type == "fail":
                node_count += 1
        if node_count == 1:
            return True
        return False

    def is_dag(self) -> bool:
        """
        Determines whether the graph is acyclic and directed

        Returns:
            bool: Returns True if it is directed and acyclic.
        """
        visited = {n: False for n in self.nodes.keys()}
        recstack = {n: False for n in self.nodes.keys()}

        for name, node in self.nodes.items():
            if not visited[name]:
                if self.is_cyclic_util(node, visited, recstack):
                    return False
        return True

    def is_cyclic_util(
        self, node: "BaseNode", visited: Dict[str, bool], recstack: Dict[str, bool]
    ) -> bool:
        """
        Recursive utility that determines if a node and neighbors has a cycle. Is used in is_dag method.

        Args:
            node (BaseNode): The node to check
            visited (dict): Dictionary storing which nodes have been checked
            recstack (dict): Stores what nodes have been visited recursively

        Returns:
            bool: True if cyclic.
        """
        visited[node.name] = True
        recstack[node.name] = True

        neighbors = node._get_neighbors()
        for neighbor in neighbors:
            neighbor_node = self.get_node_by_name(neighbor)
            if not visited[neighbor]:
                if self.is_cyclic_util(neighbor_node, visited, recstack):
                    return True
            elif recstack[neighbor]:
                return True

        recstack[node.name] = False
        return False

    def missing_neighbors(self) -> List[str]:
        """
        Iterates through nodes and gets their connecting neighbors and checks if they exist in the graph.

        Returns:
            list: List of the missing nodes. Empty list if all neighbors are in the graph.
        """
        missing_nodes = []
        for _, node in self.nodes.items():
            neighbors = node._get_neighbors()
            for neighbor in neighbors:
                try:
                    self.get_node_by_name(neighbor)
                except exceptions.NodeNotFoundError:
                    logger.exception(f"Could not find the node {neighbor}")
                    if neighbor not in missing_nodes:
                        missing_nodes.append(neighbor)
        return missing_nodes

    def add_terminal_nodes(
        self,
        success_node_name: str = "success",
        failure_node_name: str = "fail",
        internal_branch_name: str = "",
    ):
        """
        Add the success and fail nodes to the graph

        Args:
            success_node_name (str, optional): The name of the success node. Defaults to 'success'.
            failure_node_name (str, optional): The name of the failure node. Defaults to 'fail'.
        """
        success_step_config = {"type": "success"}
        success_node = create_node(
            success_node_name,
            step_config=success_step_config,
            internal_branch_name=internal_branch_name,
        )
        fail_step_config = {"type": "fail"}
        fail_node = create_node(
            failure_node_name,
            step_config=fail_step_config,
            internal_branch_name=internal_branch_name,
        )
        self.add_node(success_node)
        self.add_node(fail_node)


from runnable.nodes import BaseNode  # noqa: E402

Graph.model_rebuild()


def create_graph(dag_config: Dict[str, Any], internal_branch_name: str = "") -> Graph:
    """
    Creates a dag object from the dag definition.

    Composite nodes like map, parallel, dag can have sub-branches which are internally graphs.
    Use internal_branch_name to fit the right dot path convention.

    Args:
        dag_config (dict): The dag definition
        internal_branch_name ([type], optional): In case of sub-graph, the name of the node. Defaults to None.

    Raises:
        Exception: If the node or graph validation fails.

    Returns:
        Graph: The created graph object
    """
    description: str = dag_config.get("description", None)
    start_at: str = cast(
        str, dag_config.get("start_at")
    )  # Let the start_at be relative to the graph

    graph = Graph(
        start_at=start_at,
        description=description,
        internal_branch_name=internal_branch_name,
    )

    logger.info(f"Initialized a graph object that starts at {start_at}")
    for name, step_config in dag_config.get("steps", {}).items():
        logger.info(f"Adding node {name} with :{step_config}")

        node = create_node(
            name, step_config=step_config, internal_branch_name=internal_branch_name
        )
        graph.add_node(node)

    graph.add_terminal_nodes(internal_branch_name=internal_branch_name)

    graph.check_graph()

    return graph


def create_node(name: str, step_config: dict, internal_branch_name: Optional[str] = ""):
    """
    Creates a node object from the step configuration.

    Args:
        name (str): The name of the node
        step_config (dict): The configuration of the node
        internal_branch_name (str, optional): If the node belongs to a internal branch. Defaults to None.

    Raises:
        Exception: If the node type is not supported

    Returns:
        BaseNode: The created node object
    """
    internal_name = name
    if internal_branch_name:
        internal_name = internal_branch_name + "." + name

    try:
        node_type = step_config.pop(
            "type"
        )  # Remove the type as it is not used in node creation.
        node_mgr: BaseNode = driver.DriverManager(
            namespace="nodes", name=node_type
        ).driver

        next_node = step_config.pop("next", None)

        if next_node:
            step_config["next_node"] = next_node

        invoke_kwds = {
            "name": name,
            "internal_name": internal_name,
            "internal_branch_name": internal_branch_name,
            **step_config,
        }
        node = node_mgr.parse_from_config(config=invoke_kwds)
        return node
    except KeyError:
        msg = "The node configuration does not contain the required key 'type'."
        logger.exception(step_config)
        raise Exception(msg)
    except Exception as _e:
        msg = (
            f"Could not find the node type {node_type}. Please ensure you have installed "
            "the extension that provides the node type."
            "\nCore supports: task, success, fail, parallel, dag, map, stub"
        )
        raise Exception(msg) from _e


def search_node_by_internal_name(dag: Graph, internal_name: str):
    """
    Given a DAG, search the node by internal name of the node.

    The node naming convention follows dot path naming convention

    Currently it is implemented to search only against the base dag.

    Args:
        dag (Graph): The graph to search the node
        internal_name (str): The internal name of the node.
    """
    # If the node is not part of any branches, then the base graph is where the node belongs
    dot_path = internal_name.split(".")
    if len(dot_path) == 1:
        return dag.get_node_by_internal_name(internal_name), dag

    # Any node internal name is: And is always going to be odd in number when split against .
    # Step.Branch.Step.Branch etc
    current_node = None
    current_branch = dag

    for i in range(len(dot_path)):
        if i % 2:
            # Its odd, so we are in brach name

            current_branch = current_node._get_branch_by_name(  # type: ignore
                ".".join(dot_path[: i + 1])
            )
            logger.debug(
                f"Finding step for {internal_name} in branch: {current_branch}"
            )
        else:
            # Its even, so we are in Step, we start here!
            current_node = current_branch.get_node_by_internal_name(
                ".".join(dot_path[: i + 1])
            )
            logger.debug(f"Finding {internal_name} in node: {current_node}")

    logger.debug(f"current branch : {current_branch}, current step {current_node}")
    if current_branch and current_node:
        return current_node, current_branch

    raise exceptions.NodeNotFoundError(internal_name)


def search_branch_by_internal_name(dag: Graph, internal_name: str):
    """
    Given a DAG, search the branch by internal name of the branch.

    The branch naming convention follows dot path naming convention

    Currently it is implemented to search only against the base dag.

    Args:
        dag (Graph): The graph to search the node
        internal_name (str): The internal name of the branch.
    """
    # If the node is not part of any branches, then the base graph is where the node belongs
    dot_path = internal_name.split(".")
    if len(dot_path) == 1:
        return dag

    # Any branch internal name is: And is always going to be even in number when split against .
    # Step.Branch.Step.Branch
    current_node = None
    current_branch = dag

    for i in range(len(dot_path)):
        if i % 2:
            # Its odd, so we are in brach name
            current_branch = current_node._get_branch_by_name(  # type: ignore
                ".".join(dot_path[: i + 1])
            )
            logger.debug(
                f"Finding step for {internal_name} in branch: {current_branch}"
            )

        else:
            # Its even, so we are in Step, we start here!
            current_node = current_branch.get_node_by_internal_name(
                ".".join(dot_path[: i + 1])
            )
            logger.debug(f"Finding {internal_name} in node: {current_node}")

    logger.debug(f"current branch : {current_branch}, current step {current_node}")
    if current_branch and current_node:
        return current_branch

    raise exceptions.BranchNotFoundError(internal_name)
