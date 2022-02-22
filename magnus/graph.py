import logging
from typing import TYPE_CHECKING, Dict, List

from stevedore import driver

from magnus import defaults, exceptions

if TYPE_CHECKING:
    from magnus.nodes import BaseNode

logger = logging.getLogger(defaults.NAME)


class Graph:
    """
    A class representing a graph.

    The representation is similar to AWS step functions.
    We have nodes and traversal is based on start_at and on_failure definition of individual nodes of the graph
    """

    def __init__(self, start_at, description=None, max_time=86400, internal_branch_name=None):
        self.start_at = start_at
        self.description = description
        self.max_time = max_time
        self.internal_branch_name = internal_branch_name
        self.nodes = []

    def get_node_by_name(self, name: str) -> 'BaseNode':
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
        for node in self.nodes:
            if node.name == name:
                return node
        raise exceptions.NodeNotFoundError(name)

    def get_node_by_internal_name(self, internal_name: str) -> 'BaseNode':
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
        for node in self.nodes:
            if node.internal_name == internal_name:
                return node
        raise exceptions.NodeNotFoundError(internal_name)

    def __str__(self):  # pragma: no cover
        node_str = ', '.join([x.name for x in self.nodes])
        return f'Starts at: {self.start_at} and has a max run time of {self.max_time} and {node_str}'

    def add_node(self, node: 'BaseNode'):
        """
        Add a node to the nodes of the graph

        Args:
            node (object): The node to add
        """
        self.nodes.append(node)

    def validate(self):
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
            message = 'The graph has references to nodes (next, on_failure), these nodes are missing from the DAG:\n'
            message += f'{", ".join(missing_nodes)}'
            logger.error(message)
            messages.append(message)

        if not missing_nodes:
            if not self.is_dag():
                message = 'The DAG is cyclic or does not reach an end state'
                logger.error(message)
                messages.append(message)

        if not self.is_start_node_present():
            message = 'The start node is not part of the graph'
            logger.error(message)
            messages.append(message)

        if not self.success_node_validation():
            message = 'There should be exactly one success node'
            logger.error(message)
            messages.append(message)

        if not self.fail_node_validation():
            message = 'There should be exactly one fail node'
            logger.error(message)
            messages.append(message)

        if messages:
            raise Exception(',  '.join(messages))

    def get_success_node(self) -> 'BaseNode':
        """
        Return the success node of the graph

        Raises:
            Exception: If no success node is present in the graph

        Returns:
            object: The success node
        """
        for node in self.nodes:
            if node.node_type == 'success':
                return node
        raise Exception('No success node defined')

    def get_fail_node(self) -> 'BaseNode':
        """
        Returns the fail node of the graph

        Raises:
            Exception: If no fail  node is present in the graph

        Returns:
            object: The fail node of the graph
        """
        for node in self.nodes:
            if node.node_type == 'fail':
                return node
        raise Exception('No fail node defined')

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
            logger.exception('Could not find the node')
            return False

    def success_node_validation(self) -> bool:
        """
        Check to ensure there is one and only one success node in the graph

        Returns:
            bool: True if there is only one, false otherwise
        """
        node_count = 0
        for node in self.nodes:
            if node.node_type == 'success':
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
        for node in self.nodes:
            if node.node_type == 'fail':
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
        visited = {n.name: False for n in self.nodes}
        recstack = {n.name: False for n in self.nodes}

        for node in self.nodes:
            if not visited[node.name]:
                if self.is_cyclic_util(node, visited, recstack):
                    return False
        return True

    def is_cyclic_util(self, node: 'BaseNode', visited: Dict[str, bool], recstack: Dict[str, bool]) -> bool:
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

        neighbors = node.get_neighbors()
        for neighbor in neighbors:
            neighbor_node = self.get_node_by_name(neighbor)
            if not visited[neighbor]:
                if self.is_cyclic_util(neighbor_node, visited, recstack):
                    return True
            elif recstack[neighbor]:
                return True

        recstack[node.name] = False
        return False

    def missing_neighbors(self) -> List['BaseNode']:
        """
        Iterates through nodes and gets their connecting neighbors and checks if they exist in the graph.

        Returns:
            list: List of the missing nodes. Empty list if all neighbors are in the graph.
        """
        missing_nodes = []
        for node in self.nodes:
            neighbors = node.get_neighbors()
            for neighbor in neighbors:
                try:
                    self.get_node_by_name(neighbor)
                except exceptions.NodeNotFoundError:
                    logger.exception('Could not find the node')
                    if neighbor not in missing_nodes:
                        missing_nodes.append(neighbor)
        return missing_nodes


def create_graph(dag_config: dict, internal_branch_name: str = None) -> Graph:
    # pylint: disable=R0914,R0913
    """
    Creates a dag object from the dag definition.

    Composite nodes like map, parallel, dag can have sub-branches which are internally graphs.
    Use internal_branch_name to fit the right dot path convention.

    Args:
        dag_config (dict): The dag defintion
        internal_branch_name ([type], optional): In case of sub-graph, the name of the node. Defaults to None.

    Raises:
        Exception: If the node or graph validation fails.

    Returns:
        Graph: The created graph object
    """
    # Conditional import to avoid circular import
    # pylint: disable=C0415
    from magnus.nodes import validate_node

    description = dag_config.get('description', None)
    max_time = dag_config.get('max_time', defaults.MAX_TIME)
    start_at = dag_config.get('start_at')  # Let the start_at be relative to the graph

    graph = Graph(start_at=start_at, description=description,
                  max_time=max_time, internal_branch_name=internal_branch_name)

    logger.info(
        f'Initialized a graph object that starts at {start_at} and runs for maximum of {max_time} secs')
    messages = []
    for step in dag_config.get('steps', []):
        step_config = dag_config['steps'][step]
        logger.info(f'Adding node {step} with :{step_config}')

        task_type = step_config.get('command_type', defaults.COMMAND_TYPE)
        command_config = step_config.get('command_config', {})

        logger.info(f"Trying to get a task of type {task_type}")
        try:
            task_mgr = driver.DriverManager(
                namespace="magnus.tasks.BaseTaskType",
                name=task_type,
                invoke_on_load=True,
                invoke_kwds={'command': step_config.get('command', None),
                             'config': command_config}
            )
        except Exception as _e:
            msg = (
                f"Could not find the task type {task_type}. Please ensure you have installed the extension that"
                " provides the task type. \nCore supports: python(default), python-lambda, shell, notebook"
            )
            raise Exception(msg) from _e

        internal_name = step
        if internal_branch_name:
            internal_name = internal_branch_name + '.' + step

        try:
            node_mgr = driver.DriverManager(
                namespace="magnus.nodes.BaseNode",
                name=step_config['type'],
                invoke_on_load=True,
                invoke_kwds={"name": step, "internal_name": internal_name,
                             "config": step_config, "execution_type": task_mgr.driver,
                             "internal_branch_name": internal_branch_name}
            )
        except Exception as _e:
            msg = (
                f"Could not find the node type {step_config['type']}. Please ensure you have installed "
                "the extension that provides the node type."
                "\nCore supports: task, success, fail, parallel, dag, map, as-is")
            raise Exception(msg) from _e

        messages.extend(validate_node(node_mgr.driver))
        graph.add_node(node_mgr.driver)

    if messages:
        raise Exception(', '.join(messages))

    graph.validate()

    return graph


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
    dot_path = internal_name.split('.')
    if len(dot_path) == 1:
        return dag.get_node_by_internal_name(internal_name), dag

    # Any node internal name is: And is always going to be odd in number when split against .
    # Step.Branch.Step.Branch etc
    current_node = None
    current_branch = dag

    for i in range(len(dot_path)):
        if i % 2:
            # Its odd, so we are in brach name
            current_branch = current_node.get_branch_by_name('.'.join(dot_path[:i + 1]))  # type: ignore
            logger.debug(f'Finding step for {internal_name} in branch: {current_branch}')
        else:
            # Its even, so we are in Step, we start here!
            current_node = current_branch.get_node_by_internal_name('.'.join(dot_path[:i + 1]))
            logger.debug(f'Finding {internal_name} in node: {current_node}')

    logger.debug(f'current branch : {current_branch}, current step {current_node}')
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
    dot_path = internal_name.split('.')
    if len(dot_path) == 1:
        return dag

    # Any branch internal name is: And is always going to be even in number when split against .
    # Step.Branch.Step.Branch
    current_node = None
    current_branch = dag

    for i in range(len(dot_path)):

        if i % 2:
            # Its odd, so we are in brach name
            current_branch = current_node.get_branch_by_name('.'.join(dot_path[:i + 1]))  # type: ignore
            logger.debug(f'Finding step for {internal_name} in branch: {current_branch}')

        else:
            # Its even, so we are in Step, we start here!
            current_node = current_branch.get_node_by_internal_name('.'.join(dot_path[:i + 1]))
            logger.debug(f'Finding {internal_name} in node: {current_node}')

    logger.debug(f'current branch : {current_branch}, current step {current_node}')
    if current_branch and current_node:
        return current_branch

    raise exceptions.BranchNotFoundError(internal_name)
