import json
import logging
import multiprocessing
from collections import OrderedDict
from datetime import datetime
from typing import List, Type, Union

from pkg_resources import resource_filename

import magnus
from magnus import defaults, utils
from magnus.graph import create_graph

logger = logging.getLogger(defaults.NAME)


class BaseNode:
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

    node_type = ''

    def __init__(self, name, internal_name, config, execution_type, internal_branch_name=None):
        # pylint: disable=R0914,R0913
        self.name = name
        self.internal_name = internal_name  #  Dot notation naming of the steps
        self.config = config
        self.internal_branch_name = internal_branch_name  # parallel, map, dag only have internal names
        self.execution_type = execution_type
        self.branches = None
        self.is_composite = False

    def command_friendly_name(self, replace_with=defaults.COMMAND_FRIENDLY_CHARACTER) -> str:
        """
        Replace spaces with special character for spaces.
        Spaces in the naming of the node is convenient for the user but causes issues when used programmatically.

        Returns:
            str: The command friendly name of the node
        """
        return self.internal_name.replace(' ', replace_with)

    @ classmethod
    def get_internal_name_from_command_name(cls, command_name: str) -> str:
        """
        Replace Magnus specific character (%) with whitespace.
        The opposite of command_friendly_name.

        Args:
            command_name (str): The command friendly node name

        Returns:
            str: The internal name of the step
        """
        return command_name.replace(defaults.COMMAND_FRIENDLY_CHARACTER, ' ')

    @ classmethod
    def resolve_map_placeholders(cls, name: str, map_variable: dict = None) -> str:
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

        For nested map branches, we would get the map_variables ordered heirarchically.

        Args:
            name (str): The name to resolve
            map_variable (dict): The dictionary of map variables

        Returns:
            [str]: The resolved name
        """
        if not map_variable:
            return name

        for _, value in map_variable.items():
            name = name.replace(defaults.MAP_PLACEHOLDER, value, 1)

        return name

    def get_step_log_name(self, map_variable: dict = None) -> str:
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
        return self.resolve_map_placeholders(self.internal_name, map_variable=map_variable)

    def get_branch_log_name(self, map_variable: dict = None) -> str:
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
        return self.resolve_map_placeholders(self.internal_branch_name, map_variable=map_variable)

    def __str__(self):  # pragma: no cover
        return f'Node of type {self.node_type} and name {self.internal_name}'

    def get_on_failure_node(self) -> Union[str, None]:
        """
        If the node defines a on_failure node in the config, return this or None.

        The naming is relative to the dag, the caller is supposed to resolve it to the correct graph

        Returns:
            str: The on_failure node defined by the dag or ''
        """
        if 'on_failure' in self.config:
            return self.config['on_failure']
        return None

    def get_catalog_settings(self) -> Union[dict, None]:
        """
        If the node defines a catalog settings, return it or None

        Returns:
            dict: catalog settings defined as per the node or None
        """
        if 'catalog' in self.config:
            return self.config['catalog']
        return None

    def get_branch_by_name(self, branch_name: str):
        """
        Retrieve a branch by name.

        The name is expected to follow a dot path convention.

        This method will raise an exception if the node does not have any branches.
        i.e: task, success, fail and as-is would raise an exception

        Args:
            branch_name (str): [description]

        Raises:
            Exception: [description]
        """
        raise Exception(f'Node of type {self.node_type} does not have any branches')

    def is_terminal_node(self):
        """
        Returns whether a node has a next node

        Returns:
            bool: True or False of whether there is next node.
        """
        if 'next' in self.config:
            return False
        return True

    def get_neighbors(self):
        """
        Gets the connecting neighbor nodes, either the "next" node or "on_failure" node.

        Returns:
            list: List of connected neighbors for a given node. Empty if terminal node.
        """
        neighbors = []
        next_node = self.get_next_node()
        if next_node:
            neighbors += [next_node]

        fail_node = self.get_on_failure_node()
        if fail_node:
            neighbors += [fail_node]

        return neighbors

    def get_next_node(self) -> Union[str, None]:
        """
        Return the next node as defined by the config.

        Returns:
            str: The node name, relative to the dag, as defined by the config
        """
        if not self.is_terminal_node():
            return self.config['next']
        return None

    def get_mode_config(self, mode_type) -> dict:
        """
        Return the mode config of the node, if defined, or empty dict

        Args:
            mode_type (str): The mode type that the config refers to.

        Returns:
            dict: The mode config, if defined or an empty dict
        """
        if 'mode_config' in self.config:
            return self.config['mode_config'].get(mode_type, {}) or {}
        return {}

    def get_max_attempts(self) -> int:
        """
        The number of max attempts as defined by the config or 1.

        Returns:
            int: The number of maximum retries as defined by the config or 1.
        """
        if 'retry' in self.config:
            return int(self.config['retry']) or 1
        return 1

    def execute(self, executor, mock=False, map_variable: dict = None, **kwargs):
        """
        The actual function that does the execution of the command in the config.

        Should only be implemented for task, success, fail and as-is and never for
        composite nodes.

        Args:
            executor (magnus.executor.BaseExecutor): The executor mode class
            mock (bool, optional): Don't run, just pretend. Defaults to False.
            map_variable (str, optional): The value of the map iteration variable, if part of a map node.
                Defaults to ''.

        Raises:
            NotImplementedError: Base class, hence not implemented.
        """
        raise NotImplementedError

    def execute_as_graph(self, executor, map_variable: dict = None, **kwargs):
        """
        This function would be called to set up the execution of the individual
        branches of a composite node.

        Function should only be implemented for composite nodes like dag, map, parallel.

        Args:
            executor (magnus.executor.BaseExecutor): The executor mode.

        Raises:
            NotImplementedError: Base class, hence not implemented.
        """
        raise NotImplementedError


def validate_node(node: BaseNode) -> List[str]:
    """
    Given a node defintion, run it against a specification of fields that are
    required and should not be present.

    Args:
        node (BaseNode): The node object created before validation

    Raises:
        Exception: If the node type is not part of the specs

    Returns:
        List[str]: The list of error messages, if found
    """
    specs = utils.load_yaml(resource_filename(__name__, defaults.NODE_SPEC_FILE))
    if node.node_type not in specs:
        raise Exception('Undefined node type, please update specs')

    node_spec = specs[node.node_type]
    messages = []
    if '.' in node.name:
        messages.append('Node names cannot have . in them')

    if '%' in node.name:
        messages.append("Node names cannot have '%' in them")

    if 'required' in node_spec:
        for req in node_spec['required']:
            if not req in node.config:
                messages.append(f'{node.name} should have {req} field')
                continue

    if 'error_on' in node_spec:
        for err in node_spec['error_on']:
            if err in node.config:
                messages.append(f'{node.name} should not have {err} field')
    return messages


class TaskNode(BaseNode):
    """
    A node of type Task.

    This node does the actual function execution of the graph in all cases.
    """
    node_type = 'task'

    def execute(self, executor, mock=False, map_variable: dict = None, **kwargs):
        # Here is where the juice is
        attempt_log = executor.run_log_store.create_attempt_log()
        try:
            attempt_log.start_time = str(datetime.now())
            attempt_log.status = defaults.SUCCESS
            if not mock:
                # Do not run if we are mocking the execution, could be useful for caching and dry runs
                self.execution_type.execute_command(step_config=self.config, map_variable=map_variable)
        except Exception as _e:  # pylint: disable=W0703
            logger.exception('Task failed')
            attempt_log.status = defaults.FAIL
            attempt_log.message = str(_e)
        finally:
            attempt_log.end_time = str(datetime.now())
            attempt_log.duration = utils.get_duration_between_datetime_strings(
                attempt_log.start_time, attempt_log.end_time)
        return attempt_log

    def execute_as_graph(self, executor, map_variable: dict = None, **kwargs):
        """
        Should not be implemented for a single node.

        Args:
            executor ([type]): [description]

        Raises:
            Exception: Not a composite node, always raises an exception
        """
        raise Exception('Node is not a composite node, invalid traversal rule')


class FailNode(BaseNode):
    """
    A leaf node of the graph that represents a failure node
    """
    node_type = 'fail'

    def execute(self, executor, mock=False, map_variable: dict = None, **kwargs):
        attempt_log = executor.run_log_store.create_attempt_log()
        try:
            attempt_log.start_time = str(datetime.now())
            attempt_log.status = defaults.SUCCESS
            #  could be a branch or run log
            run_or_branch_log = executor.run_log_store.get_branch_log(
                self.get_branch_log_name(map_variable), executor.run_id)
            run_or_branch_log.status = defaults.FAIL
            executor.run_log_store.add_branch_log(run_or_branch_log, executor.run_id)
        except BaseException:  # pylint: disable=W0703
            logger.exception('Fail node execution failed')
        finally:
            attempt_log.status = defaults.SUCCESS  # This is a dummy node, so we ignore errors and mark SUCCESS
            attempt_log.end_time = str(datetime.now())
            attempt_log.duration = utils.get_duration_between_datetime_strings(
                attempt_log.start_time, attempt_log.end_time)
        return attempt_log

    def execute_as_graph(self, executor, map_variable: dict = None, **kwargs):
        """
        Should not be implemented for a single node.

        Args:
            executor ([type]): [description]

        Raises:
            Exception: Not a composite node, always raises an exception
        """
        raise Exception('Node is not a composite node, invalid traversal rule')


class SuccessNode(BaseNode):
    """
    A leaf node of the graph that represents a success node
    """
    node_type = 'success'

    def execute(self, executor, mock=False, map_variable: dict = None, **kwargs):
        attempt_log = executor.run_log_store.create_attempt_log()
        try:
            attempt_log.start_time = str(datetime.now())
            attempt_log.status = defaults.SUCCESS
            #  could be a branch or run log
            run_or_branch_log = executor.run_log_store.get_branch_log(
                self.get_branch_log_name(map_variable), executor.run_id)
            run_or_branch_log.status = defaults.SUCCESS
            executor.run_log_store.add_branch_log(run_or_branch_log, executor.run_id)
        except BaseException:  # pylint: disable=W0703
            logger.exception('Success node execution failed')
        finally:
            attempt_log.status = defaults.SUCCESS  # This is a dummy node and we make sure we mark it as success
            attempt_log.end_time = str(datetime.now())
            attempt_log.duration = utils.get_duration_between_datetime_strings(
                attempt_log.start_time, attempt_log.end_time)
        return attempt_log

    def execute_as_graph(self, executor, map_variable: dict = None, **kwargs):
        """
        Should not be implemented for a single node.

        Args:
            executor ([type]): [description]

        Raises:
            Exception: Not a composite node, always raises an exception
        """
        raise Exception('Node is not a composite node, invalid traversal rule')


class ParallelNode(BaseNode):
    """
    A composite node containing many graph objects within itself.

    The structure is generally:
        ParallelNode:
            Branch A:
                Sub graph definition
            Branch B:
                Sub graph definition
            . . .

    We currently support parallel nodes within parallel nodes.
    """
    node_type = 'parallel'

    def __init__(self, name, internal_name, config, execution_type, internal_branch_name=None):
        # pylint: disable=R0914,R0913
        super().__init__(name, internal_name, config, execution_type, internal_branch_name=internal_branch_name)
        self.branches = self.get_sub_graphs()
        self.is_composite = True

    def get_sub_graphs(self):
        """
        For the branches mentioned in the config['branches'], create a graph object.
        The graph object is also given an internal naming convention following a dot path convention

        Returns:
            dict: A branch_name: dag for every branch mentioned in the branches
        """

        branches = {}
        for branch_name, branch_config in self.config['branches'].items():
            sub_graph = create_graph(branch_config, internal_branch_name=self.internal_name + '.' + branch_name)
            branches[self.internal_name + '.' + branch_name] = sub_graph

        if not branches:
            raise Exception('A parallel node should have branches')
        return branches

    def get_branch_by_name(self, branch_name: str):
        """
        Retrieve a branch by name.
        The name is expected to follow a dot path convention.

        Returns a Graph Object

        Args:
            branch_name (str): The name of the branch to retrieve

        Raises:
            Exception: If the branch by that name does not exist
        """
        if branch_name in self.branches:
            return self.branches[branch_name]

        raise Exception(f'No branch by name: {branch_name} is present in {self.name}')

    def execute(self, executor, mock=False, map_variable: dict = None, **kwargs):
        """
        This method should never be called for a node of type Parallel

        Args:
            executor (BaseExecutor): The Executor class as defined by the config
            mock (bool, optional): If the operation is just a mock. Defaults to False.

        Raises:
            NotImplementedError: This method should never be called for a node of type Parallel
        """
        raise Exception('Node is of type composite, error in traversal rules')

    def execute_as_graph(self, executor, map_variable: dict = None, **kwargs):
        """
        This function does the actual execution of the sub-branches of the parallel node.

        From a design perspective, this function should not be called if the execution mode is 3rd party orchestrated.

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
        # Prepare the branch logs
        for internal_branch_name, branch in self.branches.items():
            effective_branch_name = self.resolve_map_placeholders(internal_branch_name, map_variable=map_variable)

            branch_log = executor.run_log_store.create_branch_log(effective_branch_name)
            branch_log.status = defaults.PROCESSING
            executor.run_log_store.add_branch_log(branch_log, executor.run_id)

        jobs = []
        # Given that we can have nesting and complex graphs, controlling the number of processess is hard.
        # A better way is to actually submit the job to some process scheduler which does resource management
        for internal_branch_name, branch in self.branches.items():
            if executor.is_parallel_execution():
                # Trigger parallel jobs
                action = magnus.pipeline.execute_single_brach
                kwargs = {
                    'configuration_file': executor.configuration_file,
                    'pipeline_file': executor.pipeline_file,
                    'variables_file': executor.variables_file,
                    'branch_name': internal_branch_name.replace(' ', defaults.COMMAND_FRIENDLY_CHARACTER),
                    'run_id': executor.run_id,
                    'map_variable': json.dumps(map_variable)
                }
                process = multiprocessing.Process(target=action, kwargs=kwargs)
                jobs.append(process)
                process.start()

            else:
                # If parallel is not enabled, execute them sequentially
                executor.execute_graph(branch, map_variable=map_variable, **kwargs)

        for job in jobs:
            job.join()  # Find status of the branches

        step_success_bool = True
        waiting = False
        for internal_branch_name, branch in self.branches.items():
            effective_branch_name = self.resolve_map_placeholders(internal_branch_name, map_variable=map_variable)
            branch_log = executor.run_log_store.get_branch_log(effective_branch_name, executor.run_id)
            if branch_log.status == defaults.FAIL:
                step_success_bool = False

            if branch_log.status == defaults.PROCESSING:
                waiting = True

        # Collate all the results and update the status of the step
        effective_internal_name = self.resolve_map_placeholders(self.internal_name, map_variable=map_variable)
        step_log = executor.run_log_store.get_step_log(effective_internal_name, executor.run_id)
        step_log.status = defaults.PROCESSING

        if step_success_bool:  #  If none failed and nothing is waiting
            if not waiting:
                step_log.status = defaults.SUCCESS
        else:
            step_log.status = defaults.FAIL

        executor.run_log_store.add_step_log(step_log, executor.run_id)


class MapNode(BaseNode):
    """
    A composite node that contains ONE graph object within itself that has to be executed with an iterable.

    The structure is genrally:
        MapNode:
            branch

        The config is expected to have a variable 'iterate_on' and iterate_as which are looked for in the parameters.
        for iter_variable in parameters['iterate_on']:
            Execute the Branch by sending {'iterate_as': iter_variable}

    The internal naming convention creates branches dynamically based on the iteration value
    """
    node_type = 'map'

    def __init__(self, name, internal_name, config, execution_type, internal_branch_name=None):
        # pylint: disable=R0914,R0913
        super().__init__(name, internal_name, config, execution_type, internal_branch_name=internal_branch_name)
        self.iterate_on = self.config.get('iterate_on', None)
        self.iterate_as = self.config.get('iterate_as', None)
        self.is_composite = True

        if not self.iterate_on:
            raise Exception('A node type of map requires a parameter iterate_on, please define it in the config')
        if not self.iterate_as:
            raise Exception('A node type of map requires a parameter iterate_as, please define it in the config')

        self.branch_placeholder_name = defaults.MAP_PLACEHOLDER
        self.branch = self.get_sub_graph()

    def get_sub_graph(self):
        """
        Create a sub-dag from the config['branch']

        The graph object has an internal branch name, that is equal to the name of the step.
        And the sub-dag nodes follow an dot path naming convention

        Returns:
            Graph: A graph object
        """

        branch_config = self.config['branch']
        branch = create_graph(
            branch_config, internal_branch_name=self.internal_name + '.' + self.branch_placeholder_name)
        return branch

    def get_branch_by_name(self, branch_name: str):
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

    def execute(self, executor, mock=False, map_variable: dict = None, **kwargs):
        """
        This method should never be called for a node of type map

        Args:
            executor (BaseExecutor): The Executor class as defined by the config
            mock (bool, optional): If the operation is just a mock. Defaults to False.

        Raises:
            NotImplementedError: This method should never be called for a node of type Parallel
        """
        raise Exception('Node is of type composite, error in traversal rules')

    def execute_as_graph(self, executor, map_variable: dict = None, **kwargs):
        """
        This function does the actual execution of the branch of the map node.

        From a design perspective, this function should not be called if the execution mode is 3rd party orchestrated.

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
        run_log = executor.run_log_store.get_run_log_by_id(executor.run_id)
        if self.iterate_on not in run_log.parameters:
            raise Exception(
                f'Expected parameter {self.iterate_on} not present in Run Log parameters, was it ever set before?')

        iterate_on = run_log.parameters[self.iterate_on]
        if not isinstance(iterate_on, list):
            raise Exception('Only list is allowed as a valid iterator type')

        # Prepare the branch logs
        for iter_variable in iterate_on:
            effective_branch_name = self.resolve_map_placeholders(
                self.internal_name + '.' + str(iter_variable),
                map_variable=map_variable)
            branch_log = executor.run_log_store.create_branch_log(effective_branch_name)
            branch_log.status = defaults.PROCESSING
            executor.run_log_store.add_branch_log(branch_log, executor.run_id)

        jobs = []
        # Given that we can have nesting and complex graphs, controlling the number of processess is hard.
        # A better way is to actually submit the job to some process scheduler which does resource management
        for iter_variable in iterate_on:
            effective_map_variable = map_variable or OrderedDict()
            effective_map_variable[self.iterate_as] = iter_variable

            if executor.is_parallel_execution():
                # Trigger parallel jobs
                action = magnus.pipeline.execute_single_brach
                kwargs = {
                    'configuration_file': executor.configuration_file,
                    'pipeline_file': executor.pipeline_file,
                    'variables_file': executor.variables_file,
                    'branch_name': self.branch.internal_branch_name.replace(' ', defaults.COMMAND_FRIENDLY_CHARACTER),
                    'run_id': executor.run_id,
                    'map_variable': json.dumps(effective_map_variable)
                }
                process = multiprocessing.Process(target=action, kwargs=kwargs)
                jobs.append(process)
                process.start()

            else:
                # If parallel is not enabled, execute them sequentially
                executor.execute_graph(self.branch, map_variable=effective_map_variable, **kwargs)

        for job in jobs:
            job.join()
        # # Find status of the branches
        step_success_bool = True
        waiting = False
        for iter_variable in iterate_on:
            effective_branch_name = self.resolve_map_placeholders(self.internal_name + '.' + str(iter_variable),
                                                                  map_variable=map_variable)
            branch_log = executor.run_log_store.get_branch_log(
                effective_branch_name, executor.run_id)
            if branch_log.status == defaults.FAIL:
                step_success_bool = False

            if branch_log.status == defaults.PROCESSING:
                waiting = True

        # Collate all the results and update the status of the step
        effective_internal_name = self.resolve_map_placeholders(self.internal_name, map_variable=map_variable)
        step_log = executor.run_log_store.get_step_log(effective_internal_name, executor.run_id)
        step_log.status = defaults.PROCESSING

        if step_success_bool:  #  If none failed and nothing is waiting
            if not waiting:
                step_log.status = defaults.SUCCESS
        else:
            step_log.status = defaults.FAIL

        executor.run_log_store.add_step_log(step_log, executor.run_id)


class DagNode(BaseNode):
    """
    A composite node that internally holds a dag.

    The structure is genrally:
        DagNode:
            dag_definition: A YAML file that holds the dag in 'dag' block

        The config is expected to have a variable 'dag_definition'.
    """
    node_type = 'dag'

    def __init__(self, name, internal_name, config, execution_type, internal_branch_name=None):
        # pylint: disable=R0914,R0913
        super().__init__(name, internal_name, config, execution_type, internal_branch_name=internal_branch_name)
        self.sub_dag_file = self.config.get('dag_definition', None)
        self.is_composite = True

        if not self.sub_dag_file:
            raise Exception('A node type of dag requires a parameter dag_definition, please define it in the config')

        self.branch = self.get_sub_graph()

    @property
    def _internal_branch_name(self):
        """
        THe internal branch name in dot path convention

        Returns:
            [type]: [description]
        """
        return self.internal_name + '.' + defaults.DAG_BRANCH_NAME

    def get_sub_graph(self):
        """
        Create a sub-dag from the config['dag_definition']

        The graph object has an internal branch name, that is equal to the name of the step.
        And the sub-dag nodes follow an dot path naming convention

        Returns:
            Graph: A graph object
        """

        dag_config = utils.load_yaml(self.sub_dag_file)
        if 'dag' not in dag_config:
            raise Exception(f'No DAG found in {self.sub_dag_file}, please provide it in dag block')

        branch = create_graph(dag_config['dag'],
                              internal_branch_name=self._internal_branch_name)
        return branch

    def get_branch_by_name(self, branch_name: str):
        """
        Retrieve a branch by name.
        The name is expected to follow a dot path convention.

        Returns a Graph Object

        Args:
            branch_name (str): The name of the branch to retrieve

        Raises:
            Exception: If the branch_name is not 'dag'
        """
        if branch_name != self._internal_branch_name:
            raise Exception(f'Node of type {self.node_type} only allows a branch of name {defaults.DAG_BRANCH_NAME}')

        return self.branch

    def execute(self, executor, mock=False, map_variable: dict = None, **kwargs):
        """
        This method should never be called for a node of type dag

        Args:
            executor (BaseExecutor): The Executor class as defined by the config
            mock (bool, optional): If the operation is just a mock. Defaults to False.

        Raises:
            NotImplementedError: This method should never be called for a node of type Parallel
        """
        raise Exception('Node is of type composite, error in traversal rules')

    def execute_as_graph(self, executor, map_variable: dict = None, **kwargs):
        """
        This function does the actual execution of the branch of the dag node.

        From a design perspective, this function should not be called if the execution mode is 3rd party orchestrated.

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
        step_success_bool = True
        waiting = False

        effective_branch_name = self.resolve_map_placeholders(self._internal_branch_name, map_variable=map_variable)
        effective_internal_name = self.resolve_map_placeholders(self.internal_name, map_variable=map_variable)

        branch_log = executor.run_log_store.create_branch_log(effective_branch_name)
        branch_log.status = defaults.PROCESSING
        executor.run_log_store.add_branch_log(branch_log, executor.run_id)

        executor.execute_graph(self.branch, map_variable=map_variable, **kwargs)

        branch_log = executor.run_log_store.get_branch_log(effective_branch_name, executor.run_id)
        if branch_log.status == defaults.FAIL:
            step_success_bool = False

        if branch_log.status == defaults.PROCESSING:
            waiting = True

        step_log = executor.run_log_store.get_step_log(effective_internal_name, executor.run_id)
        step_log.status = defaults.PROCESSING

        if step_success_bool:  #  If none failed and nothing is waiting
            if not waiting:
                step_log.status = defaults.SUCCESS
        else:
            step_log.status = defaults.FAIL

        executor.run_log_store.add_step_log(step_log, executor.run_id)


class AsISNode(BaseNode):
    """
    AsIs is a convenience design node.

    It always returns success in the attempt log and does nothing during interactive compute.

    The command given to execute is ignored but it does do the syncing of the catalog.
    This node is very akin to pass state in Step functions.

    This node type could be handy when designing the pipeline and stubbing functions

    But in render mode for job specification of a 3rd party orchestrator, this node comes handy.
    """
    node_type = 'as-is'

    def __init__(self, name, internal_name, config, execution_type, internal_branch_name=None):
        # pylint: disable=R0914,R0913
        super().__init__(name, internal_name, config, execution_type, internal_branch_name=internal_branch_name)

    def execute(self, executor, mock=False, map_variable: dict = None, **kwargs):
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
        attempt_log = executor.run_log_store.create_attempt_log()

        attempt_log.start_time = str(datetime.now())
        attempt_log.status = defaults.SUCCESS  # This is a dummy node and always will be success

        attempt_log.end_time = str(datetime.now())
        attempt_log.duration = utils.get_duration_between_datetime_strings(
            attempt_log.start_time, attempt_log.end_time)
        return attempt_log

    def execute_as_graph(self, executor, map_variable: dict = None, **kwargs):
        """
        Should not be implemented for a single node.

        Args:
            executor ([type]): [description]

        Raises:
            Exception: Not a composite node, always raises an exception
        """
        raise Exception('Node is not a composite node, invalid traversal rule')
