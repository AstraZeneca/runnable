import functools
import json
import logging
from logging.config import fileConfig
from types import FunctionType
from typing import Dict, List, Optional, Union

from pkg_resources import resource_filename

from magnus import defaults, exceptions, graph, nodes, pipeline, utils
from magnus.interaction import get_tag

logger = logging.getLogger(defaults.NAME)


class step(object):

    def __init__(
            self, name: Union[str, FunctionType],
            catalog_config: dict = None, magnus_config: str = None,
            parameters_file: str = None):
        """
        This decorator could be used to make the function within the scope of magnus.

        Since we are not orchestrating, it is expected that resource management happens outside this scope.

        Args:
            name (str, callable): The name of the step. The step log would have the same name
            catalog_config (dict): The configuration of the catalog per step.
            magnus_config (str): The name of the file having the magnus config, defaults to None.
        """
        if isinstance(name, FunctionType):
            name = name()

        self.name = name
        self.catalog_config = catalog_config
        self.active = True  # Check if we are executing the function via pipeline

        if pipeline.global_executor \
                and pipeline.global_executor.execution_plan == defaults.EXECUTION_PLAN.CHAINED.value:
            self.active = False
            return

        self.executor = pipeline.prepare_configurations(
            configuration_file=magnus_config, parameters_file=parameters_file)

        self.executor.execution_plan = defaults.EXECUTION_PLAN.UNCHAINED.value
        run_id = self.executor.step_decorator_run_id
        if not run_id:
            msg = (
                f'Step decorator expects run id from environment.'
            )
            raise Exception(msg)

        self.executor.run_id = run_id
        utils.set_magnus_environment_variables(run_id=run_id, configuration_file=magnus_config, tag=get_tag())

        try:
            # Try to get it if previous steps have created it
            # TODO: Can call the set_up_runlog now.
            run_log = self.executor.run_log_store.get_run_log_by_id(self.executor.run_id)
            if run_log.status in [defaults.FAIL, defaults.SUCCESS]:  # TODO: Remove this in preference to defaults
                """
                This check is mostly useless as we do not know when the graph ends as they are created dynamically.
                This only prevents from using a run_id which has reached a final state.
                #TODO: There is a need to create a status called step_success
                """
                msg = (
                    f'The run_log for run_id: {run_id} already exists and is in {run_log.status} state.'
                    ' Make sure that this was not run before.'
                )
                raise Exception(msg)
        except exceptions.RunLogNotFoundError:
            # Create one if they are not created
            self.executor._set_up_run_log()

    def __call__(self, func):
        """
        The function is converted into a node and called via the magnus framework.
        """
        @functools.wraps(func)
        def wrapped_f(*args, **kwargs):
            if not self.active:
                # If we are not running via decorator, execute the function
                return func(*args, **kwargs)

            step_config = {
                'command': func,
                'command_type': 'python-function',
                'type': 'task',
                'next': 'not defined',
                'catalog': self.catalog_config
            }
            node = graph.create_node(name=self.name, step_config=step_config)
            self.executor.execute_from_graph(node=node)
            run_log = self.executor.run_log_store.get_run_log_by_id(run_id=self.executor.run_id, full=False)
            # TODO: If the previous step succeeded, make the status of the run log step_success
            print(json.dumps(run_log.dict(), indent=4))
        return wrapped_f


class Task:
    def __init__(self, name: str, command: Union[str, FunctionType], command_type: str = defaults.COMMAND_TYPE,
                 command_config: Optional[dict] = None, catalog: Optional[dict] = None,
                 mode_config: Optional[dict] = None, retry: int = 1, on_failure: str = '',
                 next_node: str = ''):
        self.name = name
        self.command = command
        self.command_type = command_type
        self.command_config = command_config or {}
        self.catalog = catalog or {}
        self.mode_config = mode_config or {}
        self.retry = retry
        self.on_failure = on_failure
        self.next_node = next_node or "success"
        self.node: Optional[nodes.BaseNode] = None

    def _construct_node(self):
        # TODO: The below has issues if the function and the pipeline are in the same module
        # Something to do with __main__ being present
        if isinstance(self.command, FunctionType):
            self.command = utils.get_module_and_func_from_function(self.command)

        node_config = {
            'type': 'task',
            'next_node': self.next_node,
            'command': self.command,
            'command_type': self.command_type,
            'command_config': self.command_config,
            'catalog': self.catalog,
            'mode_config': self.mode_config,
            'retry': self.retry,
            'on_failure': self.on_failure
        }
        # The node will temporarily have invalid branch names
        self.node = graph.create_node(name=self.name, step_config=node_config, internal_branch_name='')

    def _fix_internal_name(self):
        pass


class AsIs:
    def __init__(self, name: str, mode_config: Optional[dict] = None, retry: int = 1, on_failure: str = '',
                 next_node: str = '', **kwargs):
        self.name = name
        self.mode_config = mode_config or {}
        self.retry = retry
        self.on_failure = on_failure
        self.next_node = next_node or "success"
        self.additional_kwargs = kwargs or {}
        self.node: Optional[nodes.BaseNode] = None

    def _construct_node(self):
        node_config = {
            'type': 'as-is',
            'next_node': self.next_node,
            'mode_config': self.mode_config,
            'retry': self.retry,
            'on_failure': self.on_failure
        }
        node_config.update(self.additional_kwargs)
        # The node will temporarily have invalid branch names
        self.node = graph.create_node(name=self.name, step_config=node_config, internal_branch_name='')

    def _fix_internal_name(self):
        pass


class Pipeline:
    # A way for the user to define a pipeline
    # TODO: Allow for nodes other than Task, AsIs
    def __init__(
            self, start_at: Union[Task, AsIs],
            name: str = '', description: str = '', max_time: int = defaults.MAX_TIME, internal_branch_name: str = ''):
        self.start_at = start_at
        self.name = name
        self.description = description
        self.max_time = max_time
        self.internal_branch_name = internal_branch_name
        self.dag: Optional[graph.Graph] = None

    def construct(self, steps: List[Task]):
        graph_config: Dict[str, Union[str, int]] = {
            'description': self.description,
            'name': self.name,
            'max_time': self.max_time,
            'internal_branch_name': self.internal_branch_name
        }
        messages: List[str] = []
        for step in steps:
            step._construct_node()
            messages.extend(step.node.validate())  # type: ignore

        if not steps:
            raise Exception('A dag needs at least one step')

        if messages:
            raise Exception(', '.join(messages))

        graph_config['start_at'] = self.start_at.node.name  # type: ignore

        dag = graph.Graph(**graph_config)  # type: ignore
        dag.nodes = [step.node for step in steps]  # type: ignore

        dag.add_terminal_nodes()

        dag.validate()
        self.dag = dag

    def execute(self, configuration_file: str = '', run_id: str = '', tag: str = '', parameters_file: str = '',
                log_level: str = defaults.LOG_LEVEL):

        fileConfig(resource_filename(__name__, 'log_config.ini'))
        logger = logging.getLogger(defaults.NAME)
        logger.setLevel(log_level)

        run_id = utils.generate_run_id(run_id=run_id)
        mode_executor = pipeline.prepare_configurations(
            configuration_file=configuration_file,
            run_id=run_id,
            tag=tag,
            parameters_file=parameters_file)

        mode_executor.execution_plan = defaults.EXECUTION_PLAN.CHAINED.value
        utils.set_magnus_environment_variables(run_id=run_id, configuration_file=configuration_file, tag=tag)

        mode_executor.dag = self.dag
        # Prepare for graph execution
        mode_executor.prepare_for_graph_execution()

        logger.info('Executing the graph')
        mode_executor.execute_graph(dag=mode_executor.dag)

        mode_executor.send_return_code()
