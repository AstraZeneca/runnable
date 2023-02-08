from __future__ import annotations

import functools
import json
import logging
import os
from logging.config import fileConfig
from pathlib import Path
from types import FunctionType
from typing import Dict, List, Optional, Union

from pkg_resources import resource_filename

from magnus import defaults, exceptions, graph, nodes, pipeline, utils

logger = logging.getLogger(defaults.NAME)


def track_this(step: int = 0, **kwargs):
    """
    Set up the keyword args as environment variables for tracking purposes as
    part of the run.

    For every key-value pair found in kwargs, we set up an environmental variable of
    MAGNUS_TRACK_key_{step} = json.dumps(value)

    If step=0, we ignore step for magnus purposes.

    Args:
        kwargs (dict): The dictionary of key value pairs to track.
    """
    from magnus.pipeline import \
        global_executor  # pylint: disable=import-outside-toplevel

    prefix = defaults.TRACK_PREFIX

    if step:
        prefix += f'{str(step)}_'

    for key, value in kwargs.items():
        logger.info(f'Tracking {key} with value: {value}')
        os.environ[prefix + key] = json.dumps(value)
        global_executor.experiment_tracker.set_metric(key, value, step=step)  # type: ignore


def store_parameter(update: bool = True, **kwargs: dict):
    """
    Set up the keyword args as environment variables for parameters tracking
    purposes as part pf the run.

    If update_existing is True, we override the current value if the parameter already exists.

    For every key-value pair found in kwargs, we set up an environmental variable of
    MAGNUS_PRM_key = json.dumps(value)
    """
    for key, value in kwargs.items():
        logger.info(f'Storing parameter {key} with value: {value}')
        environ_key = defaults.PARAMETER_PREFIX + key

        if environ_key in os.environ and not update:
            return

        os.environ[environ_key] = json.dumps(value)


def get_parameter(key=None) -> Union[str, dict]:
    """
    Get the parameter set as part of the user convenience function.

    We do not remove the parameter from the environment in this phase as
    as the function execution has not been completed.

    Returns all the parameters, if no key was provided.

    Args:
        key (str, optional): The parameter key to retrieve. Defaults to None.

    Raises:
        Exception: If the mentioned key was not part of the paramters

    Returns:
        Union[str, dict]: A single value of str or a dictionary if no key was specified
    """
    parameters = utils.get_user_set_parameters(remove=False)
    if not key:
        return parameters
    if key not in parameters:
        raise Exception(f'Parameter {key} is not set before')
    return parameters[key]


def get_secret(secret_name: str = None) -> str:
    """
    Get a secret by the name from the secrets manager

    Args:
        secret_name (str): The name of the secret to get. Defaults to None.

    Returns:
        str: The secret from the secrets manager, if exists. If the requested secret was None, we return all.
        Otherwise, raises exception.

    Raises:
        exceptions.SecretNotFoundError: Secret not found in the secrets manager.
    """
    from magnus.pipeline import \
        global_executor  # pylint: disable=import-outside-toplevel
    secrets_handler = global_executor.secrets_handler  # type: ignore

    try:
        return secrets_handler.get(name=secret_name)
    except exceptions.SecretNotFoundError:
        logger.exception(f'No secret by the name {secret_name} found in the store')
        raise


def get_from_catalog(name: str, destination_folder: str = None):
    """
    A convenience interaction function to get file from the catalog and place it in the destination folder

    Note: We do not perform any kind of serialization/deserialization in this way.
    Args:
        name (str): The name of the file to get from the catalog
        destination_folder (None): The place to put the file. defaults to compute data folder

    """
    from magnus.pipeline import \
        global_executor  # pylint: disable=import-outside-toplevel

    if not destination_folder:
        destination_folder = global_executor.catalog_handler.compute_data_folder  # type: ignore

    data_catalog = global_executor.catalog_handler.get(name, run_id=global_executor.run_id,  # type: ignore
                                                       compute_data_folder=destination_folder)

    if global_executor.context_step_log:  # type: ignore
        global_executor.context_step_log.add_data_catalogs(data_catalog)  # type: ignore
    else:
        logger.warning("Step log context was not found during interaction! The step log will miss the record")


def put_in_catalog(filepath: str):
    """
    A convenience interaction function to put the file in the catalog.

    Note: We do not perform any kind of serialization/deserialization in this way.

    Args:
        filepath (str): The path of the file to put in the catalog
    """
    from magnus.pipeline import \
        global_executor  # pylint: disable=import-outside-toplevel

    file_path = Path(filepath)

    data_catalog = global_executor.catalog_handler.put(file_path.name, run_id=global_executor.run_id,  # type: ignore
                                                       compute_data_folder=file_path.parent)
    if global_executor.context_step_log:  # type: ignore
        global_executor.context_step_log.add_data_catalogs(data_catalog)  # type: ignore
    else:
        logger.warning("Step log context was not found during interaction! The step log will miss the record")


def get_run_id() -> str:
    """
    Returns the run_id of the current run
    """
    return os.environ.get(defaults.ENV_RUN_ID, '')


def get_tag() -> str:
    """
    Returns the tag from the environment.

    Returns:
        str: The tag if provided for the run, otherwise None
    """
    return os.environ.get(defaults.MAGNUS_RUN_TAG, '')


def get_experiment_tracker_context():
    from magnus.pipeline import \
        global_executor  # pylint: disable=import-outside-toplevel

    experiment_tracker = global_executor.experiment_tracker
    return experiment_tracker.client_context


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
                and pipeline.global_executor.execution_plan != defaults.EXECUTION_PLAN.decorator:
            self.active = False
            return

        self.executor = pipeline.prepare_configurations(
            configuration_file=magnus_config, parameters_file=parameters_file)

        self.executor.execution_plan = defaults.EXECUTION_PLAN.decorator
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

        mode_executor.execution_plan = defaults.EXECUTION_PLAN.pipeline
        utils.set_magnus_environment_variables(run_id=run_id, configuration_file=configuration_file, tag=tag)

        mode_executor.dag = self.dag
        # Prepare for graph execution
        mode_executor.prepare_for_graph_execution()

        logger.info('Executing the graph')
        mode_executor.execute_graph(dag=mode_executor.dag)

        mode_executor.send_return_code()
