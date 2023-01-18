import functools
import json
import logging
import os
from pathlib import Path
from typing import Callable, Union

from magnus import defaults, exceptions, graph, pipeline, utils

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
        global_executor.experiment_tracker.set_metric(key, value, step=step)


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

    global_executor.catalog_handler.get(name, run_id=global_executor.run_id,  # type: ignore
                                        compute_data_folder=destination_folder)


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

    global_executor.catalog_handler.put(file_path.name, run_id=global_executor.run_id,  # type: ignore
                                        compute_data_folder=file_path.parent)


def get_run_id() -> str:
    """
    Returns the run_id of the current run
    """
    return os.environ.get(defaults.ENV_RUN_ID, None)


def get_tag() -> str:
    return os.environ.get(defaults.MAGNUS_RUN_TAG, None)


def magnus_session():
    configuration_file = os.environ.get(defaults.MAGNUS_CONFIG_FILE, None)
    run_id = get_run_id()
    tag = get_tag()
    executor = pipeline.prepare_configurations(configuration_file=configuration_file, run_id=run_id, tag=tag)
    executor.execution_plan = defaults.EXECUTION_PLAN.notebook


class step(object):

    def __init__(
            self, name: Union[str, callable],
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
        if isinstance(name, Callable):
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
                    f'The run_log for run_id: {self.run_id} already exists and is in {run_log.status} state.'
                    ' Make sure that this was not run before.'
                )
                raise Exception(msg)
        except exceptions.RunLogNotFoundError:
            # Create one if they are not created
            self.executor.set_up_run_log()

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


class Node:
    # A way for the user to define a node.
    pass


class Pipeline:
    # A way for the user to define a pipeline
    pass
