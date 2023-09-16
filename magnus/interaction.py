from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Union, cast

import magnus.context as context
from magnus import defaults, exceptions, pickler, utils
from magnus.datastore import StepLog

logger = logging.getLogger(defaults.LOGGER_NAME)


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

    if not context.run_context.executor:
        msg = (
            "There are no active executor and services. This should not have happened and is a bug."
            " Please raise a bug report."
        )
        raise Exception(msg)

    prefix = defaults.TRACK_PREFIX

    if step:
        prefix += f"{str(step)}_"

    for key, value in kwargs.items():
        logger.info(f"Tracking {key} with value: {value}")
        os.environ[prefix + key] = json.dumps(value)
        context.run_context.experiment_tracker.log_metric(key, value, step=step)


def store_parameter(update: bool = True, **kwargs: dict):
    """
    Set up the keyword args as environment variables for parameters tracking
    purposes as part pf the run.

    If update_existing is True, we override the current value if the parameter already exists.

    For every key-value pair found in kwargs, we set up an environmental variable of
    MAGNUS_PRM_key = json.dumps(value)
    """
    for key, value in kwargs.items():
        logger.info(f"Storing parameter {key} with value: {value}")
        environ_key = defaults.PARAMETER_PREFIX + key

        if environ_key in os.environ and not update:
            continue

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
        raise Exception(f"Parameter {key} is not set before")
    return parameters[key]


def get_secret(secret_name: str = "") -> str | Dict[str, str]:
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

    if not context.run_context.executor:
        msg = (
            "There are no active executor and services. This should not have happened and is a bug."
            " Please raise a bug report."
        )
        raise Exception(msg)

    secrets_handler = context.run_context.secrets_handler
    try:
        return secrets_handler.get(name=secret_name)
    except exceptions.SecretNotFoundError:
        logger.exception(f"No secret by the name {secret_name} found in the store")
        raise


def get_from_catalog(name: str, destination_folder: str = ""):
    """
    A convenience interaction function to get file from the catalog and place it in the destination folder

    Note: We do not perform any kind of serialization/deserialization in this way.
    Args:
        name (str): The name of the file to get from the catalog
        destination_folder (None): The place to put the file. defaults to compute data folder

    """

    if not context.run_context.executor:
        msg = (
            "There are no active executor and services. This should not have happened and is a bug."
            " Please raise a bug report."
        )
        raise Exception(msg)

    if not destination_folder:
        destination_folder = context.run_context.catalog_handler.compute_data_folder

    data_catalog = context.run_context.catalog_handler.get(
        name,
        run_id=context.run_context.run_id,
        compute_data_folder=destination_folder,
    )

    if not data_catalog:
        logger.warning(f"No catalog was obtained by the {name}")

    if context.run_context.executor._context_step_log:
        context.run_context.executor._context_step_log.add_data_catalogs(data_catalog)
    else:
        logger.warning("Step log context was not found during interaction! The step log will miss the record")


def put_in_catalog(filepath: str):
    """
    A convenience interaction function to put the file in the catalog.

    Note: We do not perform any kind of serialization/deserialization in this way.

    Args:
        filepath (str): The path of the file to put in the catalog
    """

    if not context.run_context.executor:
        msg = (
            "There are no active executor and services. This should not have happened and is a bug."
            " Please raise a bug report."
        )
        raise Exception(msg)

    file_path = Path(filepath)

    data_catalog = context.run_context.catalog_handler.put(
        file_path.name,
        run_id=context.run_context.run_id,
        compute_data_folder=file_path.parent,
    )
    if not data_catalog:
        logger.warning(f"No catalog was done by the {filepath}")

    if context.run_context.executor._context_step_log:
        context.run_context.executor._context_step_log.add_data_catalogs(data_catalog)
    else:
        logger.warning("Step log context was not found during interaction! The step log will miss the record")


def put_object(data: Any, name: str):
    """
    A convenient interaction function to serialize and store the object in catalog.

    Args:
        data (Any): The data object to add to catalog
        name (str): The name to give to the object
    """
    native_pickler = pickler.NativePickler()

    native_pickler.dump(data=data, path=name)
    put_in_catalog(f"{name}{native_pickler.extension}")

    # Remove the file
    os.remove(f"{name}{native_pickler.extension}")


def get_object(name: str) -> Any:
    """
    A convenient interaction function to deserialize and retrieve the object from the catalog.

    Args:
        name (str): The name of the object to retrieve

    Returns:
        Any : The object
    """
    native_pickler = pickler.NativePickler()

    get_from_catalog(name=f"{name}{native_pickler.extension}", destination_folder=".")

    try:
        data = native_pickler.load(name)

        # Remove the file
        os.remove(f"{name}{native_pickler.extension}")
        return data
    except FileNotFoundError as e:
        msg = f"No object by the name {name} has been put in the catalog before."
        logger.exception(msg)
        raise e


def get_run_id() -> str:
    """
    Returns the run_id of the current run
    """
    if not context.run_context.executor:
        msg = (
            "There are no active executor and services. This should not have happened and is a bug."
            " Please raise a bug report."
        )
        raise Exception(msg)

    return context.run_context.run_id


def get_tag() -> str:
    """
    Returns the tag from the environment.

    Returns:
        str: The tag if provided for the run, otherwise None
    """
    if not context.run_context.executor:
        msg = (
            "There are no active executor and services. This should not have happened and is a bug."
            " Please raise a bug report."
        )
        raise Exception(msg)

    return context.run_context.tag


def get_experiment_tracker_context():
    """
    Return a context session of the experiment tracker.

    You can start to use the context with the python with statement.

    eg:
    with get_experiment_tracker_context() as ctx:
        pass

    Returns:
        _type_: _description_
    """
    if not context.run_context.executor:
        msg = (
            "There are no active executor and services. This should not have happened and is a bug."
            " Please raise a bug report."
        )
        raise Exception(msg)

    experiment_tracker = context.run_context.experiment_tracker
    return experiment_tracker.client_context


def start_interactive_session(run_id: str = "", config_file: str = "", tag: str = "", parameters_file: str = ""):
    """
    During interactive python coding, either via notebooks or ipython, you can start a magnus session by calling
    this function. The executor would always be local executor as its interactive.

    If this was called during a pipeline/function/notebook execution, it will be ignored.

    Args:
        run_id (str, optional): The run id to use. Defaults to "" and would be created if not provided.
        config_file (str, optional): The configuration file to use. Defaults to "" and magnus defaults.
        tag (str, optional): The tag to attach to the run. Defaults to "".
        parameters_file (str, optional): The parameters file to use. Defaults to "".
    """

    from magnus import entrypoints, graph  # pylint: disable=import-outside-toplevel

    if context.run_context.executor:
        logger.warn("This is not an interactive session or a session has already been activated.")
        return

    run_id = utils.generate_run_id(run_id=run_id)
    context.run_context = entrypoints.prepare_configurations(
        configuration_file=config_file,
        run_id=run_id,
        tag=tag,
        parameters_file=parameters_file,
        force_local_executor=True,
    )

    executor = context.run_context.executor

    utils.set_magnus_environment_variables(run_id=run_id, configuration_file=config_file, tag=tag)

    context.run_context.execution_plan = defaults.EXECUTION_PLAN.INTERACTIVE.value
    executor.prepare_for_graph_execution()
    step_config = {
        "command": "interactive",
        "command_type": "python",
        "type": "task",
        "next": "success",
    }

    node = graph.create_node(name="interactive", step_config=step_config)
    step_log = context.run_context.run_log_store.create_step_log("interactive", node._get_step_log_name())
    executor.add_code_identities(node=node, step_log=step_log)

    step_log.step_type = node.node_type
    step_log.status = defaults.PROCESSING
    executor._context_step_log = step_log


def end_interactive_session():
    """
    Ends an interactive session.

    Does nothing if the executor is not interactive.
    """

    if not context.run_context.executor:
        logger.warn("There is no active session in play, doing nothing!")
        return

    if context.run_context.execution_plan != defaults.EXECUTION_PLAN.INTERACTIVE.value:
        logger.warn("There is not an interactive session, doing nothing!")
        return

    tracked_data = utils.get_tracked_data()
    parameters = utils.get_user_set_parameters(remove=True)

    step_log = cast(StepLog, context.run_context.executor._context_step_log)
    step_log.user_defined_metrics = tracked_data
    context.run_context.run_log_store.add_step_log(step_log, context.run_context.run_id)

    context.run_context.run_log_store.set_parameters(context.run_context.run_id, parameters)

    context.run_context.executor._context_step_log = None
    context.run_context.execution_plan = ""
    context.run_context.executor = None  # type: ignore
