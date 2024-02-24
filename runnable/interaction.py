from __future__ import annotations

import json
import logging
import os
from functools import wraps
from typing import Any, ContextManager, Dict, Optional, TypeVar, Union, cast, overload

from pydantic import BaseModel

import runnable.context as context
from runnable import defaults, exceptions, parameters, pickler, utils
from runnable.datastore import RunLog, StepLog

logger = logging.getLogger(defaults.LOGGER_NAME)

CastT = TypeVar("CastT")


def check_context(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not context.run_context.executor:
            msg = (
                "There are no active executor and services. This should not have happened and is a bug."
                " Please raise a bug report."
            )
            raise Exception(msg)
        result = func(*args, **kwargs)
        return result

    return wrapper


@check_context
def track_this(step: int = 0, **kwargs):
    """
    Tracks key-value pairs to the experiment tracker.

    The value is dumped as a dict, by alias, if it is a pydantic model.

    Args:
        step (int, optional): The step to track the data at. Defaults to 0.
        **kwargs (dict): The key-value pairs to track.

    Examples:
        >>> track_this(step=0, my_int_param=123, my_float_param=123.45, my_str_param='hello world')
        >>> track_this(step=1, my_int_param=456, my_float_param=456.78, my_str_param='goodbye world')
    """
    prefix = defaults.TRACK_PREFIX

    for key, value in kwargs.items():
        logger.info(f"Tracking {key} with value: {value}")

        if isinstance(value, BaseModel):
            value = value.model_dump(by_alias=True)

        os.environ[prefix + key + f"{defaults.STEP_INDICATOR}{step}"] = json.dumps(value)


@check_context
def set_parameter(**kwargs) -> None:
    """
    Store a set of parameters.

    !!! note
        The parameters are not stored in run log at this point in time.
        They are collected now and stored in the run log after completion of the task.

    Parameters:
        **kwargs (dict): A dictionary of key-value pairs to store as parameters.

    Returns:
        None

    Examples:
        >>> set_parameter(my_int_param=123, my_float_param=123.45, my_bool_param=True, my_str_param='hello world')
        >>> get_parameter('my_int_param', int)
        123
        >>> get_parameter('my_float_param', float)
        123.45
        >>> get_parameter('my_bool_param', bool)
        True
        >>> get_parameter('my_str_param', str)
        'hello world'

        >>> # Example of using Pydantic models
        >>> class MyModel(BaseModel):
        ...     field1: str
        ...     field2: int
        >>> set_parameter(my_model_param=MyModel(field1='value1', field2=2))
        >>> get_parameter('my_model_param', MyModel)
        MyModel(field1='value1', field2=2)

    """
    parameters.set_user_defined_params_as_environment_variables(kwargs)


@overload
def get_parameter(key: str, cast_as: Optional[CastT]) -> CastT:
    ...


@overload
def get_parameter(cast_as: Optional[CastT]) -> CastT:
    ...


@check_context
def get_parameter(key: Optional[str] = None, cast_as: Optional[CastT] = None) -> Union[Dict[str, Any], CastT]:
    """
    Get a parameter by its key.
    If the key is not provided, all parameters will be returned.

    cast_as is not required for JSON supported type (int, float, bool, str).
    For complex nested parameters, cast_as could package them into a pydantic model.
    If cast_as is not provided, the type will remain as dict for nested structures.

    Note that the cast_as pydantic model is the class, not an instance.

    Args:
        key (str, optional): The key of the parameter to retrieve. If not provided, all parameters will be returned.
        cast_as (Type, optional): The type to cast the parameter to. If not provided, the type will remain as it is
            for simple data types (int, float, bool, str). For nested parameters, it would be a dict.

    Raises:
        Exception: If the parameter does not exist and key is not provided.
        ValidationError: If the parameter cannot be cast as pydantic model, when cast_as is provided.

    Examples:
        >>> get_parameter('my_int_param', int)
        123
        >>> get_parameter('my_float_param', float)
        123.45
        >>> get_parameter('my_bool_param', bool)
        True
        >>> get_parameter('my_str_param', str)
        'hello world'
        >>> get_parameter('my_model_param', MyModel)
        MyModel(field1='value1', field2=2)
        >>> get_parameter(cast_as=MyModel)
        MyModel(field1='value1', field2=2)

    """
    params = parameters.get_user_set_parameters(remove=False)

    if not key:
        # Return all parameters
        return cast(CastT, parameters.cast_parameters_as_type(params, cast_as))  # type: ignore

    if key not in params:
        raise Exception(f"Parameter {key} is not set before")

    # Return the parameter value, casted as asked.
    return cast(CastT, parameters.cast_parameters_as_type(params[key], cast_as))  # type: ignore


@check_context
def get_secret(secret_name: str) -> str:
    """
    Retrieve a secret from the secret store.

    Args:
        secret_name (str): The name of the secret to retrieve.

    Raises:
        SecretNotFoundError: If the secret does not exist in the store.

    Returns:
        str: The secret value.
    """
    secrets_handler = context.run_context.secrets_handler
    try:
        return secrets_handler.get(name=secret_name)
    except exceptions.SecretNotFoundError:
        logger.exception(f"No secret by the name {secret_name} found in the store")
        raise


@check_context
def get_from_catalog(name: str, destination_folder: str = ""):
    """
    Get data from the catalog.

    The name can be a wildcard pattern following globing rules.

    Args:
        name (str): The name of the data catalog entry.
        destination_folder (str, optional): The destination folder to download the data to.
            If not provided, the default destination folder set in the catalog will be used.
    """
    if not destination_folder:
        destination_folder = context.run_context.catalog_handler.compute_data_folder

    data_catalog = context.run_context.catalog_handler.get(
        name,
        run_id=context.run_context.run_id,
    )

    if context.run_context.executor._context_step_log:
        context.run_context.executor._context_step_log.add_data_catalogs(data_catalog)
    else:
        logger.warning("Step log context was not found during interaction! The step log will miss the record")


@check_context
def put_in_catalog(filepath: str):
    """
    Add a file or folder to the data catalog.
    You can use wild cards following globing rules.

    Args:
        filepath (str): The path to the file or folder added to the catalog
    """

    data_catalog = context.run_context.catalog_handler.put(
        filepath,
        run_id=context.run_context.run_id,
    )
    if not data_catalog:
        logger.warning(f"No catalog was done by the {filepath}")

    if context.run_context.executor._context_step_log:
        context.run_context.executor._context_step_log.add_data_catalogs(data_catalog)
    else:
        logger.warning("Step log context was not found during interaction! The step log will miss the record")


@check_context
def put_object(data: Any, name: str):
    """
    Serialize and store a python object in the data catalog.

    This function behaves the same as `put_in_catalog`
    but with python objects.

    Args:
        data (Any): The python data object to store.
        name (str): The name to store it against.
    """
    native_pickler = pickler.NativePickler()

    native_pickler.dump(data=data, path=name)
    put_in_catalog(f"{name}{native_pickler.extension}")

    # Remove the file
    os.remove(f"{name}{native_pickler.extension}")


@check_context
def get_object(name: str) -> Any:
    """
    Retrieve and deserialize a python object from the data catalog.

    This function behaves the same as `get_from_catalog` but with
    python objects.

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


@check_context
def get_run_id() -> str:
    """
    Returns the run_id of the current run.

    You can also access this from the environment variable `MAGNUS_RUN_ID`.
    """
    return context.run_context.run_id


@check_context
def get_run_log() -> RunLog:
    """
    Returns the run_log of the current run.

    The return is a deep copy of the run log to prevent any modification.
    """
    return context.run_context.run_log_store.get_run_log_by_id(
        context.run_context.run_id,
        full=True,
    ).copy(deep=True)


@check_context
def get_tag() -> str:
    """
    Returns the tag from the environment.

    Returns:
        str: The tag if provided for the run, otherwise None
    """
    return context.run_context.tag


@check_context
def get_experiment_tracker_context() -> ContextManager:
    """
    Return a context session of the experiment tracker.

    You can start to use the context with the python ```with``` statement.
    """
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

    from runnable import entrypoints, graph  # pylint: disable=import-outside-toplevel

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
    set_parameters = parameters.get_user_set_parameters(remove=True)

    step_log = cast(StepLog, context.run_context.executor._context_step_log)
    step_log.user_defined_metrics = tracked_data
    context.run_context.run_log_store.add_step_log(step_log, context.run_context.run_id)

    context.run_context.run_log_store.set_parameters(context.run_context.run_id, set_parameters)

    context.run_context.executor._context_step_log = None
    context.run_context.execution_plan = ""
    context.run_context.executor = None  # type: ignore
