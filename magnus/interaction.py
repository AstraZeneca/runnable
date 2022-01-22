import json
import logging
import os
from pathlib import Path
from typing import Union

from magnus import defaults, exceptions, utils

logger = logging.getLogger(defaults.NAME)


def track_this(**kwargs):
    """
    Set up the keyword args as environment variables for tracking purposes as
    part pf the run.

    For every key-value pair found in kwargs, we set up an environmental variable of
    MAGNUS_TRACK_key = json.dumps(value)

    Args:
        kwargs (dict): The dictionary of key value pairs to track.
    """
    for key, value in kwargs.items():
        logger.info(f'Tracking {key} with value: {value}')
        os.environ[defaults.TRACK_PREFIX + key] = json.dumps(value)


def store_parameter(**kwargs: dict):
    """
    Set up the keyword args as environment variables for parameters tracking
    purposes as part pf the run.

    For every key-value pair found in kwargs, we set up an environmental variable of
    MAGNUS_PRM_key = json.dumps(value)
    """
    for key, value in kwargs.items():
        logger.info(f'Storing parameter {key} with value: {value}')
        os.environ[defaults.PARAMETER_PREFIX + key] = json.dumps(value)


def get_parameter(key=None) -> Union[str, dict]:
    """
    Get the parameter set as part of the user convenience function.

    We do not remove the parameter from the environment in this phase as
    as the function execution has not been completed.

    Returns all the parameters, if no key was provided.

    Args:
        key (str, optional): The parameter key to retrieve. Defaults to None.

    Raises:
        Exception: If the menionted key was not part of the paramters

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
