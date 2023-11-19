import json
import logging
import os
from dataclasses import is_dataclass
from typing import Any, Dict, Optional, Type, Union

from pydantic import BaseModel
from typing_extensions import TypeAlias

from magnus import defaults
from magnus.utils import remove_prefix

logger = logging.getLogger(defaults.LOGGER_NAME)


cast_typeT: TypeAlias = Union[Type[Any], Dict[str, Any], BaseModel, Any]  # Hate it but dataclasses are a bit tricky!


def get_user_set_parameters(remove: bool = False) -> Dict[str, Any]:
    """
    Scans the environment variables for any user returned parameters that have a prefix MAGNUS_PRM_.

    This function does not deal with any type conversion of the parameters.
    It just deserializes the parameters and returns them as a dictionary.

    Args:
        remove (bool, optional): Flag to remove the parameter if needed. Defaults to False.

    Returns:
        dict: The dictionary of found user returned parameters
    """
    parameters = {}
    for env_var, value in os.environ.items():
        if env_var.startswith(defaults.PARAMETER_PREFIX):
            key = remove_prefix(env_var, defaults.PARAMETER_PREFIX)
            try:
                parameters[key.lower()] = json.loads(value)
            except json.decoder.JSONDecodeError:
                logger.error(f"Parameter {key} could not be JSON decoded, adding the literal value")
                parameters[key.lower()] = value

            if remove:
                del os.environ[env_var]
    return parameters


def set_user_defined_params_as_environment_variables(params: Dict[str, Any], update: bool = True):
    """
    Sets the user set parameters as environment variables.

    At this point in time, the params are already in Dict or some kind of literal

    Args:
        parameters (Dict[str, Any]): The parameters to set as environment variables
        update (bool, optional): Flag to update the environment variables. Defaults to True.

    """
    for key, value in params.items():
        logger.info(f"Storing parameter {key} with value: {value}")
        environ_key = defaults.PARAMETER_PREFIX + key

        if environ_key in os.environ and not update:
            continue

        os.environ[environ_key] = serialize_parameter_as_str(value)


def cast_parameters_as_type(value: Any, newT: Optional[Type] = None) -> cast_typeT:
    """
    Casts the environment variable to the given type.

    Note: Only pydantic models and dataclasses are special, everything else just goes through.

    Args:
        value (Any): The value to cast
        newT (T): The type to cast to

    Returns:
        T: The casted value
    """
    if not newT:
        return value

    if issubclass(newT, BaseModel):
        return newT(**value)

    if is_dataclass(newT):
        return newT(**value)

    if issubclass(newT, Dict):
        return dict(value)

    if type(value) != newT:
        logger.warning(f"Casting {value} of {type(value)} to {newT} seems wrong!!")

    return newT(value)


def serialize_parameter_as_str(value: Any) -> str:
    if isinstance(value, BaseModel):
        return json.dumps(value.model_dump())

    if is_dataclass(value):
        return json.dumps(value.as_dict())

    return json.dumps(value)
