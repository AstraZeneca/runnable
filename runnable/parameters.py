import inspect
import json
import logging
import os
from typing import Any, Dict, Type

from pydantic import BaseModel, ConfigDict
from typing_extensions import Callable

from runnable import defaults
from runnable.datastore import JsonParameter, ObjectParameter
from runnable.defaults import TypeMapVariable
from runnable.utils import remove_prefix

logger = logging.getLogger(defaults.LOGGER_NAME)


def get_user_set_parameters(remove: bool = False) -> Dict[str, JsonParameter]:
    """
    Scans the environment variables for any user returned parameters that have a prefix runnable_PRM_.

    This function does not deal with any type conversion of the parameters.
    It just deserializes the parameters and returns them as a dictionary.

    Args:
        remove (bool, optional): Flag to remove the parameter if needed. Defaults to False.

    Returns:
        dict: The dictionary of found user returned parameters
    """
    parameters: Dict[str, JsonParameter] = {}
    for env_var, value in os.environ.items():
        if env_var.startswith(defaults.PARAMETER_PREFIX):
            key = remove_prefix(env_var, defaults.PARAMETER_PREFIX)
            try:
                parameters[key.lower()] = JsonParameter(kind="json", value=json.loads(value))
            except json.decoder.JSONDecodeError:
                logger.error(f"Parameter {key} could not be JSON decoded, adding the literal value")
                parameters[key.lower()] = JsonParameter(kind="json", value=value)

            if remove:
                del os.environ[env_var]
    return parameters


def serialize_parameter_as_str(value: Any) -> str:
    if isinstance(value, BaseModel):
        return json.dumps(value.model_dump())

    return json.dumps(value)


def filter_arguments_for_func(
    func: Callable[..., Any], params: Dict[str, Any], map_variable: TypeMapVariable = None
) -> Dict[str, Any]:
    """
    Inspects the function to be called as part of the pipeline to find the arguments of the function.
    Matches the function arguments to the parameters available either by command line or by up stream steps.


    Args:
        func (Callable): The function to inspect
        parameters (dict): The parameters available for the run

    Returns:
        dict: The parameters matching the function signature
    """
    function_args = inspect.signature(func).parameters

    # Update parameters with the map variables
    for key, v in (map_variable or {}).items():
        params[key] = JsonParameter(kind="json", value=v)

    unassigned_params = set(params.keys())
    bound_args = {}
    for name, value in function_args.items():
        if name not in params:
            # No parameter of this name was provided
            if value.default == inspect.Parameter.empty:
                # No default value is given in the function signature. error as parameter is required.
                raise ValueError(f"Parameter {name} is required for {func.__name__} but not provided")
            # default value is given in the function signature, nothing further to do.
            continue

        if issubclass(value.annotation, BaseModel):
            # We try to cast it as a pydantic model.
            named_param = params[name].get_value()

            if not isinstance(named_param, dict):
                # A case where the parameter is a one attribute model
                named_param = {name: named_param}

            bound_model = bind_args_for_pydantic_model(named_param, value.annotation)
            bound_args[name] = bound_model
            unassigned_params = unassigned_params.difference(bound_model.model_fields.keys())
        elif isinstance(params[name], ObjectParameter):
            # It is an object, retrieve it
            bound_args[name] = params[name].get_value()
        else:
            # simple python data type.
            bound_args[name] = params[name].get_value()

        unassigned_params.remove(name)

        params = {key: params[key] for key in unassigned_params}  # remove keys from params if they are assigned

    return bound_args


def bind_args_for_pydantic_model(params: Dict[str, Any], model: Type[BaseModel]) -> BaseModel:
    class EasyModel(model):  # type: ignore
        model_config = ConfigDict(extra="ignore")

    swallow_all = EasyModel(**params)
    bound_model = model(**swallow_all.model_dump())
    return bound_model
