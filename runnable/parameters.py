import argparse
import inspect
import json
import logging
import os
from typing import Any, Dict, Type

from pydantic import BaseModel, ConfigDict
from typing_extensions import Callable

from runnable import defaults
from runnable.datastore import JsonParameter, ObjectParameter
from runnable.defaults import MapVariableType
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
                parameters[key.lower()] = JsonParameter(
                    kind="json", value=json.loads(value)
                )
            except json.decoder.JSONDecodeError:
                logger.warning(
                    f"Parameter {key} could not be JSON decoded, adding the literal value"
                )
                parameters[key.lower()] = JsonParameter(kind="json", value=value)

            if remove:
                del os.environ[env_var]
    return parameters


def return_json_parameters(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns the parameters as a JSON serializable dictionary.

    Args:
        params (dict): The parameters to serialize.

    Returns:
        dict: The JSON serializable dictionary.
    """
    return_params = {}
    for key, value in params.items():
        if isinstance(value, ObjectParameter):
            continue

        return_params[key] = value.get_value()
    return return_params


def filter_arguments_for_func(
    func: Callable[..., Any],
    params: Dict[str, Any],
    map_variable: MapVariableType = None,
) -> Dict[str, Any]:
    """
    Inspects the function to be called as part of the pipeline to find the arguments of the function.
    Matches the function arguments to the parameters available either by static parameters or by up stream steps.

    The function "func" signature could be:
    - def my_function(arg1: int, arg2: str, arg3: float):
    - def my_function(arg1: int, arg2: str, arg3: float, **kwargs):
        in this case, we would need to send in remaining keyword arguments as a dictionary.
    - def my_function(arg1: int, arg2: str, arg3: float, args: argparse.Namespace):
        In this case, we need to send the rest of the parameters as attributes of the args object.

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

    bound_args = {}
    missing_required_args: list[str] = []
    var_keyword_param = None
    namespace_param = None

    # First pass: Handle regular parameters and identify special parameters
    for name, value in function_args.items():
        # Ignore any *args
        if value.kind == inspect.Parameter.VAR_POSITIONAL:
            logger.warning(f"Ignoring parameter {name} as it is VAR_POSITIONAL")
            continue

        # Check for **kwargs parameter
        if value.kind == inspect.Parameter.VAR_KEYWORD:
            var_keyword_param = name
            continue

        # Check for argparse.Namespace parameter
        if value.annotation == argparse.Namespace:
            namespace_param = name
            continue

        # Handle regular parameters
        if name not in params:
            if value.default != inspect.Parameter.empty:
                # Default value is given in the function signature, we can use it
                bound_args[name] = value.default
            else:
                # This is a required parameter that's missing
                missing_required_args.append(name)
        else:
            # We have a parameter of this name, lets bind it
            param_value = params[name]

            if (
                inspect.isclass(value.annotation)
                and issubclass(value.annotation, BaseModel)
            ) and not isinstance(param_value, ObjectParameter):
                # Even if the annotation is a pydantic model, it can be passed as an object parameter
                # We try to cast it as a pydantic model if asked
                named_param = params[name].get_value()

                if not isinstance(named_param, dict):
                    # A case where the parameter is a one attribute model
                    named_param = {name: named_param}

                bound_model = bind_args_for_pydantic_model(
                    named_param, value.annotation
                )
                bound_args[name] = bound_model

            elif value.annotation in [str, int, float, bool] and callable(
                value.annotation
            ):
                # Cast it if its a primitive type. Ensure the type matches the annotation.
                try:
                    bound_args[name] = value.annotation(params[name].get_value())
                except (ValueError, TypeError) as e:
                    raise ValueError(
                        f"Cannot cast parameter '{name}' to {value.annotation.__name__}: {e}"
                    )
            else:
                # We do not know type of parameter, we send the value as found
                bound_args[name] = params[name].get_value()

    # Find extra parameters (parameters in params but not consumed by regular function parameters)
    consumed_param_names = set(bound_args.keys()) | set(missing_required_args)
    extra_params = {k: v for k, v in params.items() if k not in consumed_param_names}

    # Second pass: Handle **kwargs and argparse.Namespace parameters
    if var_keyword_param is not None:
        # Function accepts **kwargs - add all extra parameters directly to bound_args
        for param_name, param_value in extra_params.items():
            bound_args[param_name] = param_value.get_value()
    elif namespace_param is not None:
        # Function accepts argparse.Namespace - create namespace with extra parameters
        args_namespace = argparse.Namespace()
        for param_name, param_value in extra_params.items():
            setattr(args_namespace, param_name, param_value.get_value())
        bound_args[namespace_param] = args_namespace
    elif extra_params:
        # Function doesn't accept **kwargs or namespace, but we have extra parameters
        # This should only be an error if we also have missing required parameters
        # or if the function truly can't handle the extra parameters
        if missing_required_args:
            # We have both missing required and extra parameters - this is an error
            raise ValueError(
                f"Function {func.__name__} has parameters {missing_required_args} that are not present in the parameters"
            )
        # If we only have extra parameters and no missing required ones, we just ignore the extras
        # This allows for more flexible parameter passing

    # Check for missing required parameters
    if missing_required_args:
        if var_keyword_param is None and namespace_param is None:
            # No way to handle missing parameters
            raise ValueError(
                f"Function {func.__name__} has parameters {missing_required_args} that are not present in the parameters"
            )
        # If we have **kwargs or namespace, missing parameters might be handled there
        # But if they're truly required (no default), we should still error
        raise ValueError(
            f"Function {func.__name__} has parameters {missing_required_args} that are not present in the parameters"
        )

    return bound_args


def bind_args_for_pydantic_model(
    params: Dict[str, Any], model: Type[BaseModel]
) -> BaseModel:
    class EasyModel(model):  # type: ignore
        model_config = ConfigDict(extra="ignore")

    swallow_all = EasyModel(**params)
    bound_model = model(**swallow_all.model_dump())
    return bound_model
