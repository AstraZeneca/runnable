import inspect
import json
import logging
import os
from typing import Any, Dict, Optional, Type, Union

from pydantic import BaseModel, ConfigDict
from typing_extensions import Callable

from magnus import defaults
from magnus.defaults import TypeMapVariable
from magnus.utils import remove_prefix

logger = logging.getLogger(defaults.LOGGER_NAME)


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


def cast_parameters_as_type(value: Any, newT: Optional[Type] = None) -> Union[BaseModel, Dict[str, Any]]:
    """
    Casts the environment variable to the given type.

    Note: Only pydantic models special, everything else just goes through.

    Args:
        value (Any): The value to cast
        newT (T): The type to cast to

    Returns:
        T: The casted value

    Examples:
        >>> class MyBaseModel(BaseModel):
        ...     a: int
        ...     b: str
        >>>
        >>> class MyDict(dict):
        ...     pass
        >>>
        >>> cast_parameters_as_type({"a": 1, "b": "2"}, MyBaseModel)
        MyBaseModel(a=1, b="2")
        >>> cast_parameters_as_type({"a": 1, "b": "2"}, MyDict)
        MyDict({'a': 1, 'b': '2'})
        >>> cast_parameters_as_type(MyBaseModel(a=1, b="2"), MyBaseModel)
        MyBaseModel(a=1, b="2")
        >>> cast_parameters_as_type(MyDict({"a": 1, "b": "2"}), MyBaseModel)
        MyBaseModel(a=1, b="2")
        >>> cast_parameters_as_type({"a": 1, "b": "2"}, MyDict[str, int])
        MyDict({'a': 1, 'b': '2'})
        >>> cast_parameters_as_type({"a": 1, "b": "2"}, Dict[str, int])
        MyDict({'a': 1, 'b': '2'})
        >>> with pytest.warns(UserWarning):
        ...     cast_parameters_as_type(1, MyBaseModel)
        MyBaseModel(a=1, b=None)
        >>> with pytest.raises(TypeError):
        ...     cast_parameters_as_type(1, MyDict)
    """
    if not newT:
        return value

    if issubclass(newT, BaseModel):
        return newT(**value)

    if issubclass(newT, Dict):
        return dict(value)

    if type(value) != newT:
        logger.warning(f"Casting {value} of {type(value)} to {newT} seems wrong!!")

    return newT(value)


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
    params.update(map_variable or {})

    unassigned_params = set(params.keys())
    bound_args = {}
    for name, value in function_args.items():
        if issubclass(value.annotation, BaseModel):
            bound_model = bind_args_for_pydantic_model(params, value.annotation)
            bound_args[name] = bound_model
            unassigned_params = unassigned_params.difference(bound_model.model_fields.keys())
        else:
            # No annotation is need, no casting required, we trust what we have stored before.
            if name not in params:
                if value.default == inspect.Parameter.empty:
                    raise ValueError(f"Parameter {name} is required for {func.__name__} but not provided")
                bound_args[name] = value.default
                continue

            bound_args[name] = params[name]
            unassigned_params.remove(name)

        params = {key: params[key] for key in unassigned_params}  # remove keys from params if they are assigned

    return bound_args


def bind_args_for_pydantic_model(params: Dict[str, Any], model: Type[BaseModel]) -> BaseModel:
    class EasyModel(model):  # type: ignore
        model_config = ConfigDict(extra="ignore")

    swallow_all = EasyModel(**params)
    bound_model = model(**swallow_all.model_dump())
    return bound_model
