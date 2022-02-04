import os
import json
import subprocess
import importlib
import sys
import logging

from magnus import utils
from magnus import defaults

logger = logging.getLogger(defaults.NAME)


class BaseTaskType:  # pylint: disable=too-few-public-methods
    """
    A base task class which does the execution of command defined by the user
    """
    task_type = ''

    def __init__(self, command=None):
        self.command = command

    def get_parameters(self, map_variable: dict = None, **kwargs) -> dict:
        """
        Return the parameters in scope for the execution

        Args:
            map_variable (dict, optional): If the command is part of map node, the value of map. Defaults to None.

        Returns:
            dict: The parameters dictionary in-scope for the task execution
        """
        return utils.get_user_set_parameters(remove=False)

    def execute_command(self, command: str, map_variable: dict = None):
        """
        The function to execute the command mentioned in command.

        And map_variable is sent in as an argument into the function.

        Args:
            command (str): The actual command to run
            parameters (dict, optional): The parameters available across the system. Defaults to None.
            map_variable (dict, optional): If the command is part of map node, the value of map. Defaults to None.

        Raises:
            NotImplementedError: Base class, not implemented
        """
        raise NotImplementedError()

    def set_parameters(self, parameters: dict = None, **kwargs):
        """
        Set the parameters back to the environment variables.

        Args:
            parameters (dict, optional): The parameters to set back as env variables. Defaults to None.
        """
        if not isinstance(parameters, dict):
            msg = (
                f'call to function {self.command} returns of type: {type(parameters)}. '
                'Only dictionaries are supported as return values for functions as part part of magnus pipeline.')
            raise Exception(msg)
        for key, value in parameters.items():
            logger.info(f'Setting User defined parameter {key} with value: {value}')
            os.environ[defaults.PARAMETER_PREFIX + key] = json.dumps(value)


class PythonExecutionType(BaseTaskType):  # pylint: disable=too-few-public-methods
    """
    The execution class for python command
    """
    task_type = 'python'

    def execute_command(self, map_variable: dict = None):
        module, func = utils.get_module_and_func_names(self.command)
        sys.path.insert(0, os.getcwd())  # Need to add the current directory to path
        imported_module = importlib.import_module(module)
        f = getattr(imported_module, func)

        parameters = self.get_parameters()
        filtered_parameters = utils.filter_arguments_for_func(f, parameters, map_variable)

        logger.info(f'Calling {func} from {module} with {filtered_parameters}')
        try:
            user_set_parameters = f(**filtered_parameters)
        except Exception as _e:
            msg = (
                f'Call to the function {self.command} with {filtered_parameters} did not succeed.\n'
            )
            logger.exception(msg)
            logger.exception(_e)
            raise

        self.set_parameters(user_set_parameters)


class PythonLambdaExecutionType(BaseTaskType):  # pylint: disable=too-few-public-methods
    """
    The execution class for python command
    """
    task_type = 'python-lambda'

    def execute_command(self, map_variable: dict = None):
        if '_' in self.command or '__' in self.command:
            msg = (
                f'Command given to {self.task_type} cannot have _ or __ in them. '
                'The string is supposed to be for simple expressions only.'
            )
            raise Exception(msg)

        f = eval(self.command)

        parameters = self.get_parameters()
        filtered_parameters = utils.filter_arguments_for_func(f, parameters, map_variable)

        logger.info(f'Calling lambda function: {self.command} with {filtered_parameters}')
        try:
            user_set_parameters = f(**filtered_parameters)
        except Exception as _e:
            msg = (
                f'Call to the function {self.command} with {filtered_parameters} did not succeed.\n'
            )
            logger.exception(msg)
            logger.exception(_e)
            raise

        self.set_parameters()


class ShellExecutionType(BaseTaskType):
    """
    The execution class for shell based commands
    """
    task_type = 'shell'

    def execute_command(self, map_variable: dict = None):
        # TODO can we do this without shell=True. Hate that but could not find a way out
        # This is horribly weird, focussing only on python ways of doing for now
        # It might be that we have to write a bash/windows script that does things for us
        # Need to over-ride set parameters too
        subprocess_env = os.environ.copy()
        if map_variable:
            subprocess_env[defaults.PARAMETER_PREFIX + 'MAP_VARIABLE'] = json.dumps(map_variable)
        result = subprocess.run(self.command, check=True, env=subprocess_env, shell=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        print(result.stdout)
        print(result.stderr)
