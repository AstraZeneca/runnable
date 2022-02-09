import importlib
import json
import logging
import os
import subprocess
import sys

from magnus import defaults, utils

logger = logging.getLogger(defaults.NAME)

try:
    import papermill as pm
except ImportError:
    pm = None


class BaseTaskType:  # pylint: disable=too-few-public-methods
    """
    A base task class which does the execution of command defined by the user
    """
    task_type = ''

    def __init__(self, command: str, config: dict = None):
        self.command = command
        self.config = config or {}

    def get_parameters(self, map_variable: dict = None, **kwargs) -> dict:
        """
        Return the parameters in scope for the execution

        Args:
            map_variable (dict, optional): If the command is part of map node, the value of map. Defaults to None.

        Returns:
            dict: The parameters dictionary in-scope for the task execution
        """
        return utils.get_user_set_parameters(remove=False)

    def execute_command(self, map_variable: dict = None, **kwargs):
        """
        The function to execute the command.

        And map_variable is sent in as an argument into the function.

        Args:
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
        # Nothing to do
        if not parameters:
            return

        if not isinstance(parameters, dict):
            msg = (
                f'call to function {self.command} returns of type: {type(parameters)}. '
                'Only dictionaries are supported as return values for functions as part part of magnus pipeline.')
            raise Exception(msg)

        for key, value in parameters.items():
            logger.info(f'Setting User defined parameter {key} with value: {value}')
            os.environ[defaults.PARAMETER_PREFIX + key] = json.dumps(value)


class PythonTaskType(BaseTaskType):  # pylint: disable=too-few-public-methods
    """
    The task class for python command
    """
    task_type = 'python'

    def execute_command(self, map_variable: dict = None, **kwargs):
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


class PythonLambdaTaskType(BaseTaskType):  # pylint: disable=too-few-public-methods
    """
    The task class for python-lambda command
    """
    task_type = 'python-lambda'

    def execute_command(self, map_variable: dict = None, **kwargs):
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

        self.set_parameters(user_set_parameters)


class NotebookTaskType(BaseTaskType):
    """
    The task class for Notebook based execution
    """
    task_type = 'notebook'

    def __init__(self, command: str, config: dict = None):
        if not command.endswith('.ipynb'):
            raise Exception('Notebook task should point to a ipynb file')

        super().__init__(command, config)

    def execute_command(self, map_variable: dict = None, **kwargs):
        try:
            if not pm:
                raise ImportError('Papermill is required for notebook type node')

            parameters = self.get_parameters()

            notebook_parameters = pm.inspect_notebook(self.command)
            filtered_parameters = utils.filter_arguments_from_parameters(parameters=parameters,
                                                                         signature_parameters=notebook_parameters,
                                                                         map_variable=map_variable)
            notebook_output_path = self.config.get('notebook_output_path',
                                                   ''.join(self.command.split('.')[:-1]) + '_out.ipynb')
            kernel = self.config.get('notebook_kernel', None)
            papermill_optional_args = self.config.get('optional_papermill_args', {})

            kwds = {
                'input_path': self.command,
                'output_path': notebook_output_path,
                'parameters': filtered_parameters,
            }

            kwds.update(papermill_optional_args)

            if kernel:
                kwds['kernel_name'] = kernel

            pm.execute_notebook(**kwds)
        except ImportError as e:
            msg = (
                f'Task type of notebook requires papermill to be installed. Please install via optional: notebook'
            )
            raise Exception(msg) from e


class ShellTaskType(BaseTaskType):
    """
    The task class for shell based commands
    """
    task_type = 'shell'

    def execute_command(self, map_variable: dict = None, **kwargs):
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
