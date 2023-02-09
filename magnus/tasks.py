import contextlib
import importlib
import io
import json
import logging
import os
import subprocess
import sys

from pydantic import BaseModel

from magnus import defaults, put_in_catalog, utils

logger = logging.getLogger(defaults.NAME)

try:
    import ploomber_engine as pm
except ImportError:
    pm = None


@contextlib.contextmanager
def output_to_file(path: str):
    """
    Context manager to put the output of a function execution to catalog

    Args:
        path (str): Mostly the command you are executing.

    """
    log_file = open(f"{path}.log", 'w')
    f = io.StringIO()
    try:
        with contextlib.redirect_stdout(f):
            yield
    finally:
        print(f.getvalue())  # print to console
        log_file.write(f.getvalue())  # Print to file

        f.close()
        log_file.close()
        put_in_catalog(log_file.name)
        os.remove(log_file.name)


class BaseTaskType:  # pylint: disable=too-few-public-methods
    """
    A base task class which does the execution of command defined by the user
    """
    task_type = ''

    class Config(BaseModel):
        command: str

    def __init__(self, config: dict = None):
        config = config or {}
        self.config = self.Config(**config)

    @property
    def command(self):
        return self.config.command

    def _to_dict(self) -> dict:
        """
        Return a dictionary representation of the task
        """
        task = {}
        task['command'] = self.command
        task['config'] = self.config

        return task

    def _get_parameters(self, map_variable: dict = None, **kwargs) -> dict:
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

    def _set_parameters(self, parameters: dict = None, **kwargs):
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
            logger.warn(msg)
            return

        for key, value in parameters.items():
            logger.info(f'Setting User defined parameter {key} with value: {value}')
            os.environ[defaults.PARAMETER_PREFIX + key] = json.dumps(value)


class PythonFunctionType(BaseTaskType):
    task_type = 'python-function'

    def execute_command(self, map_variable: dict = None, **kwargs):
        parameters = self._get_parameters()
        filtered_parameters = utils.filter_arguments_for_func(self.command, parameters, map_variable)

        if map_variable:
            os.environ[defaults.PARAMETER_PREFIX + 'MAP_VARIABLE'] = json.dumps(map_variable)

        logger.info(f'Calling {self.command} with {filtered_parameters}')

        with output_to_file(self.command) as _:
            try:
                user_set_parameters = self.command(**filtered_parameters)
            except Exception as _e:
                msg = (
                    f'Call to the function {self.command} with {filtered_parameters} did not succeed.\n'
                )
                logger.exception(msg)
                logger.exception(_e)
                raise

            if map_variable:
                del os.environ[defaults.PARAMETER_PREFIX + 'MAP_VARIABLE']

            self._set_parameters(user_set_parameters)


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

        parameters = self._get_parameters()
        filtered_parameters = utils.filter_arguments_for_func(f, parameters, map_variable)

        if map_variable:
            os.environ[defaults.PARAMETER_PREFIX + 'MAP_VARIABLE'] = json.dumps(map_variable)

        logger.info(f'Calling {func} from {module} with {filtered_parameters}')
        with output_to_file(self.command) as _:
            try:
                user_set_parameters = f(**filtered_parameters)
            except Exception as _e:
                msg = (
                    f'Call to the function {self.command} with {filtered_parameters} did not succeed.\n'
                )
                logger.exception(msg)
                logger.exception(_e)
                raise

            if map_variable:
                del os.environ[defaults.PARAMETER_PREFIX + 'MAP_VARIABLE']

            self._set_parameters(user_set_parameters)


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

        parameters = self._get_parameters()
        filtered_parameters = utils.filter_arguments_for_func(f, parameters, map_variable)

        if map_variable:
            os.environ[defaults.PARAMETER_PREFIX + 'MAP_VARIABLE'] = json.dumps(map_variable)

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

        if map_variable:
            del os.environ[defaults.PARAMETER_PREFIX + 'MAP_VARIABLE']

        self._set_parameters(user_set_parameters)


class NotebookTaskType(BaseTaskType):
    """
    The task class for Notebook based execution
    """
    task_type = 'notebook'

    class Config(BaseTaskType.Config):
        notebook_output_path: str = ''
        optional_ploomber_args: dict = {}

    @property
    def notebook_output_path(self):
        if self.config.notebook_output_path:
            return self.config.notebook_output_path

        return ''.join(self.command.split('.')[:-1]) + '_out.ipynb'

    def __init__(self, config: dict = None):
        super().__init__(config)

        if not self.config.command.endswith('.ipynb'):
            raise Exception('Notebook task should point to a ipynb file')

    def execute_command(self, map_variable: dict = None, **kwargs):
        try:
            if not pm:
                raise ImportError('Ploomber engine is required for notebook type node')

            parameters = self._get_parameters()
            filtered_parameters = parameters

            if map_variable:
                os.environ[defaults.PARAMETER_PREFIX + 'MAP_VARIABLE'] = json.dumps(map_variable)

            notebook_output_path = self.notebook_output_path

            ploomber_optional_args = self.config.optional_ploomber_args  # type: ignore

            kwds = {
                'input_path': self.command,
                'output_path': notebook_output_path,
                'parameters': filtered_parameters,
                'log_output': True,
                'progress_bar': False
            }

            kwds.update(ploomber_optional_args)

            pm.execute_notebook(**kwds)

            put_in_catalog(notebook_output_path)
            if map_variable:
                del os.environ[defaults.PARAMETER_PREFIX + 'MAP_VARIABLE']

        except ImportError as e:
            msg = (
                f'Task type of notebook requires ploomber engine to be installed. Please install via optional: notebook'
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
