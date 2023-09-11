import contextlib
import importlib
import io
import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import ClassVar, List, Tuple

from pydantic import BaseModel, Extra, validator
from stevedore import driver

import magnus.context as context
from magnus import defaults, utils

logger = logging.getLogger(defaults.LOGGER_NAME)
logging.getLogger("stevedore").setLevel(logging.CRITICAL)


# --8<-- [start:docs]


class BaseTaskType(BaseModel):
    """A base task class which does the execution of command defined by the user."""

    task_type: ClassVar[str] = ""

    node_name: str

    class Config:
        extra = Extra.forbid

    @property
    def _context(self):
        return context.run_context

    def get_cli_options(self) -> Tuple[str, dict]:
        """
        Key is the name of the cli option and value is the value of the cli option.
        This should always be in sync with the cli options defined in execute_*.

        Returns:
            str: The name of the cli option.
            dict: The dict of cli options for the task.

        Raises:
            NotImplementedError: Base class, not implemented
        """
        raise NotImplementedError()

    def _get_parameters(self, map_variable: dict = None, **kwargs) -> dict:
        """Return the parameters in scope for the execution.

        Args:
            map_variable (dict, optional): If the command is part of map node, the value of map. Defaults to None.

        Returns:
            dict: The parameters dictionary in-scope for the task execution
        """
        return utils.get_user_set_parameters(remove=False)

    def execute_command(self, map_variable: dict = None, **kwargs):
        """The function to execute the command.

        And map_variable is sent in as an argument into the function.

        Args:
            map_variable (dict, optional): If the command is part of map node, the value of map. Defaults to None.

        Raises:
            NotImplementedError: Base class, not implemented
        """
        raise NotImplementedError()

    def _set_parameters(self, parameters: dict = None, **kwargs):
        """Set the parameters back to the environment variables.

        Args:
            parameters (dict, optional): The parameters to set back as env variables. Defaults to None.
        """
        # Nothing to do
        if not parameters:
            return

        if not isinstance(parameters, dict):
            msg = (
                f"call to function {self.command} returns of type: {type(parameters)}. "
                "Only dictionaries are supported as return values for functions as part part of magnus pipeline."
            )
            logger.warn(msg)
            return

        for key, value in parameters.items():
            logger.info(f"Setting User defined parameter {key} with value: {value}")
            os.environ[defaults.PARAMETER_PREFIX + key] = json.dumps(value)

    @contextlib.contextmanager
    def output_to_file(self, map_variable: dict = None):
        """Context manager to put the output of a function execution to catalog.

        Args:
            map_variable (dict, optional): If the command is part of map node, the value of map. Defaults to None.

        """
        from magnus import put_in_catalog  # Causing cyclic imports

        log_file_name = self.node_name.replace(" ", "_")
        if map_variable:
            for _, value in map_variable.items():
                log_file_name += "_" + str(value)

        log_file = open(log_file_name, "w")

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


# --8<-- [end:docs]


class PythonTaskType(BaseTaskType):  # pylint: disable=too-few-public-methods
    """The task class for python command."""

    task_type: ClassVar[str] = "python"

    command: str

    @validator("command")
    def validate_command(cls, command: str):
        if not command:
            raise Exception("Command cannot be empty for shell task")

        return command

    def get_cli_options(self) -> Tuple[str, dict]:
        """Return the cli options for the task.

        Returns:
            dict: The cli options for the task
        """
        return "function", {"command": self.command}

    def execute_command(self, map_variable: dict = None, **kwargs):
        """Execute the notebook as defined by the command."""
        module, func = utils.get_module_and_func_names(self.command)
        sys.path.insert(0, os.getcwd())  # Need to add the current directory to path
        imported_module = importlib.import_module(module)
        f = getattr(imported_module, func)

        parameters = self._get_parameters()
        filtered_parameters = utils.filter_arguments_for_func(f, parameters, map_variable)

        if map_variable:
            os.environ[defaults.PARAMETER_PREFIX + "MAP_VARIABLE"] = json.dumps(map_variable)

        logger.info(f"Calling {func} from {module} with {filtered_parameters}")

        with self.output_to_file(map_variable=map_variable) as _:
            try:
                user_set_parameters = f(**filtered_parameters)
            except Exception as _e:
                msg = f"Call to the function {self.command} with {filtered_parameters} did not succeed.\n"
                logger.exception(msg)
                logger.exception(_e)
                raise

            if map_variable:
                del os.environ[defaults.PARAMETER_PREFIX + "MAP_VARIABLE"]

            self._set_parameters(user_set_parameters)


class PythonLambdaTaskType(BaseTaskType):  # pylint: disable=too-few-public-methods
    """The task class for python-lambda command."""

    task_type: ClassVar[str] = "python-lambda"

    command: str

    @validator("command")
    def validate_command(cls, command: str):
        if not command:
            raise Exception("Command cannot be empty for shell task")

        return command

    def execute_command(self, map_variable: dict = None, **kwargs):
        """Execute the lambda function as defined by the command.

        Args:
            map_variable (dict, optional): If the node is part of an internal branch. Defaults to None.

        Raises:
            Exception: If the lambda function has _ or __ in it that can cause issues.
        """
        if "_" in self.command or "__" in self.command:
            msg = (
                f"Command given to {self.task_type} cannot have _ or __ in them. "
                "The string is supposed to be for simple expressions only."
            )
            raise Exception(msg)

        f = eval(self.command)

        parameters = self._get_parameters()
        filtered_parameters = utils.filter_arguments_for_func(f, parameters, map_variable)

        if map_variable:
            os.environ[defaults.PARAMETER_PREFIX + "MAP_VARIABLE"] = json.dumps(map_variable)

        logger.info(f"Calling lambda function: {self.command} with {filtered_parameters}")
        try:
            user_set_parameters = f(**filtered_parameters)
        except Exception as _e:
            msg = f"Call to the function {self.command} with {filtered_parameters} did not succeed.\n"
            logger.exception(msg)
            logger.exception(_e)
            raise

        if map_variable:
            del os.environ[defaults.PARAMETER_PREFIX + "MAP_VARIABLE"]

        self._set_parameters(user_set_parameters)


class NotebookTaskType(BaseTaskType):
    """The task class for Notebook based execution."""

    task_type: ClassVar[str] = "notebook"

    command: str
    notebook_output_path: str = ""
    optional_ploomber_args: dict = {}

    @validator("command")
    def notebook_should_end_with_ipynb(cls, command: str):
        if not command:
            raise Exception("Command should point to the ipynb file")

        if not command.endswith(".ipynb"):
            raise Exception("Notebook task should point to a ipynb file")

        return command

    @validator("notebook_output_path")
    def correct_notebook_output_path(cls, notebook_output_path: str, values: dict):
        if notebook_output_path:
            return notebook_output_path

        return "".join(values["command"].split(".")[:-1]) + "_out.ipynb"

    def get_cli_options(self) -> Tuple[str, dict]:
        return "notebook", {"command": self.command, "notebook-output-path": self.notebook_output_path}

    def execute_command(self, map_variable: dict = None, **kwargs):
        """Execute the python notebook as defined by the command.

        Args:
            map_variable (dict, optional): If the node is part of internal branch. Defaults to None.

        Raises:
            ImportError: If necessary dependencies are not installed
            Exception: If anything else fails
        """
        try:
            import ploomber_engine as pm

            from magnus import put_in_catalog  # Causes issues with cyclic import

            parameters = self._get_parameters()
            filtered_parameters = parameters

            notebook_output_path = self.notebook_output_path

            if map_variable:
                os.environ[defaults.PARAMETER_PREFIX + "MAP_VARIABLE"] = json.dumps(map_variable)

                for _, value in map_variable.items():
                    notebook_output_path += "_" + str(value)

            ploomber_optional_args = self.optional_ploomber_args

            kwds = {
                "input_path": self.command,
                "output_path": notebook_output_path,
                "parameters": filtered_parameters,
                "log_output": True,
                "progress_bar": False,
            }

            kwds.update(ploomber_optional_args)

            with self.output_to_file(map_variable=map_variable) as _:
                pm.execute_notebook(**kwds)

            put_in_catalog(notebook_output_path)
            if map_variable:
                del os.environ[defaults.PARAMETER_PREFIX + "MAP_VARIABLE"]

        except ImportError as e:
            msg = (
                "Task type of notebook requires ploomber engine to be installed. Please install via optional: notebook"
            )
            raise Exception(msg) from e


class ShellTaskType(BaseTaskType):
    """
    The task class for shell based commands.
    TODO: There is a way to read in parameters or tracking information from stdout and regex
    """

    task_type: ClassVar[str] = "shell"

    command: str

    @validator("command")
    def validate_command(cls, command: str):
        if not command:
            raise Exception("Command cannot be empty for shell task")

        return command

    def execute_command(self, map_variable: dict = None, **kwargs):
        # Using shell=True as we want to have chained commands to be executed in the same shell.
        """Execute the shell command as defined by the command.

        Args:
            map_variable (dict, optional): If the node is part of an internal branch. Defaults to None.
        """
        subprocess_env = os.environ.copy()

        if map_variable:
            subprocess_env[defaults.PARAMETER_PREFIX + "MAP_VARIABLE"] = json.dumps(map_variable)

        with subprocess.Popen(
            self.command,
            shell=True,
            env=subprocess_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        ) as proc, self.output_to_file(map_variable=map_variable) as _:
            for line in proc.stdout:  # type: ignore
                logger.info(line)
                print(line)

            proc.wait()
            if proc.returncode != 0:
                raise Exception("Command failed")


class ContainerTaskType(BaseTaskType):
    """
    The task class for container based execution.
    """

    task_type: ClassVar[str] = "container"

    image: str
    context_path: str = defaults.DEFAULT_CONTAINER_CONTEXT_PATH
    command: str = ""  # Would be defaulted to the entrypoint of the container
    data_folder: str = defaults.DEFAULT_CONTAINER_DATA_PATH  # Would be relative to the context_path
    output_parameters_file: str = defaults.DEFAULT_CONTAINER_OUTPUT_PARAMETERS  # would be relative to the context_path
    secrets: List[str] = []
    experiment_tracking_file: str = ""

    _temp_dir: str = ""

    class Config:
        underscore_attrs_are_private = True

    def get_cli_options(self) -> Tuple[str, dict]:
        return "container", {
            "image": self.image,
            "context-path": self.context_path,
            "command": self.command,
            "data-folder": self.data_folder,
            "output-parameters_file": self.output_parameters_file,
            "secrets": self.secrets,
            "experiment-tracking-file": self.experiment_tracking_file,
        }

    def execute_command(self, map_variable: dict = None, **kwargs):
        # Conditional import
        from magnus import track_this
        from magnus.context import run_context

        try:
            import docker  # pylint: disable=C0415

            client = docker.from_env()
            api_client = docker.APIClient()
        except ImportError as e:
            msg = "Task type of container requires docker to be installed. Please install via optional: docker"
            logger.exception(msg)
            raise Exception(msg) from e
        except Exception as ex:
            logger.exception("Could not get access to docker")
            raise Exception("Could not get the docker socket file, do you have docker installed?") from ex

        container_env_variables = {}

        for key, value in self._get_parameters().items():
            container_env_variables[defaults.PARAMETER_PREFIX + key] = value

        if map_variable:
            container_env_variables[defaults.PARAMETER_PREFIX + "MAP_VARIABLE"] = json.dumps(map_variable)

        for secret_name in self.secrets:
            secret_value = run_context.secrets_handler.get(secret_name)
            container_env_variables[secret_name] = secret_value

        mount_volumes = self.get_mount_volumes()

        executor_config = run_context.executor._resolve_executor_config(run_context.executor._context_node)
        optional_docker_args = executor_config.get("optional_docker_args", {})

        try:
            container = client.containers.create(
                self.image,
                command=shlex.split(self.command),
                auto_remove=False,
                network_mode="host",
                environment=container_env_variables,
                volumes=mount_volumes,
                **optional_docker_args,
            )

            container.start()
            stream = api_client.logs(container=container.id, timestamps=True, stream=True, follow=True)
            while True:
                try:
                    output = next(stream).decode("utf-8")
                    output = output.strip("\r\n")
                    logger.info(output)
                except StopIteration:
                    logger.info("Docker Run completed")
                    break

            exit_status = api_client.inspect_container(container.id)["State"]["ExitCode"]
            container.remove(force=True)

            if exit_status != 0:
                msg = (
                    f"Docker command failed with exit code {exit_status}."
                    "Hint: When chaining multiple commands, use sh -c"
                )
                raise Exception(msg)

            container_return_parameters = {}
            experiment_tracking_variables = {}
            if self._temp_dir:
                parameters_file = Path(self._temp_dir) / self.output_parameters_file
                if parameters_file.is_file():
                    with open(parameters_file, "r") as f:
                        container_return_parameters = json.load(f)

                experiment_tracking_file = Path(self._temp_dir) / self.experiment_tracking_file
                if experiment_tracking_file.is_file():
                    with open(experiment_tracking_file, "r") as f:
                        experiment_tracking_variables = json.load(f)

                self._set_parameters(container_return_parameters)
                track_this(**experiment_tracking_variables)

        except Exception as _e:
            logger.exception("Problems with spinning up the container")
            raise _e
        finally:
            if self._temp_dir:
                shutil.rmtree(self._temp_dir)

    def get_mount_volumes(self) -> dict:
        """
        Get the required mount volumes from the configuration.
        We need to mount both the catalog and the parameter.json files.

        Returns:
            dict: The mount volumes in the format that docker expects.
        """
        from magnus.context import run_context

        compute_data_folder = run_context.executor.get_effective_compute_data_folder()
        mount_volumes = {}

        # Create temporary directory for parameters.json and map it to context_path
        self._temp_dir = tempfile.mkdtemp()
        mount_volumes[str(Path(self._temp_dir).resolve())] = {
            "bind": f"{str(Path(self.context_path).resolve())}/",
            "mode": "rw",
        }
        logger.info(f"Mounting {str(Path(self._temp_dir).resolve())} to {str(Path(self.context_path).resolve())}/")

        # Map the data folder to context_path/data_folder
        if compute_data_folder:
            path_to_data = Path(self.context_path) / self.data_folder
            mount_volumes[str(Path(compute_data_folder).resolve())] = {
                "bind": f"{str(path_to_data)}/",
                "mode": "rw",
            }
            logger.info(f"Mounting {compute_data_folder} to {str(path_to_data)}/")

        return mount_volumes


def create_task(kwargs_for_init) -> BaseTaskType:
    """
    Creates a task object from the command configuration.

    Args:
        A dictionary of keyword arguments that are sent by the user to the task.
        Check against the model class for the validity of it.

    Returns:
        tasks.BaseTaskType: The command object
    """
    command_type = kwargs_for_init.pop("command_type", defaults.COMMAND_TYPE)

    try:
        task_mgr = driver.DriverManager(
            namespace="tasks",
            name=command_type,
            invoke_on_load=True,
            invoke_kwds=kwargs_for_init,
        )
        return task_mgr.driver
    except Exception as _e:
        msg = (
            f"Could not find the task type {command_type}. Please ensure you have installed "
            "the extension that provides the node type."
        )
        raise Exception(msg) from _e
