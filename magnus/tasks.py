import ast
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
from typing import Any, Dict, List, Tuple

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator
from pydantic._internal._model_construction import ModelMetaclass
from stevedore import driver

import magnus.context as context
from magnus import defaults, parameters, utils
from magnus.defaults import TypeMapVariable

logger = logging.getLogger(defaults.LOGGER_NAME)
logging.getLogger("stevedore").setLevel(logging.CRITICAL)


# TODO: Can we add memory peak, cpu usage, etc. to the metrics?

# --8<-- [start:docs]


class BaseTaskType(BaseModel):
    """A base task class which does the execution of command defined by the user."""

    task_type: str = Field(serialization_alias="command_type")
    node_name: str = Field(exclude=True)

    model_config = ConfigDict(extra="forbid")

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

    def _get_parameters(self, map_variable: TypeMapVariable = None, **kwargs) -> Dict[str, Any]:
        """
        By this step, all the parameters are present as environment variables as json strings.
        Return the parameters in scope for the execution.

        Args:
            map_variable (dict, optional): If the command is part of map node, the value of map. Defaults to None.

        Returns:
            dict: The parameters dictionary in-scope for the task execution
        """
        return parameters.get_user_set_parameters(remove=False)

    def execute_command(self, map_variable: TypeMapVariable = None, **kwargs):
        """The function to execute the command.

        And map_variable is sent in as an argument into the function.

        Args:
            map_variable (dict, optional): If the command is part of map node, the value of map. Defaults to None.

        Raises:
            NotImplementedError: Base class, not implemented
        """
        raise NotImplementedError()

    def _set_parameters(self, params: BaseModel, **kwargs):
        """Set the parameters back to the environment variables.

        Args:
            parameters (dict, optional): The parameters to set back as env variables. Defaults to None.
        """
        # Nothing to do
        if not params:
            return

        if not isinstance(params, BaseModel) or isinstance(params, ModelMetaclass):
            raise ValueError("Output variable of a function can only be a pydantic model or dynamic model.")

        parameters.set_user_defined_params_as_environment_variables(params.model_dump(by_alias=True))

    @contextlib.contextmanager
    def output_to_file(self, map_variable: TypeMapVariable = None):
        """Context manager to put the output of a function execution to catalog.

        Args:
            map_variable (dict, optional): If the command is part of map node, the value of map. Defaults to None.

        """
        from magnus import put_in_catalog  # Causing cyclic imports

        log_file_name = self.node_name.replace(" ", "_") + ".execution.log"
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


class EasyModel(BaseModel):
    model_config = ConfigDict(extra="allow")


def make_pydantic_model(
    variables: Dict[str, Any],
    prefix: str = "",
) -> BaseModel:
    prefix_removed = {utils.remove_prefix(k, prefix): v for k, v in variables.items()}
    return EasyModel(**prefix_removed)


class PythonTaskType(BaseTaskType):  # pylint: disable=too-few-public-methods
    """The task class for python command."""

    task_type: str = Field(default="python", serialization_alias="command_type")
    command: str

    @field_validator("command")
    @classmethod
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

    def execute_command(self, map_variable: TypeMapVariable = None, **kwargs):
        """Execute the notebook as defined by the command."""
        module, func = utils.get_module_and_attr_names(self.command)
        sys.path.insert(0, os.getcwd())  # Need to add the current directory to path
        imported_module = importlib.import_module(module)
        f = getattr(imported_module, func)

        params = self._get_parameters()
        filtered_parameters = parameters.filter_arguments_for_func(f, params, map_variable)

        if map_variable:
            os.environ[defaults.MAP_VARIABLE] = json.dumps(map_variable)

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
                del os.environ[defaults.MAP_VARIABLE]

            self._set_parameters(user_set_parameters)


class PythonLambdaTaskType(BaseTaskType):  # pylint: disable=too-few-public-methods
    """The task class for python-lambda command."""

    task_type: str = Field(default="python-lambda", serialization_alias="command_type")
    command: str

    @field_validator("command")
    @classmethod
    def validate_command(cls, command: str):
        if not command:
            raise Exception("Command cannot be empty for shell task")

        return command

    def execute_command(self, map_variable: TypeMapVariable = None, **kwargs):
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

        params = self._get_parameters()
        filtered_parameters = parameters.filter_arguments_for_func(f, params, map_variable)

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

    task_type: str = Field(default="notebook", serialization_alias="command_type")
    command: str
    notebook_output_path: str = Field(default="", validate_default=True)
    output_cell_tag: str = Field(default="magnus_output", validate_default=True)
    optional_ploomber_args: dict = {}

    _output_tag: str = "magnus_output"

    @field_validator("command")
    @classmethod
    def notebook_should_end_with_ipynb(cls, command: str):
        if not command:
            raise Exception("Command should point to the ipynb file")

        if not command.endswith(".ipynb"):
            raise Exception("Notebook task should point to a ipynb file")

        return command

    @field_validator("notebook_output_path")
    @classmethod
    def correct_notebook_output_path(cls, notebook_output_path: str, info: ValidationInfo):
        if notebook_output_path:
            return notebook_output_path

        command = info.data["command"]
        return "".join(command.split(".")[:-1]) + "_out.ipynb"

    def get_cli_options(self) -> Tuple[str, dict]:
        return "notebook", {"command": self.command, "notebook-output-path": self.notebook_output_path}

    def _parse_notebook_for_output(self, notebook: Any):
        collected_params = {}

        for cell in notebook.cells:
            d = cell.dict()
            # identify the tags attached to the cell.
            tags = d.get("metadata", {}).get("tags", {})
            if self.output_cell_tag in tags:
                # There is a tag that has output
                outputs = d["outputs"]

                for out in outputs:
                    params = out.get("text", "{}")
                    collected_params.update(ast.literal_eval(params))

        return collected_params

    def execute_command(self, map_variable: TypeMapVariable = None, **kwargs):
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
                os.environ[defaults.MAP_VARIABLE] = json.dumps(map_variable)

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

            collected_params: Dict[str, Any] = {}
            with self.output_to_file(map_variable=map_variable) as _:
                out = pm.execute_notebook(**kwds)
                collected_params = self._parse_notebook_for_output(out)

            collected_params_model = make_pydantic_model(collected_params)
            self._set_parameters(collected_params_model)

            put_in_catalog(notebook_output_path)
            if map_variable:
                del os.environ[defaults.MAP_VARIABLE]

        except ImportError as e:
            msg = (
                "Task type of notebook requires ploomber engine to be installed. Please install via optional: notebook"
            )
            raise Exception(msg) from e


class ShellTaskType(BaseTaskType):
    """
    The task class for shell based commands.
    """

    task_type: str = Field(default="shell", serialization_alias="command_type")
    command: str

    @field_validator("command")
    @classmethod
    def validate_command(cls, command: str):
        if not command:
            raise Exception("Command cannot be empty for shell task")

        return command

    def execute_command(self, map_variable: TypeMapVariable = None, **kwargs):
        # Using shell=True as we want to have chained commands to be executed in the same shell.
        """Execute the shell command as defined by the command.

        Args:
            map_variable (dict, optional): If the node is part of an internal branch. Defaults to None.
        """
        subprocess_env = os.environ.copy()

        if map_variable:
            subprocess_env[defaults.MAP_VARIABLE] = json.dumps(map_variable)

        command = self.command.strip() + " && env | grep MAGNUS"
        logger.info(f"Executing shell command: {command}")

        output_parameters = {}

        with subprocess.Popen(
            command,
            shell=True,
            env=subprocess_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ) as proc, self.output_to_file(map_variable=map_variable) as _:
            for line in proc.stdout:  # type: ignore
                logger.info(line)
                print(line)

                if line.startswith(defaults.PARAMETER_PREFIX):
                    key, value = line.strip().split("=", 1)
                    try:
                        output_parameters[key] = json.loads(value)
                    except json.JSONDecodeError:
                        output_parameters[key] = value  # simple data types

                if line.startswith(defaults.TRACK_PREFIX):
                    key, value = line.split("=", 1)
                    os.environ[key] = value.strip()

            proc.wait()
            if proc.returncode != 0:
                raise Exception("Command failed")

        self._set_parameters(
            params=make_pydantic_model(
                output_parameters,
                defaults.PARAMETER_PREFIX,
            )
        )


class ContainerTaskType(BaseTaskType):
    """
    The task class for container based execution.
    """

    task_type: str = Field(default="container", serialization_alias="command_type")
    image: str
    context_path: str = defaults.DEFAULT_CONTAINER_CONTEXT_PATH
    command: str = ""  # Would be defaulted to the entrypoint of the container
    data_folder: str = defaults.DEFAULT_CONTAINER_DATA_PATH  # Would be relative to the context_path
    output_parameters_file: str = defaults.DEFAULT_CONTAINER_OUTPUT_PARAMETERS  # would be relative to the context_path
    secrets: List[str] = []
    experiment_tracking_file: str = ""

    _temp_dir: str = ""

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

    def execute_command(self, map_variable: TypeMapVariable = None, **kwargs):
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

                self._set_parameters(container_return_parameters)  # type: ignore # TODO: Not fixing this for now.
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
    # The dictionary cannot be modified
    kwargs = kwargs_for_init.copy()
    command_type = kwargs.pop("command_type", defaults.COMMAND_TYPE)

    try:
        task_mgr = driver.DriverManager(
            namespace="tasks",
            name=command_type,
            invoke_on_load=True,
            invoke_kwds=kwargs,
        )
        return task_mgr.driver
    except Exception as _e:
        msg = (
            f"Could not find the task type {command_type}. Please ensure you have installed "
            "the extension that provides the node type."
        )
        raise Exception(msg) from _e
