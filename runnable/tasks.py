import contextlib
import importlib
import io
import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator
from stevedore import driver

import runnable.context as context
from runnable import defaults, parameters, utils
from runnable.datastore import JsonParameter, ObjectParameter, Parameter, StepAttempt
from runnable.defaults import TypeMapVariable
from runnable.nodes import BaseNode

logger = logging.getLogger(defaults.LOGGER_NAME)
logging.getLogger("stevedore").setLevel(logging.CRITICAL)


# TODO: Can we add memory peak, cpu usage, etc. to the metrics?


class BaseTaskType(BaseModel):
    """A base task class which does the execution of command defined by the user."""

    task_type: str = Field(serialization_alias="command_type")
    node_name: str = Field(exclude=True)
    secrets: Dict[str, str] = Field(default_factory=dict)
    returns: List[str] = Field(default_factory=list, alias="returns")

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

    def set_secrets_as_env_variables(self):
        for key, value in self.secrets.items():
            secret_value = context.run_context.secrets_handler.get(key)
            self.secrets[value] = secret_value
            os.environ[value] = secret_value

    def delete_secrets_from_env_variables(self):
        for _, value in self.secrets.items():
            if value in os.environ:
                del os.environ[value]

    def execute_command(
        self,
        node: BaseNode,
        map_variable: TypeMapVariable = None,
        **kwargs,
    ) -> StepAttempt:
        """The function to execute the command.

        And map_variable is sent in as an argument into the function.

        Args:
            map_variable (dict, optional): If the command is part of map node, the value of map. Defaults to None.

        Raises:
            NotImplementedError: Base class, not implemented
        """
        raise NotImplementedError()

    @contextlib.contextmanager
    def expose_secrets(self):
        """Context manager to expose secrets to the execution.

        Args:
            map_variable (dict, optional): If the command is part of map node, the value of map. Defaults to None.

        """
        self.set_secrets_as_env_variables()
        try:
            yield
        finally:
            self.delete_secrets_from_env_variables()

    @contextlib.contextmanager
    def execution_context(self, map_variable: TypeMapVariable = None, allow_complex: bool = False):
        params = self._context.run_log_store.get_parameters(run_id=self._context.run_id).copy()

        log_file_name = self.node_name.replace(" ", "_") + ".execution.log"
        if map_variable:
            for _, value in map_variable.items():
                log_file_name += "_" + str(value)

        log_file = open(log_file_name, "w")

        f = io.StringIO()
        try:
            with contextlib.redirect_stdout(f):
                yield params
        finally:
            print(f.getvalue())  # print to console
            log_file.write(f.getvalue())  # Print to file

            f.close()
            log_file.close()
            # put_in_catalog(log_file.name)
            os.remove(log_file.name)

            # Update parameters
            self._context.run_log_store.set_parameters(parameters=params, run_id=self._context.run_id)


def classify_value_as_json_or_object(name: str, value: Any) -> Parameter:
    try:
        if isinstance(value, BaseModel):
            # nested pydantic models
            return JsonParameter(value=value.model_dump(by_alias=True), kind="json")

        # simpler types
        json_value = json.load(value)
        return JsonParameter(json_value, kind="json")
    except (json.JSONDecodeError, AttributeError):
        # We cannot serialize the value as json. This has to pickled and stored.
        # return the location of the object
        obj = ObjectParameter(value=name, kind="object")
        obj.put_object(data=value)
        return obj


class PythonTaskType(BaseTaskType):  # pylint: disable=too-few-public-methods
    """The task class for python command."""

    task_type: str = Field(default="python", serialization_alias="command_type")
    command: str

    def get_cli_options(self) -> Tuple[str, dict]:
        """Return the cli options for the task.

        Returns:
            dict: The cli options for the task
        """
        return "function", {"command": self.command}

    def execute_command(
        self,
        map_variable: TypeMapVariable = None,
        **kwargs,
    ) -> StepAttempt:
        """Execute the notebook as defined by the command."""
        attempt_log = StepAttempt(status=defaults.FAIL, start_time=str(datetime.now()))

        with self.execution_context(map_variable=map_variable) as params, self.expose_secrets() as _:
            module, func = utils.get_module_and_attr_names(self.command)
            sys.path.insert(0, os.getcwd())  # Need to add the current directory to path
            imported_module = importlib.import_module(module)
            f = getattr(imported_module, func)

            filtered_parameters = parameters.filter_arguments_for_func(f, params, map_variable)

            logger.info(f"Calling {func} from {module} with {filtered_parameters}")

            try:
                user_set_parameters = f(**filtered_parameters)
                attempt_log.input_parameters = params.copy()

                output_parameters: Dict[str, Parameter] = {}

                if isinstance(user_set_parameters, BaseModel):
                    # Support pydantic models
                    for key, value in dict(user_set_parameters).items():
                        # The value could either be a simple type or object
                        output_parameters[key] = classify_value_as_json_or_object(name=key, value=value)
                else:
                    # retrieve from returns
                    for i, return_key in enumerate(self.returns):
                        # The value could either be a simple type or object
                        output_parameters[return_key] = classify_value_as_json_or_object(
                            name=return_key,
                            value=user_set_parameters[i],
                        )

                if output_parameters:
                    attempt_log.output_parameters = output_parameters
                    params.update(output_parameters)

                attempt_log.status = defaults.SUCCESS

            except Exception as _e:
                msg = f"Call to the function {self.command} with {filtered_parameters} did not succeed.\n"
                logger.exception(msg)
                logger.exception(_e)
                attempt_log.status = defaults.FAIL

        attempt_log.end_time = str(datetime.now())

        return attempt_log


class NotebookTaskType(BaseTaskType):
    """The task class for Notebook based execution."""

    task_type: str = Field(default="notebook", serialization_alias="command_type")
    command: str
    notebook_output_path: str = Field(default="", validate_default=True)
    returns: Optional[List[str]] = Field(default_factory=list)
    output_cell_tag: str = Field(default="runnable_output", validate_default=True)
    optional_ploomber_args: dict = {}

    @field_validator("command")
    @classmethod
    def notebook_should_end_with_ipynb(cls, command: str):
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

    def execute_command(
        self,
        node: BaseNode,
        map_variable: TypeMapVariable = None,
        **kwargs,
    ) -> StepAttempt:
        """Execute the python notebook as defined by the command.

        Args:
            map_variable (dict, optional): If the node is part of internal branch. Defaults to None.

        Raises:
            ImportError: If necessary dependencies are not installed
            Exception: If anything else fails
        """

        try:
            import ploomber_engine as pm
            from ploomber_engine.ipython import PloomberClient

            from runnable import put_in_catalog  # Causes issues with cyclic import

            notebook_output_path = self.notebook_output_path

            with self.execution_context(node=node, map_variable=map_variable) as params, self.expose_secrets() as _:
                if map_variable:
                    for key, value in map_variable.items():
                        notebook_output_path += "_" + str(value)

                        params[key] = value

                ploomber_optional_args = self.optional_ploomber_args

                kwds = {
                    "input_path": self.command,
                    "output_path": notebook_output_path,
                    "parameters": params,
                    "log_output": True,
                    "progress_bar": False,
                }
                kwds.update(ploomber_optional_args)

                collected_params: Dict[str, Any] = {}

                pm.execute_notebook(**kwds)

                put_in_catalog(notebook_output_path)

            client = PloomberClient.from_path(path=notebook_output_path)
            namespace = client.get_namespace()

            for key, value in namespace.items():
                if key in (self.returns or []):
                    if isinstance(value, BaseModel):
                        collected_params[key] = value.model_dump(by_alias=True)
                        continue
                    collected_params[key] = value

            return collected_params

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
    returns: Optional[List[str]] = Field(default_factory=list)
    command: str

    def execute_command(
        self,
        node: BaseNode,
        map_variable: TypeMapVariable = None,
        **kwargs,
    ) -> StepAttempt:
        # Using shell=True as we want to have chained commands to be executed in the same shell.
        """Execute the shell command as defined by the command.

        Args:
            map_variable (dict, optional): If the node is part of an internal branch. Defaults to None.
        """

        runnable_env_vars: Dict[str, Any] = {}

        # Expose RUNNABLE environment variables, ignoring the parameters, to be passed to the subprocess.
        for key, value in os.environ.items():
            if key.startswith("RUNNABLE_") and not key.startswith("RUNNABLE_PRM_"):
                runnable_env_vars[key] = value

        subprocess_env = {}  # {**params, **runnable_env_vars}

        # Expose map variable as environment variables
        if map_variable:
            for key, value in map_variable.items():  # type: ignore
                subprocess_env[key] = str(value)

        # Expose secrets as environment variables
        if self.secrets:
            for key, value in self.secrets.items():
                secret_value = context.run_context.secrets_handler.get(key)
                subprocess_env[value] = secret_value

        # Json dumps all runnable environment variables
        for key, value in subprocess_env.items():
            subprocess_env[key] = json.dumps(value)

        collect_delimiter = "=== COLLECT ==="

        command = self.command.strip() + f" && echo '{collect_delimiter}'  && env"
        logger.info(f"Executing shell command: {command}")

        output_parameters = {}
        capture = False

        with subprocess.Popen(
            command,
            shell=True,
            env=subprocess_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ) as proc, self.execution_context(node=node, map_variable=map_variable):
            for line in proc.stdout:  # type: ignore
                logger.info(line)
                print(line)

                if line.strip() == collect_delimiter:
                    capture = True
                    continue

                if capture:
                    key, value = line.strip().split("=", 1)
                    if key in (self.returns or []):
                        try:
                            output_parameters[key] = json.loads(value)
                        except json.JSONDecodeError:
                            output_parameters[key] = value  # simple data types

            proc.wait()
            if proc.returncode != 0:
                raise Exception("Command failed")

        return output_parameters


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
