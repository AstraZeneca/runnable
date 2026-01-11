import contextlib
import copy
import importlib
import inspect
import io
import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from pickle import PicklingError
from string import Template
from typing import Any, Callable, Dict, List, Literal, Optional, cast

import logfire_api as logfire
from pydantic import BaseModel, ConfigDict, Field, field_validator
from rich.segment import Segment
from rich.style import Style
from stevedore import driver

import runnable.context as context
from runnable import console, defaults, exceptions, parameters, task_console, utils
from runnable.datastore import (
    JsonParameter,
    MetricParameter,
    ObjectParameter,
    Parameter,
    StepAttempt,
)
from runnable.defaults import IterableParameterModel
from runnable.telemetry import truncate_value

logger = logging.getLogger(defaults.LOGGER_NAME)


class TeeIO(io.StringIO):
    """
    A custom class to write to the buffer, output stream, and Rich console simultaneously.

    This implementation directly adds to Rich Console's internal recording buffer using
    proper Segment objects, avoiding the infinite recursion that occurs when using
    Rich Console's print() method.
    """

    def __init__(
        self, output_stream=sys.stdout, rich_console=None, stream_type="stdout"
    ):
        super().__init__()
        self.output_stream = output_stream
        self.rich_console = rich_console
        self.stream_type = stream_type

    def write(self, s):
        if s:  # Only process non-empty strings
            super().write(s)  # Write to the buffer for later retrieval
            self.output_stream.write(s)  # Display immediately

            # Record directly to Rich's internal buffer using proper Segments
            # Note: We record ALL content including newlines, not just stripped content
            if self.rich_console:
                if self.stream_type == "stderr":
                    # Red style for stderr
                    style = Style(color="red")
                    segment = Segment(s, style)
                else:
                    # No style for stdout
                    segment = Segment(s)

                # Add to Rich's record buffer (no recursion!)
                self.rich_console._record_buffer.append(segment)

        return len(s) if s else 0

    def flush(self):
        super().flush()
        self.output_stream.flush()


@contextlib.contextmanager
def redirect_output(console=None):
    """
    Context manager that captures output to both display and Rich console recording.

    This implementation uses TeeIO which directly records to Rich Console's internal
    buffer, eliminating the need for post-processing and avoiding infinite recursion.

    Args:
        console: Rich Console instance for recording (typically task_console)
    """
    # Backup the original stdout and stderr
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    # Create TeeIO instances that handle display + Rich console recording simultaneously
    sys.stdout = TeeIO(original_stdout, rich_console=console, stream_type="stdout")
    sys.stderr = TeeIO(original_stderr, rich_console=console, stream_type="stderr")

    # Update logging handlers to use the new stdout
    original_streams = []
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.StreamHandler):
            original_streams.append((handler, handler.stream))
            handler.stream = sys.stdout

    try:
        yield sys.stdout, sys.stderr
    finally:
        # Restore the original stdout and stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr

        # Restore logging handler streams
        for handler, original_stream in original_streams:
            handler.stream = original_stream

        # No additional Rich console processing needed - TeeIO handles it directly!


class TaskReturns(BaseModel):
    name: str
    kind: Literal["json", "object", "metric"] = Field(default="json")


class BaseTaskType(BaseModel):
    """A base task class which does the execution of command defined by the user."""

    task_type: str = Field(serialization_alias="command_type")
    secrets: List[str] = Field(
        default_factory=list
    )  # A list of secrets to expose by secrets manager
    returns: List[TaskReturns] = Field(
        default_factory=list, alias="returns"
    )  # The return values of the task
    internal_branch_name: str = Field(default="")

    model_config = ConfigDict(extra="forbid")

    def get_summary(self) -> Dict[str, Any]:
        return self.model_dump(by_alias=True)

    @property
    def _context(self):
        current_context = context.get_run_context()
        if current_context is None:
            raise RuntimeError("No run context available in current execution context")
        return current_context

    def set_secrets_as_env_variables(self):
        # Preparing the environment for the task execution
        current_context = context.get_run_context()
        if current_context is None:
            raise RuntimeError("No run context available for secrets")

        for key in self.secrets:
            secret_value = current_context.secrets.get(key)
            os.environ[key] = secret_value

    def delete_secrets_from_env_variables(self):
        # Cleaning up the environment after the task execution
        for key in self.secrets:
            if key in os.environ:
                del os.environ[key]

    def execute_command(
        self,
        iter_variable: Optional[IterableParameterModel] = None,
    ) -> StepAttempt:
        """The function to execute the command.

        And map_variable is sent in as an argument into the function.

        Args:
            map_variable (dict, optional): If the command is part of map node, the value of map. Defaults to None.

        Raises:
            NotImplementedError: Base class, not implemented
        """
        raise NotImplementedError()

    async def execute_command_async(
        self,
        iter_variable: Optional[IterableParameterModel] = None,
        event_callback: Optional[Callable[[dict], None]] = None,
    ) -> StepAttempt:
        """
        Async command execution.

        Only implemented by task types that support async execution
        (AsyncPythonTaskType). Sync task types (PythonTaskType,
        NotebookTaskType, ShellTaskType) raise NotImplementedError.

        Args:
            map_variable: If the command is part of map node.
            event_callback: Optional callback for streaming events.

        Raises:
            NotImplementedError: If task type does not support async execution.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support async execution. "
            f"Use AsyncPythonTask for async functions."
        )

    def _diff_parameters(
        self, parameters_in: Dict[str, Parameter], context_params: Dict[str, Parameter]
    ) -> Dict[str, Parameter]:
        # If the parameter is different from existing parameters, then it is updated
        diff: Dict[str, Parameter] = {}
        for param_name, param in context_params.items():
            if param_name in parameters_in:
                if parameters_in[param_name] != param:
                    diff[param_name] = param
                continue

            diff[param_name] = param

        return diff

    @contextlib.contextmanager
    def expose_secrets(self):
        """Context manager to expose secrets to the execution."""
        self.set_secrets_as_env_variables()
        try:
            yield
        except Exception as e:  # pylint: disable=broad-except
            logger.exception(e)
        finally:
            self.delete_secrets_from_env_variables()

    def _safe_serialize_params(self, params: Dict[str, Parameter]) -> Dict[str, Any]:
        """Safely serialize parameters for telemetry, truncating per value.

        ObjectParameter values are not serializable (pickled objects),
        so they are represented as "<object>".
        """
        serializable: Dict[str, Any] = {}
        for k, v in params.items():
            if isinstance(v, ObjectParameter):
                serializable[k] = "<object>"
            else:
                serializable[k] = truncate_value(v.get_value())
        return serializable

    def _emit_event(self, event: Dict[str, Any]) -> None:
        """Push event to stream queue if one is set (for SSE streaming)."""
        from runnable.telemetry import get_stream_queue

        q = get_stream_queue()
        if q is not None:
            q.put_nowait(event)

    @contextlib.contextmanager
    def execution_context(
        self,
        iter_variable: Optional[IterableParameterModel] = None,
        allow_complex: bool = True,
    ):
        params = self._context.run_log_store.get_parameters(
            run_id=self._context.run_id, internal_branch_name=self.internal_branch_name
        ).copy()
        logger.info(f"Parameters available for the execution: {params}")

        task_console.log("Parameters available for the execution:")
        task_console.log(params)

        logger.debug(f"Resolved parameters: {params}")

        if not allow_complex:
            params = {
                key: value
                for key, value in params.items()
                if isinstance(value, JsonParameter)
                or isinstance(value, MetricParameter)
            }

        parameters_in = copy.deepcopy(params)
        try:
            yield params
        except Exception as e:  # pylint: disable=broad-except
            console.log(e, style=defaults.error_style)
            logger.exception(e)
        finally:
            # Update parameters
            # This should only update the parameters that are changed at the root level.
            diff_parameters = self._diff_parameters(
                parameters_in=parameters_in, context_params=params
            )
            self._context.run_log_store.set_parameters(
                parameters=diff_parameters,
                run_id=self._context.run_id,
                internal_branch_name=self.internal_branch_name,
            )


def task_return_to_parameter(task_return: TaskReturns, value: Any) -> Parameter:
    # implicit support for pydantic models
    if isinstance(value, BaseModel) and task_return.kind == "json":
        try:
            return JsonParameter(kind="json", value=value.model_dump(by_alias=True))
        except PicklingError:
            logging.warning("Pydantic model is not serializable")

    if task_return.kind == "json":
        return JsonParameter(kind="json", value=value)

    if task_return.kind == "metric":
        return MetricParameter(kind="metric", value=value)

    if task_return.kind == "object":
        obj = ObjectParameter(value=task_return.name, kind="object")
        obj.put_object(data=value)
        return obj

    raise Exception(f"Unknown return type: {task_return.kind}")


class PythonTaskType(BaseTaskType):  # pylint: disable=too-few-public-methods
    """
    --8<-- [start:python_reference]
    An execution node of the pipeline of python functions.
    Please refer to define pipeline/tasks/python for more information.

    As part of the dag definition, a python task is defined as follows:

    dag:
      steps:
        python_task: # The name of the node
          type: task
          command_type: python # this is default
          command: my_module.my_function # the dotted path to the function. Please refer to the yaml section of
            define pipeline/tasks/python for concrete details.
          returns:
            - name: # The name to assign the return value
              kind: json # the default value is json,
                can be object for python objects and metric for metrics
          secrets:
            - my_secret_key # A list of secrets to expose by secrets manager
          catalog:
            get:
              - A list of glob patterns to get from the catalog to the local file system
            put:
              - A list of glob patterns to put to the catalog from the local file system
          on_failure: The name of the step to traverse in case of failure
          overrides:
            Individual tasks can override the global configuration config by referring to the
            specific override.

            For example,
            #Global configuration
            executor:
            type: local-container
            config:
              docker_image: "runnable/runnable:latest"
              overrides:
              custom_docker_image:
                docker_image: "runnable/runnable:custom"

            ## In the node definition
            overrides:
            local-container:
              docker_image: "runnable/runnable:custom"

            This instruction will override the docker image for the local-container executor.
          next: The next node to execute after this task, use "success" to terminate the pipeline successfully
            or "fail" to terminate the pipeline with an error.
    --8<-- [end:python_reference]
    """

    task_type: str = Field(default="python", serialization_alias="command_type")
    command: str

    def execute_command(
        self,
        iter_variable: Optional[IterableParameterModel] = None,
    ) -> StepAttempt:
        """Execute the notebook as defined by the command."""
        attempt_log = StepAttempt(
            status=defaults.FAIL,
            start_time=str(datetime.now()),
            retry_indicator=self._context.retry_indicator,
        )

        with logfire.span(
            "task:{task_name}",
            task_name=self.command,
            task_type=self.task_type,
        ):
            with (
                self.execution_context(iter_variable=iter_variable) as params,
                self.expose_secrets() as _,
            ):
                logfire.info(
                    "Task started",
                    inputs=self._safe_serialize_params(params),
                )
                self._emit_event(
                    {
                        "type": "task_started",
                        "name": self.command,
                        "inputs": self._safe_serialize_params(params),
                    }
                )

                module, func = utils.get_module_and_attr_names(self.command)
                sys.path.insert(
                    0, os.getcwd()
                )  # Need to add the current directory to path
                imported_module = importlib.import_module(module)
                f = getattr(imported_module, func)

                try:
                    try:
                        filtered_parameters = parameters.filter_arguments_for_func(
                            f, params.copy(), iter_variable
                        )
                        logger.info(
                            f"Calling {func} from {module} with {filtered_parameters}"
                        )
                        with redirect_output(console=task_console) as (
                            buffer,
                            stderr_buffer,
                        ):
                            user_set_parameters = f(
                                **filtered_parameters
                            )  # This is a tuple or single value
                    except Exception as e:
                        raise exceptions.CommandCallError(
                            f"Function call: {self.command} did not succeed.\n"
                        ) from e
                    finally:
                        attempt_log.input_parameters = params.copy()
                        if iter_variable and iter_variable.map_variable:
                            attempt_log.input_parameters.update(
                                {
                                    k: JsonParameter(value=v.value, kind="json")
                                    for k, v in iter_variable.map_variable.items()
                                }
                            )

                    if self.returns:
                        if not isinstance(
                            user_set_parameters, tuple
                        ):  # make it a tuple
                            user_set_parameters = (user_set_parameters,)

                        if len(user_set_parameters) != len(self.returns):
                            raise ValueError(
                                "Returns task signature does not match the function returns"
                            )

                        output_parameters: Dict[str, Parameter] = {}
                        metrics: Dict[str, Parameter] = {}

                        for i, task_return in enumerate(self.returns):
                            output_parameter = task_return_to_parameter(
                                task_return=task_return,
                                value=user_set_parameters[i],
                            )

                            if task_return.kind == "metric":
                                metrics[task_return.name] = output_parameter

                            output_parameters[task_return.name] = output_parameter

                        attempt_log.output_parameters = output_parameters
                        attempt_log.user_defined_metrics = metrics
                        params.update(output_parameters)

                        logfire.info(
                            "Task completed",
                            outputs=self._safe_serialize_params(output_parameters),
                            status="success",
                        )
                        self._emit_event(
                            {
                                "type": "task_completed",
                                "name": self.command,
                                "outputs": self._safe_serialize_params(
                                    output_parameters
                                ),
                            }
                        )
                    else:
                        logfire.info("Task completed", status="success")
                        self._emit_event(
                            {
                                "type": "task_completed",
                                "name": self.command,
                            }
                        )

                    attempt_log.status = defaults.SUCCESS
                except Exception as _e:
                    msg = f"Call to the function {self.command} did not succeed.\n"
                    attempt_log.message = msg
                    task_console.print_exception(show_locals=False)
                    task_console.log(_e, style=defaults.error_style)
                    logfire.error("Task failed", error=str(_e)[:256])
                    self._emit_event(
                        {
                            "type": "task_error",
                            "name": self.command,
                            "error": str(_e)[:256],
                        }
                    )

        attempt_log.end_time = str(datetime.now())

        return attempt_log


class NotebookTaskType(BaseTaskType):
    """
    --8<-- [start:notebook_reference]
    An execution node of the pipeline of notebook execution.
    Please refer to define pipeline/tasks/notebook for more information.

    As part of the dag definition, a notebook task is defined as follows:

    dag:
      steps:
        notebook_task: # The name of the node
          type: task
          command_type: notebook
          command: the path to the notebook relative to project root.
          optional_ploomber_args: a dictionary of arguments to be passed to ploomber engine
          returns:
            - name: # The name to assign the return value
              kind: json # the default value is json,
                can be object for python objects and metric for metrics
          secrets:
            - my_secret_key # A list of secrets to expose by secrets manager
          catalog:
            get:
              - A list of glob patterns to get from the catalog to the local file system
            put:
              - A list of glob patterns to put to the catalog from the local file system
          on_failure: The name of the step to traverse in case of failure
          overrides:
            Individual tasks can override the global configuration config by referring to the
            specific override.

            For example,
            #Global configuration
            executor:
            type: local-container
            config:
              docker_image: "runnable/runnable:latest"
              overrides:
                custom_docker_image:
                  docker_image: "runnable/runnable:custom"

            ## In the node definition
            overrides:
              local-container:
                docker_image: "runnable/runnable:custom"

            This instruction will override the docker image for the local-container executor.
          next: The next node to execute after this task, use "success" to terminate the pipeline successfully
            or "fail" to terminate the pipeline with an error.
    --8<-- [end:notebook_reference]
    """

    task_type: str = Field(default="notebook", serialization_alias="command_type")
    command: str
    optional_ploomber_args: dict = {}

    @field_validator("command")
    @classmethod
    def notebook_should_end_with_ipynb(cls, command: str) -> str:
        if not command.endswith(".ipynb"):
            raise Exception("Notebook task should point to a ipynb file")

        return command

    def get_notebook_output_path(
        self,
        iter_variable: Optional[IterableParameterModel] = None,
    ) -> str:
        tag = ""
        if iter_variable and iter_variable.map_variable:
            for key, value_model in iter_variable.map_variable.items():
                tag += f"{key}_{value_model.value}_"

        if isinstance(self._context, context.PipelineContext):
            assert self._context.pipeline_executor._context_node
            tag += self._context.pipeline_executor._context_node.name

        tag = "".join(x for x in tag if x.isalnum()).strip("-")

        output_path = Path(".", self.command)
        file_name = output_path.parent / (output_path.stem + f"-{tag}_out.ipynb")

        return str(file_name)

    def execute_command(
        self,
        iter_variable: Optional[IterableParameterModel] = None,
    ) -> StepAttempt:
        """Execute the python notebook as defined by the command.

        Args:
            map_variable (dict, optional): If the node is part of internal branch. Defaults to None.

        Raises:
            ImportError: If necessary dependencies are not installed
            Exception: If anything else fails
        """
        attempt_log = StepAttempt(
            status=defaults.FAIL,
            start_time=str(datetime.now()),
            retry_indicator=self._context.retry_indicator,
        )

        with logfire.span(
            "task:{task_name}",
            task_name=self.command,
            task_type=self.task_type,
        ):
            try:
                import ploomber_engine as pm
                from ploomber_engine.ipython import PloomberClient

                notebook_output_path = self.get_notebook_output_path(
                    iter_variable=iter_variable
                )

                with (
                    self.execution_context(
                        iter_variable=iter_variable, allow_complex=False
                    ) as params,
                    self.expose_secrets() as _,
                ):
                    logfire.info(
                        "Task started",
                        inputs=self._safe_serialize_params(params),
                    )
                    self._emit_event(
                        {
                            "type": "task_started",
                            "name": self.command,
                            "inputs": self._safe_serialize_params(params),
                        }
                    )

                    attempt_log.input_parameters = params.copy()
                    copy_params = copy.deepcopy(params)

                    if iter_variable and iter_variable.map_variable:
                        for key, value_model in iter_variable.map_variable.items():
                            copy_params[key] = JsonParameter(
                                kind="json", value=value_model.value
                            )

                    notebook_params = {k: v.get_value() for k, v in copy_params.items()}

                    ploomber_optional_args = self.optional_ploomber_args

                    kwds = {
                        "input_path": self.command,
                        "output_path": notebook_output_path,
                        "parameters": notebook_params,
                        "log_output": True,
                        "progress_bar": False,
                    }
                    kwds.update(ploomber_optional_args)

                    with redirect_output(console=task_console) as (
                        buffer,
                        stderr_buffer,
                    ):
                        pm.execute_notebook(**kwds)

                    current_context = context.get_run_context()
                    if current_context is None:
                        raise RuntimeError(
                            "No run context available for catalog operations"
                        )
                    current_context.catalog.put(name=notebook_output_path)

                    client = PloomberClient.from_path(path=notebook_output_path)
                    namespace = client.get_namespace()

                    output_parameters: Dict[str, Parameter] = {}
                    try:
                        for task_return in self.returns:
                            template_vars = {}
                            if iter_variable and iter_variable.map_variable:
                                template_vars = {
                                    k: v.value
                                    for k, v in iter_variable.map_variable.items()
                                }
                            param_name = Template(task_return.name).safe_substitute(
                                template_vars  # type: ignore
                            )

                            output_parameters[param_name] = task_return_to_parameter(
                                task_return=task_return,
                                value=namespace[task_return.name],
                            )
                    except PicklingError as e:
                        logger.exception("Notebooks cannot return objects")
                        logger.exception(e)
                        logfire.error("Notebook pickling error", error=str(e)[:256])
                        raise

                    if output_parameters:
                        attempt_log.output_parameters = output_parameters
                        params.update(output_parameters)
                        logfire.info(
                            "Task completed",
                            outputs=self._safe_serialize_params(output_parameters),
                            status="success",
                        )
                        self._emit_event(
                            {
                                "type": "task_completed",
                                "name": self.command,
                                "outputs": self._safe_serialize_params(
                                    output_parameters
                                ),
                            }
                        )
                    else:
                        logfire.info("Task completed", status="success")
                        self._emit_event(
                            {
                                "type": "task_completed",
                                "name": self.command,
                            }
                        )

                    attempt_log.status = defaults.SUCCESS

            except (ImportError, Exception) as e:
                msg = (
                    f"Call to the notebook command {self.command} did not succeed.\n"
                    "Ensure that you have installed runnable with notebook extras"
                )
                logger.exception(msg)
                logger.exception(e)
                logfire.error("Task failed", error=str(e)[:256])
                self._emit_event(
                    {"type": "task_error", "name": self.command, "error": str(e)[:256]}
                )

                attempt_log.status = defaults.FAIL

        attempt_log.end_time = str(datetime.now())

        return attempt_log


class ShellTaskType(BaseTaskType):
    """
    --8<-- [start:shell_reference]
    An execution node of the pipeline of shell execution.
    Please refer to define pipeline/tasks/shell for more information.

    As part of the dag definition, a shell task is defined as follows:

    dag:
      steps:
        shell_task: # The name of the node
          type: task
          command_type: shell
          command: The command to execute, it could be multiline
          optional_ploomber_args: a dictionary of arguments to be passed to ploomber engine
          returns:
            - name: # The name to assign the return value
            kind: json # the default value is json,
                can be object for python objects and metric for metrics
          secrets:
            - my_secret_key # A list of secrets to expose by secrets manager
          catalog:
            get:
              - A list of glob patterns to get from the catalog to the local file system
            put:
              - A list of glob patterns to put to the catalog from the local file system
          on_failure: The name of the step to traverse in case of failure
          overrides:
            Individual tasks can override the global configuration config by referring to the
            specific override.

            For example,
            #Global configuration
            executor:
            type: local-container
            config:
              docker_image: "runnable/runnable:latest"
              overrides:
                custom_docker_image:
                  docker_image: "runnable/runnable:custom"

            ## In the node definition
            overrides:
              local-container:
                docker_image: "runnable/runnable:custom"

            This instruction will override the docker image for the local-container executor.
          next: The next node to execute after this task, use "success" to terminate the pipeline successfully
            or "fail" to terminate the pipeline with an error.
    --8<-- [end:shell_reference]
    """

    task_type: str = Field(default="shell", serialization_alias="command_type")
    command: str

    @field_validator("returns")
    @classmethod
    def returns_should_be_json(cls, returns: List[TaskReturns]):
        for task_return in returns:
            if task_return.kind == "object" or task_return.kind == "pydantic":
                raise ValueError(
                    "Pydantic models or Objects are not allowed in returns"
                )

        return returns

    def execute_command(
        self,
        iter_variable: Optional[IterableParameterModel] = None,
    ) -> StepAttempt:
        # Using shell=True as we want to have chained commands to be executed in the same shell.
        """Execute the shell command as defined by the command.

        Args:
            map_variable (dict, optional): If the node is part of an internal branch. Defaults to None.
        """
        attempt_log = StepAttempt(
            status=defaults.FAIL,
            start_time=str(datetime.now()),
            retry_indicator=self._context.retry_indicator,
        )
        subprocess_env = {}

        # Expose RUNNABLE environment variables to be passed to the subprocess.
        for key, value in os.environ.items():
            if key.startswith("RUNNABLE_"):
                subprocess_env[key] = value

        # Expose map variable as environment variables
        if iter_variable and iter_variable.map_variable:
            for key, value_model in iter_variable.map_variable.items():
                subprocess_env[key] = str(value_model.value)

        # Expose secrets as environment variables
        if self.secrets:
            current_context = context.get_run_context()
            if current_context is None:
                raise RuntimeError("No run context available for secrets")

            for key in self.secrets:
                secret_value = current_context.secrets.get(key)
                subprocess_env[key] = secret_value

        with logfire.span(
            "task:{task_name}",
            task_name=self.command[:100],  # Truncate long commands
            task_type=self.task_type,
        ):
            try:
                with self.execution_context(
                    iter_variable=iter_variable, allow_complex=False
                ) as params:
                    logfire.info(
                        "Task started",
                        inputs=self._safe_serialize_params(params),
                    )
                    self._emit_event(
                        {
                            "type": "task_started",
                            "name": self.command[:100],
                            "inputs": self._safe_serialize_params(params),
                        }
                    )

                    subprocess_env.update({k: v.get_value() for k, v in params.items()})

                    attempt_log.input_parameters = params.copy()
                    # Json dumps all runnable environment variables
                    for key, value in subprocess_env.items():
                        if isinstance(value, str):
                            continue
                        subprocess_env[key] = json.dumps(value)

                    collect_delimiter = "=== COLLECT ==="

                    command = (
                        self.command.strip() + f" && echo '{collect_delimiter}'  && env"
                    )
                    logger.info(f"Executing shell command: {command}")

                    capture = False
                    return_keys = {x.name: x for x in self.returns}

                    proc = subprocess.Popen(
                        command,
                        shell=True,
                        env=subprocess_env,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                    result = proc.communicate()
                    logger.debug(result)
                    logger.info(proc.returncode)

                    if proc.returncode != 0:
                        msg = ",".join(result[1].split("\n"))
                        task_console.print(msg, style=defaults.error_style)
                        raise exceptions.CommandCallError(msg)

                    # for stderr
                    for line in result[1].split("\n"):
                        if line.strip() == "":
                            continue
                        task_console.print(line, style=defaults.warning_style)

                    output_parameters: Dict[str, Parameter] = {}
                    metrics: Dict[str, Parameter] = {}

                    # only from stdout
                    for line in result[0].split("\n"):
                        if line.strip() == "":
                            continue

                        logger.info(line)
                        task_console.print(line)

                        if line.strip() == collect_delimiter:
                            # The lines from now on should be captured
                            capture = True
                            continue

                        if capture:
                            key, value = line.strip().split("=", 1)
                            if key in return_keys:
                                task_return = return_keys[key]

                                try:
                                    value = json.loads(value)
                                except json.JSONDecodeError:
                                    value = value

                                output_parameter = task_return_to_parameter(
                                    task_return=task_return,
                                    value=value,
                                )

                                if task_return.kind == "metric":
                                    metrics[task_return.name] = output_parameter

                                output_parameters[task_return.name] = output_parameter

                        attempt_log.output_parameters = output_parameters
                        attempt_log.user_defined_metrics = metrics
                        params.update(output_parameters)

                    if output_parameters:
                        logfire.info(
                            "Task completed",
                            outputs=self._safe_serialize_params(output_parameters),
                            status="success",
                        )
                        self._emit_event(
                            {
                                "type": "task_completed",
                                "name": self.command[:100],
                                "outputs": self._safe_serialize_params(
                                    output_parameters
                                ),
                            }
                        )
                    else:
                        logfire.info("Task completed", status="success")
                        self._emit_event(
                            {
                                "type": "task_completed",
                                "name": self.command[:100],
                            }
                        )

                    attempt_log.status = defaults.SUCCESS
            except exceptions.CommandCallError as e:
                msg = f"Call to the command {self.command} did not succeed"
                logger.exception(msg)
                logger.exception(e)

                task_console.log(msg, style=defaults.error_style)
                task_console.log(e, style=defaults.error_style)
                logfire.error("Task failed", error=str(e)[:256])
                self._emit_event(
                    {
                        "type": "task_error",
                        "name": self.command[:100],
                        "error": str(e)[:256],
                    }
                )

                attempt_log.status = defaults.FAIL

        attempt_log.end_time = str(datetime.now())
        return attempt_log


class AsyncPythonTaskType(BaseTaskType):
    """
    An execution node for async Python functions.

    This task type is designed for async functions that need to be awaited.
    It supports AsyncGenerator functions for streaming events.

    Usage in pipeline definition:
        task = AsyncPythonTask(
            function=my_async_function,
            name="async_task",
            returns=[...]
        )
    """

    task_type: str = Field(default="async-python", serialization_alias="command_type")
    command: str
    stream_end_type: str = Field(default="done")

    def execute_command(
        self,
        iter_variable: Optional[IterableParameterModel] = None,
    ) -> StepAttempt:
        """Sync execution is not supported for async tasks."""
        raise RuntimeError(
            "AsyncPythonTaskType requires async execution. "
            "Use execute_command_async() or run the pipeline with execute_async()."
        )

    async def execute_command_async(
        self,
        iter_variable: Optional[IterableParameterModel] = None,
        event_callback: Optional[Callable[[dict], None]] = None,
    ) -> StepAttempt:
        """Execute the async Python function."""
        attempt_log = StepAttempt(
            status=defaults.FAIL,
            start_time=str(datetime.now()),
            retry_indicator=self._context.retry_indicator,
        )

        with logfire.span(
            "task:{task_name}",
            task_name=self.command,
            task_type=self.task_type,
        ):
            with (
                self.execution_context(iter_variable=iter_variable) as params,
                self.expose_secrets() as _,
            ):
                logfire.info(
                    "Task started",
                    inputs=self._safe_serialize_params(params),
                )
                self._emit_event(
                    {
                        "type": "task_started",
                        "name": self.command,
                        "inputs": self._safe_serialize_params(params),
                    }
                )

                module, func = utils.get_module_and_attr_names(self.command)
                sys.path.insert(0, os.getcwd())
                imported_module = importlib.import_module(module)
                f = getattr(imported_module, func)

                try:
                    try:
                        filtered_parameters = parameters.filter_arguments_for_func(
                            f, params.copy(), iter_variable
                        )
                        logger.info(
                            f"Calling async {func} from {module} with {filtered_parameters}"
                        )

                        with redirect_output(console=task_console) as (
                            buffer,
                            stderr_buffer,
                        ):
                            result = f(**filtered_parameters)

                            # Check if result is an AsyncGenerator for streaming
                            if inspect.isasyncgen(result):
                                user_set_parameters = None
                                async for item in result:
                                    if isinstance(item, dict) and "type" in item:
                                        # It's an event - emit it
                                        if event_callback:
                                            event_callback(item)
                                        self._emit_event(item)

                                        # Extract return values from the final event
                                        # The stream end event contains the actual return values
                                        if item.get("type") == self.stream_end_type:
                                            # Remove the "type" key and use remaining keys as return values
                                            return_data = {
                                                k: v
                                                for k, v in item.items()
                                                if k != "type"
                                            }
                                            # If only one value, return it directly; otherwise return tuple
                                            if len(return_data) == 1:
                                                user_set_parameters = list(
                                                    return_data.values()
                                                )[0]
                                            elif len(return_data) > 1:
                                                user_set_parameters = tuple(
                                                    return_data.values()
                                                )
                            elif inspect.iscoroutine(result):
                                # Regular async function
                                user_set_parameters = await result
                            else:
                                # Sync function called through async task (shouldn't happen but handle it)
                                user_set_parameters = result

                    except Exception as e:
                        raise exceptions.CommandCallError(
                            f"Async function call: {self.command} did not succeed.\n"
                        ) from e
                    finally:
                        attempt_log.input_parameters = params.copy()
                        if iter_variable and iter_variable.map_variable:
                            attempt_log.input_parameters.update(
                                {
                                    k: JsonParameter(value=v.value, kind="json")
                                    for k, v in iter_variable.map_variable.items()
                                }
                            )

                    if self.returns:
                        if not isinstance(user_set_parameters, tuple):
                            user_set_parameters = (user_set_parameters,)

                        if len(user_set_parameters) != len(self.returns):
                            raise ValueError(
                                "Returns task signature does not match the function returns"
                            )

                        output_parameters: Dict[str, Parameter] = {}
                        metrics: Dict[str, Parameter] = {}

                        for i, task_return in enumerate(self.returns):
                            output_parameter = task_return_to_parameter(
                                task_return=task_return,
                                value=user_set_parameters[i],
                            )

                            if task_return.kind == "metric":
                                metrics[task_return.name] = output_parameter

                            output_parameters[task_return.name] = output_parameter

                        attempt_log.output_parameters = output_parameters
                        attempt_log.user_defined_metrics = metrics
                        params.update(output_parameters)

                        logfire.info(
                            "Task completed",
                            outputs=self._safe_serialize_params(output_parameters),
                            status="success",
                        )
                        self._emit_event(
                            {
                                "type": "task_completed",
                                "name": self.command,
                                "outputs": self._safe_serialize_params(
                                    output_parameters
                                ),
                            }
                        )
                    else:
                        logfire.info("Task completed", status="success")
                        self._emit_event(
                            {
                                "type": "task_completed",
                                "name": self.command,
                            }
                        )

                    attempt_log.status = defaults.SUCCESS
                except Exception as _e:
                    msg = (
                        f"Call to the async function {self.command} did not succeed.\n"
                    )
                    attempt_log.message = msg
                    task_console.print_exception(show_locals=False)
                    task_console.log(_e, style=defaults.error_style)
                    logfire.error("Task failed", error=str(_e)[:256])
                    self._emit_event(
                        {
                            "type": "task_error",
                            "name": self.command,
                            "error": str(_e)[:256],
                        }
                    )

        attempt_log.end_time = str(datetime.now())

        return attempt_log


def convert_binary_to_string(data):
    """
    Recursively converts 1 and 0 values in a nested dictionary to "1" and "0".

    Args:
        data (dict or any): The input data (dictionary, list, or other).

    Returns:
        dict or any: The modified data with binary values converted to strings.
    """

    if isinstance(data, dict):
        for key, value in data.items():
            data[key] = convert_binary_to_string(value)
        return data
    elif isinstance(data, list):
        return [convert_binary_to_string(item) for item in data]
    elif data == 1:
        return "1"
    elif data == 0:
        return "0"
    else:
        return data  # Return other values unchanged


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

    kwargs = convert_binary_to_string(kwargs)

    try:
        task_mgr: driver.DriverManager = driver.DriverManager(
            namespace="tasks",
            name=command_type,
            invoke_on_load=True,
            invoke_kwds=kwargs,
        )
        return cast(BaseTaskType, task_mgr.driver)
    except Exception as _e:
        msg = (
            f"Could not find the task type {command_type}. Please ensure you have installed "
            "the extension that provides the node type."
        )
        raise Exception(msg) from _e
