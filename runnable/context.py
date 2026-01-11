import contextvars
import hashlib
import importlib
import json
import logging
import os
import sys
from datetime import datetime
from enum import Enum
from functools import cached_property, partial
from typing import TYPE_CHECKING, Annotated, Any, Callable, Dict, Optional

import logfire_api as logfire
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
)
from stevedore import driver

from runnable import console, defaults, exceptions, names, utils
from runnable.catalog import BaseCatalog
from runnable.datastore import BaseRunLogStore
from runnable.executor import BaseJobExecutor, BasePipelineExecutor
from runnable.graph import Graph, create_graph
from runnable.nodes import BaseNode
from runnable.pickler import BasePickler
from runnable.secrets import BaseSecrets
from runnable.tasks import BaseTaskType


logger = logging.getLogger(defaults.LOGGER_NAME)


def get_pipeline_spec_from_yaml(pipeline_file: str) -> Graph:
    """
    Reads the pipeline file from a YAML file and sets the pipeline spec in the run context
    """
    pipeline_config = utils.load_yaml(pipeline_file)
    logger.info("The input pipeline:")
    logger.info(json.dumps(pipeline_config, indent=4))

    dag_config = pipeline_config["dag"]

    dag = create_graph(dag_config)
    return dag


def get_pipeline_spec_from_python(python_module: str) -> Graph:
    # Call the SDK to get the dag
    # Import the module and call the function to get the dag
    module_file = python_module.rstrip(".py")
    module, func = utils.get_module_and_attr_names(module_file)
    sys.path.insert(0, os.getcwd())  # Need to add the current directory to path
    imported_module = importlib.import_module(module)

    dag = getattr(imported_module, func)().return_dag()

    return dag


def get_job_spec_from_python(
    job_file: str,
) -> tuple[BaseTaskType, list[str]]:
    """
    Reads the job file from a Python file and sets the job spec in the run context
    """
    from runnable.sdk import BaseJob

    # Import the module and call the function to get the job
    module_file = job_file.rstrip(".py")
    module, func = utils.get_module_and_attr_names(module_file)
    sys.path.insert(0, os.getcwd())  # Need to add the current directory to path
    imported_module = importlib.import_module(module)

    job: BaseJob = getattr(imported_module, func)()

    return job.get_task(), job.catalog.put if job.catalog else []


def get_service_by_name(namespace: str, service_config: dict[str, Any], _) -> Any:  # noqa: ANN401, ANN001
    """Get the service by name."""
    service_config = service_config.copy()

    kind = service_config.pop("type", None)

    if "config" in service_config:
        service_config = service_config.get("config", {})

    logger.debug(
        f"Trying to get a service of {namespace} with config: {service_config}"
    )
    try:
        mgr: driver.DriverManager = driver.DriverManager(
            namespace=namespace,  # eg: reader
            name=kind,  # eg: csv, pdf
            invoke_on_load=True,
            invoke_kwds={**service_config},
        )
        return mgr.driver
    except Exception as _e:
        raise Exception(
            f"Could not find the service of kind: {kind} in namespace:{namespace} with config: {service_config}"
        ) from _e


def get_service(service: str) -> Callable:
    """Get the service by name.

    Args:
        service (str): service name.

    Returns:
        Callable: callable function of service.
    """
    return partial(get_service_by_name, service)


InstantiatedCatalog = Annotated[BaseCatalog, BeforeValidator(get_service("catalog"))]
InstantiatedSecrets = Annotated[BaseSecrets, BeforeValidator(get_service("secrets"))]
InstantiatedPickler = Annotated[BasePickler, BeforeValidator(get_service("pickler"))]
InstantiatedRunLogStore = Annotated[
    BaseRunLogStore, BeforeValidator(get_service("run_log_store"))
]
InstantiatedPipelineExecutor = Annotated[
    BasePipelineExecutor, BeforeValidator(get_service("pipeline_executor"))
]
InstantiatedJobExecutor = Annotated[
    BaseJobExecutor, BeforeValidator(get_service("job_executor"))
]


class ExecutionMode(str, Enum):
    YAML = "yaml"
    PYTHON = "python"


class ExecutionContext(str, Enum):
    PIPELINE = "pipeline"
    JOB = "job"


class ServiceConfigurations(BaseModel):
    configuration_file: Optional[str] = Field(
        default=None, exclude=True, description="Path to the configuration file."
    )
    execution_context: ExecutionContext = ExecutionContext.PIPELINE
    variables: dict[str, str] = Field(
        default_factory=utils.gather_variables,
        exclude=True,
        description="Variables to be used.",
    )

    @field_validator("configuration_file", mode="before")
    @classmethod
    def override_configuration_file(cls, configuration_file: str | None) -> str | None:
        """Determine the configuration file to use, following the order of precedence."""
        # 1. Environment variable
        env_config = os.environ.get(defaults.RUNNABLE_CONFIGURATION_FILE)
        if env_config:
            return env_config

        # 2. User-provided at runtime
        if configuration_file:
            return configuration_file

        # 3. Default user config file
        if utils.does_file_exist(defaults.USER_CONFIG_FILE):
            return defaults.USER_CONFIG_FILE

        # 4. No config file
        return None

    @computed_field  # type: ignore
    @property
    def services(self) -> dict[str, Any]:
        """Get the effective services"""
        _services = defaults.DEFAULT_SERVICES.copy()

        if not self.configuration_file:
            return _services

        # Load the configuration file
        templated_config = utils.load_yaml(self.configuration_file)
        config = utils.apply_variables(templated_config, self.variables)

        for key, value in config.items():
            _services[key.replace("-", "_")] = value

        if self.execution_context == ExecutionContext.JOB:
            _services.pop("pipeline_executor", None)
        elif self.execution_context == ExecutionContext.PIPELINE:
            _services.pop("job_executor", None)
        else:
            raise ValueError(
                f"Invalid execution context: {self.execution_context}. Must be 'pipeline' or 'job'."
            )

        return _services


class RunnableContext(BaseModel):
    model_config = ConfigDict(use_enum_values=True, loc_by_alias=True)

    execution_mode: ExecutionMode = ExecutionMode.PYTHON

    parameters_file: Optional[str] = Field(
        default=None, exclude=True, description="Path to the parameters file."
    )
    configuration_file: Optional[str] = Field(
        default=None, exclude=True, description="Path to the configuration file."
    )
    variables: dict[str, str] = Field(
        default_factory=utils.gather_variables,
        exclude=True,
        description="Variables to be used.",
    )
    run_id: str = Field(
        description="The run ID, generated automatically if not provided"
    )
    tag: Optional[str] = Field(default=None, description="Tag to be used for the run.")

    # TODO: Verify the design
    object_serialisation: bool = (
        True  # Should be validated against executor being local
    )
    return_objects: Dict[
        str, Any
    ] = {}  # Should be validated against executor being local, should this be here?

    @field_validator("parameters_file", mode="before")
    @classmethod
    def override_parameters_file(cls, parameters_file: str) -> str:
        """Override the parameters file if provided."""
        if os.environ.get(defaults.RUNNABLE_PARAMETERS_FILE, None):
            return os.environ.get(defaults.RUNNABLE_PARAMETERS_FILE, parameters_file)
        return parameters_file

    @field_validator("configuration_file", mode="before")
    @classmethod
    def override_configuration_file(cls, configuration_file: str) -> str:
        """Override the configuration file if provided."""
        return os.environ.get(defaults.RUNNABLE_CONFIGURATION_FILE, configuration_file)

    @field_validator("run_id", mode="before")
    @classmethod
    def generate_run_id(cls, run_id: str) -> str:
        """Generate a run id if not provided."""
        # Convert None to empty string for consistency
        if run_id is None:
            run_id = ""

        # Check for retry run id first - this takes precedence
        retry_run_id = os.environ.get(defaults.RETRY_RUN_ID, "")
        if retry_run_id:
            return retry_run_id

        if not run_id:
            run_id = os.environ.get(defaults.ENV_RUN_ID, "")

        # If both are not given, generate one
        if not run_id:
            now = datetime.now()
            run_id = f"{names.get_random_name()}-{now.hour:02}{now.minute:02}"

        return run_id

    @computed_field  # type: ignore
    @property
    def retry_indicator(self) -> str:
        """Indicator for retry executions to distinguish attempt logs."""
        return os.environ.get(defaults.RETRY_INDICATOR, "")

    @computed_field  # type: ignore
    @property
    def is_retry(self) -> bool:
        """Flag indicating if this is a retry run based on environment variable."""
        retry_run_id = os.environ.get(defaults.RETRY_RUN_ID, "")
        return bool(retry_run_id)

    def model_post_init(self, __context: Any) -> None:
        os.environ[defaults.ENV_RUN_ID] = self.run_id

        if self.configuration_file:
            os.environ[defaults.RUNNABLE_CONFIGURATION_FILE] = self.configuration_file
        if self.tag:
            os.environ[defaults.RUNNABLE_RUN_TAG] = self.tag

        # Set the context using contextvars for proper isolation
        set_run_context(self)

    def execute(self):
        "Execute the pipeline or the job"
        raise NotImplementedError


class PipelineContext(RunnableContext):
    pipeline_executor: InstantiatedPipelineExecutor
    catalog: InstantiatedCatalog
    secrets: InstantiatedSecrets
    pickler: InstantiatedPickler
    run_log_store: InstantiatedRunLogStore

    pipeline_definition_file: str

    @computed_field  # type: ignore
    @cached_property
    def dag(self) -> Graph | None:
        """Get the dag."""
        if self.execution_mode == ExecutionMode.YAML:
            return get_pipeline_spec_from_yaml(self.pipeline_definition_file)
        elif self.execution_mode == ExecutionMode.PYTHON:
            return get_pipeline_spec_from_python(self.pipeline_definition_file)
        else:
            raise ValueError(
                f"Invalid execution mode: {self.execution_mode}. Must be 'yaml' or 'python'."
            )

    @computed_field  # type: ignore
    @cached_property
    def dag_hash(self) -> str:
        dag = self.dag
        if not dag:
            return ""
        dag_str = json.dumps(dag.model_dump(), sort_keys=True, ensure_ascii=True)
        return hashlib.sha1(dag_str.encode("utf-8")).hexdigest()

    def get_node_callable_command(
        self,
        node: BaseNode,
        iter_variable: defaults.IterableParameterModel | None = None,
        over_write_run_id: str = "",
        log_level: str = "",
    ) -> str:
        run_id = self.run_id

        if over_write_run_id:
            run_id = over_write_run_id

        log_level = log_level or logging.getLevelName(logger.getEffectiveLevel())

        action = (
            f"runnable execute-single-node {run_id} "
            f"{self.pipeline_definition_file} "
            f"{node._command_friendly_name()} "
            f"--log-level {log_level} "
        )

        # yaml is the default mode
        if self.execution_mode == ExecutionMode.PYTHON:
            action = action + "--mode python "

        if iter_variable:
            action = action + f"--iter-variable '{iter_variable.model_dump_json()}' "

        if self.configuration_file:
            action = action + f"--config {self.configuration_file} "

        if self.parameters_file:
            action = action + f"--parameters-file {self.parameters_file} "

        if self.tag:
            action = action + f"--tag {self.tag}"

        console.log(
            f"Generated command for node {node._command_friendly_name()}: {action}"
        )

        return action

    def get_fan_command(
        self,
        node: BaseNode,
        mode: str,
        run_id: str,
        iter_variable: defaults.IterableParameterModel | None = None,
        log_level: str = "",
    ) -> str:
        """
        Return the fan "in or out" command for this pipeline context.

        Args:
            node (BaseNode): The composite node that we are fanning in or out
            mode (str): "in" or "out"
            map_variable (dict, optional): If the node is a map, we have the map variable. Defaults to None.
            log_level (str, optional): Log level. Defaults to "".

        Returns:
            str: The fan in or out command
        """
        log_level = log_level or logging.getLevelName(logger.getEffectiveLevel())
        action = (
            f"runnable fan {run_id} "
            f"{node._command_friendly_name()} "
            f"{self.pipeline_definition_file} "
            f"{mode} "
            f"--log-level {log_level}"
        )
        if self.configuration_file:
            action += f" --config-file {self.configuration_file}"
        if self.parameters_file:
            action += f" --parameters-file {self.parameters_file}"
        if iter_variable:
            action += f" --iter-variable '{iter_variable.model_dump_json()}'"
        if self.execution_mode == ExecutionMode.PYTHON:
            action += " --mode python"
        if self.tag:
            action += f" --tag {self.tag}"

        console.log(
            f"Generated command for fan {mode} for node {node._command_friendly_name()}: {action}"
        )
        return action

    def execute(self):
        assert self.dag is not None

        pipeline_name = getattr(self.dag, "name", "unnamed")

        with logfire.span(
            "pipeline:{pipeline_name}",
            pipeline_name=pipeline_name,
            run_id=self.run_id,
            executor=self.pipeline_executor.__class__.__name__,
        ):
            logfire.info("Pipeline execution started")

            console.print("Working with context:")
            console.print(get_run_context())
            console.rule(style="[dark orange]")

            # Prepare for graph execution
            if self.pipeline_executor._should_setup_run_log_at_traversal:
                self.pipeline_executor._set_up_run_log(exists_ok=False)

            try:
                self.pipeline_executor.execute_graph(dag=self.dag)
                if not self.pipeline_executor._should_setup_run_log_at_traversal:
                    # non local executors just traverse the graph and do nothing
                    logfire.info("Pipeline submitted", status="submitted")
                    return {}

                ctx = get_run_context()
                assert ctx
                assert isinstance(ctx, PipelineContext)
                run_log = ctx.run_log_store.get_run_log_by_id(
                    run_id=ctx.run_id, full=False
                )

                if run_log.status == defaults.SUCCESS:
                    console.print(
                        "Pipeline executed successfully!", style=defaults.success_style
                    )
                    logfire.info("Pipeline completed", status="success")
                else:
                    console.print(
                        "Pipeline execution failed.", style=defaults.error_style
                    )
                    logfire.error("Pipeline failed", status="failed")
                    raise exceptions.ExecutionFailedError(ctx.run_id)
            except Exception as e:  # noqa: E722
                console.print(e, style=defaults.error_style)
                logfire.error("Pipeline failed with exception", error=str(e)[:256])
                raise

            if self.pipeline_executor._should_setup_run_log_at_traversal:
                ctx = get_run_context()
                assert ctx
                assert isinstance(ctx, PipelineContext)
                return ctx.run_log_store.get_run_log_by_id(run_id=ctx.run_id)

    def _handle_completion(self):
        """Handle post-execution - shared by sync/async."""
        ctx = get_run_context()
        assert ctx
        assert isinstance(ctx, PipelineContext)
        run_log = ctx.run_log_store.get_run_log_by_id(run_id=ctx.run_id, full=False)

        if run_log.status == defaults.SUCCESS:
            console.print(
                "Pipeline executed successfully!", style=defaults.success_style
            )
            logfire.info("Pipeline completed", status="success")
        else:
            console.print("Pipeline execution failed.", style=defaults.error_style)
            logfire.error("Pipeline failed", status="failed")
            raise exceptions.ExecutionFailedError(ctx.run_id)

    async def execute_async(self):
        """Async pipeline execution."""
        assert self.dag is not None

        pipeline_name = getattr(self.dag, "name", "unnamed")

        with logfire.span(
            "pipeline:{pipeline_name}",
            pipeline_name=pipeline_name,
            run_id=self.run_id,
            executor=self.pipeline_executor.__class__.__name__,
        ):
            logfire.info("Async pipeline execution started")

            console.print("Working with context:")
            console.print(get_run_context())
            console.rule(style="[dark orange]")

            if self.pipeline_executor._should_setup_run_log_at_traversal:
                self.pipeline_executor._set_up_run_log(exists_ok=False)

            try:
                await self.pipeline_executor.execute_graph_async(dag=self.dag)
                self._handle_completion()

            except Exception as e:
                console.print(e, style=defaults.error_style)
                logfire.error("Pipeline failed with exception", error=str(e)[:256])
                raise

            if self.pipeline_executor._should_setup_run_log_at_traversal:
                ctx = get_run_context()
                assert ctx
                assert isinstance(ctx, PipelineContext)
                return ctx.run_log_store.get_run_log_by_id(run_id=ctx.run_id)


class AsyncPipelineContext(RunnableContext):
    """
    Simplified context for async pipeline execution.

    Unlike PipelineContext, this accepts the DAG directly rather than
    introspecting from a file. This simplifies async execution since
    we only support local executors for async pipelines.
    """

    pipeline_executor: InstantiatedPipelineExecutor
    catalog: InstantiatedCatalog
    secrets: InstantiatedSecrets
    pickler: InstantiatedPickler
    run_log_store: InstantiatedRunLogStore

    # DAG is passed directly, not computed from a file
    dag: Graph

    @computed_field  # type: ignore
    @cached_property
    def dag_hash(self) -> str:
        dag = self.dag
        if not dag:
            return ""
        dag_str = json.dumps(dag.model_dump(), sort_keys=True, ensure_ascii=True)
        return hashlib.sha1(dag_str.encode("utf-8")).hexdigest()

    async def execute_async(self):
        """Async pipeline execution."""
        assert self.dag is not None

        pipeline_name = getattr(self.dag, "name", "unnamed")

        with logfire.span(
            "pipeline:{pipeline_name}",
            pipeline_name=pipeline_name,
            run_id=self.run_id,
            executor=self.pipeline_executor.__class__.__name__,
        ):
            logfire.info("Async pipeline execution started")

            console.print("Working with context:")
            console.print(get_run_context())
            console.rule(style="[dark orange]")

            if self.pipeline_executor._should_setup_run_log_at_traversal:
                self.pipeline_executor._set_up_run_log(exists_ok=False)

            try:
                await self.pipeline_executor.execute_graph_async(dag=self.dag)

                run_log = self.run_log_store.get_run_log_by_id(
                    run_id=self.run_id, full=False
                )

                if run_log.status == defaults.SUCCESS:
                    console.print(
                        "Pipeline executed successfully!", style=defaults.success_style
                    )
                    logfire.info("Pipeline completed", status="success")
                else:
                    console.print(
                        "Pipeline execution failed.", style=defaults.error_style
                    )
                    logfire.error("Pipeline failed", status="failed")
                    raise exceptions.ExecutionFailedError(self.run_id)

            except Exception as e:
                console.print(e, style=defaults.error_style)
                logfire.error("Pipeline failed with exception", error=str(e)[:256])
                raise

            if self.pipeline_executor._should_setup_run_log_at_traversal:
                return self.run_log_store.get_run_log_by_id(run_id=self.run_id)


class JobContext(RunnableContext):
    job_executor: InstantiatedJobExecutor
    catalog: InstantiatedCatalog
    secrets: InstantiatedSecrets
    pickler: InstantiatedPickler
    run_log_store: InstantiatedRunLogStore

    job_definition_file: str
    catalog_store_copy: bool = Field(default=True, alias="catalog_store_copy")

    @computed_field  # type: ignore
    @cached_property
    def job(self) -> BaseTaskType:
        job, _ = get_job_spec_from_python(self.job_definition_file)
        return job

    @computed_field  # type: ignore
    @cached_property
    def catalog_settings(self) -> list[str] | None:
        _, catalog_config = get_job_spec_from_python(self.job_definition_file)
        return catalog_config

    def get_job_callable_command(
        self,
        over_write_run_id: str = "",
    ):
        run_id = self.run_id

        if over_write_run_id:
            run_id = over_write_run_id

        log_level = logging.getLevelName(logger.getEffectiveLevel())

        action = (
            f"runnable execute-job {self.job_definition_file} {run_id} "
            f" --log-level {log_level}"
        )

        if self.configuration_file:
            action = action + f" --config {self.configuration_file}"

        if self.parameters_file:
            action = action + f" --parameters {self.parameters_file}"

        if self.tag:
            action = action + f" --tag {self.tag}"

        return action

    def execute(self):
        with logfire.span(
            "job:{job_name}",
            job_name=self.job_definition_file,
            run_id=self.run_id,
            executor=self.job_executor.__class__.__name__,
        ):
            logfire.info("Job execution started")

            console.print("Working with context:")
            console.print(get_run_context())
            console.rule(style="[dark orange]")

            try:
                self.job_executor.submit_job(
                    job=self.job, catalog_settings=self.catalog_settings
                )
                logfire.info("Job submitted", status="submitted")
            except Exception as e:
                logfire.error("Job failed", error=str(e)[:256])
                raise
            finally:
                console.print(f"Job execution completed for run id: {self.run_id}")

            logger.info(
                "Executing the job from the user. We are still in the caller's compute"
                " environment"
            )

            if self.job_executor._should_setup_run_log_at_traversal:
                ctx = get_run_context()
                assert ctx
                assert isinstance(ctx, JobContext)
                return ctx.run_log_store.get_run_log_by_id(run_id=ctx.run_id)


# Context variable for thread/async-safe run context storage
if TYPE_CHECKING:
    from typing import Union

    RunnableContextType = Union[
        "RunnableContext", "PipelineContext", "JobContext", "AsyncPipelineContext"
    ]
else:
    RunnableContextType = Any

_run_context_var: contextvars.ContextVar[
    Optional["PipelineContext | JobContext | AsyncPipelineContext | RunnableContext"]
] = contextvars.ContextVar("run_context", default=None)


def get_run_context() -> (
    Optional["PipelineContext | JobContext | AsyncPipelineContext | RunnableContext"]
):
    """Get the current run context for this execution context."""
    return _run_context_var.get()


def set_run_context(
    context: Optional[
        "PipelineContext | JobContext | AsyncPipelineContext | RunnableContext"
    ],
) -> None:
    """Set the run context for this execution context."""
    _run_context_var.set(context)


# BREAKING CHANGE: The global run_context variable has been replaced with
# get_run_context() and set_run_context() functions for proper context isolation.
# All code must be updated to use the new API.
#
# Migration guide:
#   Before: run_context.run_log_store
#   After:  get_run_context().run_log_store
#
# This change was necessary to fix concurrency issues by using contextvars
# for proper thread and async isolation of run contexts.
