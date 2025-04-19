import importlib
import json
import logging
import os
import sys
from datetime import datetime
from enum import Enum
from functools import cached_property, partial
from typing import Annotated, Any, Callable, Dict, List, Literal, Optional

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
)
from rich.progress import Progress
from stevedore import driver

from runnable import defaults, names, utils
from runnable.catalog import BaseCatalog
from runnable.datastore import BaseRunLogStore
from runnable.executor import BaseJobExecutor, BasePipelineExecutor
from runnable.graph import Graph, create_graph
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


def get_service_by_name(namespace: str, service_config: dict[str, Any], _) -> Any:  # noqa: ANN401, ANN001
    """Get the service by name."""
    service_config = service_config.copy()

    kind = service_config.pop("type", None)

    # if not kind:
    #     kind = defaults.DEFAULT_SERVICES[kind]

    if "config" in service_config:
        service_config = service_config.get("config", {})

    logger.debug(
        f"Trying to get a service of {namespace} with config: {service_config}"
    )
    try:
        mgr = driver.DriverManager(
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


def get_task(service: str) -> Callable:
    """Get the task by name.

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
InstantiatedJob = Annotated[BaseTaskType, BeforeValidator(get_task("tasks"))]


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

    @field_validator("configuration_file", mode="before")
    @classmethod
    def override_configuration_file(cls, configuration_file: str) -> str | None:
        """Override the configuration file if provided."""
        if os.environ.get(defaults.RUNNABLE_CONFIGURATION_FILE, None):
            # If the env var is set, use it
            return os.environ.get(
                defaults.RUNNABLE_CONFIGURATION_FILE, configuration_file
            )

    @computed_field
    @property
    def services(self) -> dict[str, Any]:
        """Get the effective services"""
        # TODO: Take care of pipeline vs job here
        services = defaults.DEFAULT_SERVICES

        if not self.configuration_file:
            return services

        # Load the configuration file
        config = utils.load_yaml(self.configuration_file)
        for key, value in config.items():
            services[key.replace("-", "_")] = value

        if self.execution_context == ExecutionContext.JOB:
            services.pop("pipeline_executor", None)
        elif self.execution_context == ExecutionContext.PIPELINE:
            services.pop("job_executor", None)
        else:
            raise ValueError(
                f"Invalid execution context: {self.execution_context}. Must be 'pipeline' or 'job'."
            )

        return services


class RunnableContext(BaseModel):
    model_config = ConfigDict(use_enum_values=True, loc_by_alias=True)

    execution_mode: ExecutionMode = ExecutionMode.PYTHON

    parameters_file: Optional[str] = Field(
        default=None, exclude=True, description="Path to the parameters file."
    )
    configuration_file: Optional[str] = Field(
        default=None, exclude=True, description="Path to the configuration file."
    )
    run_id: str = ""  # Should be annotated to generate one if not provided
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
        if os.environ.get(defaults.RUNNABLE_CONFIGURATION_FILE, None):
            # If the env var is set, use it
            return os.environ.get(
                defaults.RUNNABLE_CONFIGURATION_FILE, configuration_file
            )
        return os.environ.get(defaults.RUNNABLE_CONFIGURATION_FILE, configuration_file)

    @field_validator("run_id", mode="before")
    @classmethod
    def generate_run_id(cls, run_id: str) -> str:
        """Generate a run id if not provided."""
        if not run_id:
            run_id = os.environ.get(defaults.ENV_RUN_ID, "")

        # If both are not given, generate one
        if not run_id:
            now = datetime.now()
            run_id = f"{names.get_random_name()}-{now.hour:02}{now.minute:02}"

        return run_id

    @computed_field
    @cached_property
    def variables(self) -> Dict[str, str]:
        """Get the variables."""
        variables = {}

        for env_var, value in os.environ.items():
            if env_var.startswith(defaults.VARIABLE_PREFIX):
                key = env_var.replace(defaults.VARIABLE_PREFIX, "", 1)
                variables[key] = value

        return variables

    def model_post_init(self, __context: Any) -> None:
        os.environ[defaults.ENV_RUN_ID] = self.run_id

        if self.configuration_file:
            os.environ[defaults.RUNNABLE_CONFIGURATION_FILE] = self.configuration_file
        if self.tag:
            os.environ[defaults.RUNNABLE_RUN_TAG] = self.tag


# THe structure of the config file as presented by the user.
# There are default services that are used if the user does not provide any.
class PipelineContext(RunnableContext):
    pipeline_executor: InstantiatedPipelineExecutor = Field(alias="pipeline_executor")
    catalog: InstantiatedCatalog
    secrets: InstantiatedSecrets
    pickler: InstantiatedPickler
    run_log_store: InstantiatedRunLogStore = Field(alias="run_log_store")

    pipeline_definition_file: str

    @computed_field
    def from_sdk(self) -> bool:
        """Check if the pipeline/job is from SDK."""
        if self.execution_mode == ExecutionMode.PYTHON:
            return True
        return False

    @computed_field
    @cached_property
    def dag(self) -> Graph | None:
        """Get the dag."""
        if self.execution_mode == ExecutionMode.YAML:
            return get_pipeline_spec_from_yaml(self.pipeline_definition_file)
        elif self.execution_mode == ExecutionMode.PYTHON:
            # If its local execution, we don't need to call the function to get dag
            if self.pipeline_executor._is_local:
                return None
            return get_pipeline_spec_from_python(self.pipeline_definition_file)
        else:
            raise ValueError(
                f"Invalid execution mode: {self.execution_mode}. Must be 'yaml' or 'python'."
            )

    @computed_field
    @cached_property
    def dag_hash(self) -> str:
        dag = self.dag
        if not dag:
            return ""
        return utils.get_dag_hash(dag.model_dump(exclude_none=True))


class JobContext(RunnableContext):
    kind: Literal["job"]

    job_executor: InstantiatedJobExecutor
    catalog: InstantiatedCatalog
    secrets: InstantiatedSecrets
    pickler: InstantiatedPickler
    run_log_store: InstantiatedRunLogStore

    job_definition_file: str
    job: InstantiatedJob
    job_catalog_settings: Optional[List[str]] = Field(default=None)
    # TODO: the typing may be wrong here

    @computed_field
    @property
    def from_sdk(self) -> bool:
        """Check if the pipeline/job is from SDK."""
        if self.job_definition_file.endswith(".py"):
            return True
        return False


# class Context(BaseModel):
#     executor: SerializeAsAny[BaseExecutor]
#     run_log_store: SerializeAsAny[BaseRunLogStore]
#     secrets_handler: SerializeAsAny[BaseSecrets]
#     catalog_handler: SerializeAsAny[BaseCatalog]
#     pickler: SerializeAsAny[BasePickler]
#     progress: SerializeAsAny[Optional[Progress]] = Field(default=None, exclude=True)

#     model_config = ConfigDict(arbitrary_types_allowed=True)

#     pipeline_file: Optional[str] = ""
#     job_definition_file: Optional[str] = ""
#     parameters_file: Optional[str] = ""
#     configuration_file: Optional[str] = ""
#     from_sdk: bool = False

#     run_id: str = ""
#     object_serialisation: bool = True
#     return_objects: Dict[str, Any] = {}

#     tag: str = ""
#     variables: Dict[str, str] = {}

#     dag: Optional[Graph] = None
#     dag_hash: str = ""

#     job: Optional[BaseTaskType] = None
#     job_catalog_settings: Optional[List[str]] = []


run_context: PipelineContext | JobContext = None  # type: ignore
progress: Progress = None  # type: ignore
