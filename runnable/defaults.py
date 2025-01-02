from typing import (
    Any,
    Dict,
    Mapping,
    Optional,
    TypedDict,  # type: ignore[unused-ignore]
    Union,
)

from rich.style import Style
from typing_extensions import TypeAlias

NAME = "runnable"
LOGGER_NAME = "runnable"

# CLI settings
LOG_LEVEL = "WARNING"


# Type definitions
class ServiceConfig(TypedDict):
    type: str
    config: Mapping[str, Any]


class RunnableConfig(TypedDict, total=False):
    run_log_store: Optional[ServiceConfig]
    secrets: Optional[ServiceConfig]
    catalog: Optional[ServiceConfig]
    pipeline_executor: Optional[ServiceConfig]
    pickler: Optional[ServiceConfig]


TypeMapVariable: TypeAlias = Optional[Dict[str, Union[str, int, float]]]


# Config file environment variable
RUNNABLE_CONFIG_FILE = "RUNNABLE_CONFIG_FILE"
RUNNABLE_RUN_TAG = "RUNNABLE_RUN_TAG"

# Interaction settings
TRACK_PREFIX = "RUNNABLE_TRACK_"
PARAMETER_PREFIX = "RUNNABLE_PRM_"
MAP_VARIABLE = "RUNNABLE_MAP_VARIABLE"
VARIABLE_PREFIX = "RUNNABLE_VAR_"
ENV_RUN_ID = "RUNNABLE_RUN_ID"
ATTEMPT_NUMBER = "RUNNABLE_STEP_ATTEMPT"

## Generated pipeline file
GENERATED_PIPELINE_FILE = "generated_pipeline.yaml"

# STATUS progression
# For Branch, CREATED -> PROCESSING -> SUCCESS OR FAIL
# For a step, CREATED -> TRIGGERED ->  PROCESSING -> SUCCESS OR FAIL
CREATED = "CREATED"
PROCESSING = "PROCESSING"
SUCCESS = "SUCCESS"
FAIL = "FAIL"
TRIGGERED = "TRIGGERED"

# Node and Command settings
COMMAND_TYPE = "python"
COMMAND_FRIENDLY_CHARACTER = "%"

# Default services
DEFAULT_PIPELINE_EXECUTOR = ServiceConfig(type="local", config={})
DEFAULT_JOB_EXECUTOR = ServiceConfig(type="local", config={})
DEFAULT_RUN_LOG_STORE = ServiceConfig(type="file-system", config={})
DEFAULT_CATALOG = ServiceConfig(type="file-system", config={})
DEFAULT_SECRETS = ServiceConfig(type="env-secrets", config={})
DEFAULT_PICKLER = ServiceConfig(type="pickle", config={})

# Map state
MAP_PLACEHOLDER = "map_variable_placeholder"

# Dag node
DAG_BRANCH_NAME = "dag"

# RUN settings
RANDOM_RUN_ID_LEN = 6
MAX_TIME = 86400  # 1 day in seconds

# User extensions
USER_CONFIG_FILE = "runnable-config.yaml"

# Executor settings
ENABLE_PARALLEL = False

# RUN log store settings
LOG_LOCATION_FOLDER = ".run_log_store"

# Dag node
DAG_BRANCH_NAME = "dag"

# Data catalog settings
CATALOG_LOCATION_FOLDER = ".catalog"
COMPUTE_DATA_FOLDER = "."

# Secrets settings
DOTENV_FILE_LOCATION = ".env"

LEN_SHA_FOR_TAG = 8

# JOB CONFIG
DEFAULT_JOB_NAME = "job"

## Logging settings

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": True,
    "formatters": {
        "standard": {"format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"},
        "runnable_formatter": {"format": "%(message)s", "datefmt": "[%X]"},
    },
    "handlers": {
        "default": {
            "formatter": "standard",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",  # Default is stderr
        },
        "runnable_handler": {
            "formatter": "runnable_formatter",
            "class": "rich.logging.RichHandler",
            "rich_tracebacks": True,
        },
    },
    "loggers": {
        "": {
            "handlers": ["default"],
            "propagate": True,
        },  # Root logger
        LOGGER_NAME: {"handlers": ["runnable_handler"], "propagate": False},
    },
}


# styles
error_style = Style(color="red", bold=True)
warning_style = Style(color="yellow", bold=True)
success_style = Style(color="green", bold=True)
info_style = Style(color="blue", bold=True)
