from typing import Any, Dict, Optional, OrderedDict, Union

from pydantic import BaseModel, Field, field_validator
from rich.style import Style
from typing_extensions import TypeAlias

NAME = "runnable"
LOGGER_NAME = "runnable"

# CLI settings
LOG_LEVEL = "WARNING"


MapVariableType: TypeAlias = Optional[Dict[str, Union[str, int, float]]]

# Config file environment variable
RUNNABLE_CONFIGURATION_FILE = "RUNNABLE_CONFIGURATION_FILE"
RUNNABLE_RUN_TAG = "RUNNABLE_RUN_TAG"
RUNNABLE_PARAMETERS_FILE = "RUNNABLE_PARAMETERS_FILE"

# Interaction settings
PARAMETER_PREFIX = "RUNNABLE_PRM_"
MAP_VARIABLE = "RUNNABLE_MAP_VARIABLE"
VARIABLE_PREFIX = "RUNNABLE_VAR_"
ENV_RUN_ID = "RUNNABLE_RUN_ID"
RETRY_RUN_ID = "RUNNABLE_RETRY_RUN_ID"
RETRY_INDICATOR = "RUNNABLE_RETRY_INDICATOR"
ATTEMPT_NUMBER = "RUNNABLE_STEP_ATTEMPT"


class MapVariableModel(BaseModel):
    value: Any

    @field_validator("value")
    @classmethod
    def validate_json_serializable(cls, v):
        """Ensure the value is JSON serializable"""
        import json

        try:
            json.dumps(v)
            return v
        except (TypeError, ValueError) as e:
            raise ValueError(f"Value must be JSON serializable: {e}") from e


class LoopIndexModel(BaseModel):
    value: int


class IterableParameterModel(BaseModel):
    # {i1: {value: v1}, i2: {value: v2}} where i1 is outside map and i2 is nested map
    map_variable: OrderedDict[str, MapVariableModel] | None = Field(
        default_factory=OrderedDict
    )
    # [ {value: v1}, {value: v2} ] for  index based iteration,
    # i1 is outside loop and i2 is nested loop
    loop_variable: list[LoopIndexModel] | None = Field(default_factory=lambda: [])


## Generated pipeline file
GENERATED_PIPELINE_FILE = "generated_pipeline.yaml"

# STATUS progression
# For Branch, CREATED -> PROCESSING -> SUCCESS OR FAIL
# For a step, CREATED -> TRIGGERED ->  PROCESSING -> SUCCESS OR FAIL
CREATED = "CREATED"
PROCESSING = "PROCESSING"
SUCCESS = "SUCCESS"
FAIL = "FAIL"

# Node and Command settings
COMMAND_TYPE = "python"
COMMAND_FRIENDLY_CHARACTER = "%"

# Default services
DEFAULT_SERVICES: dict[str, Any] = {
    "pipeline_executor": {"type": "local", "config": {}},
    "job_executor": {"type": "local", "config": {}},
    "run_log_store": {"type": "file-system", "config": {}},
    "catalog": {"type": "file-system", "config": {}},
    "pickler": {"type": "pickle", "config": {}},
    "secrets": {"type": "env-secrets", "config": {}},
}

# Map state
MAP_PLACEHOLDER = "map_variable_placeholder"
LOOP_PLACEHOLDER = "loop_variable_placeholder"

# Dag node
DAG_BRANCH_NAME = "dag"

# RUN settings
RANDOM_RUN_ID_LEN = 6
MAX_TIME = 86400  # 1 day in seconds

# User extensions
USER_CONFIG_FILE = "runnable-config.yaml"


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

# Hash computation settings
HASH_ALGORITHM = "sha256"  # More secure and faster than MD5
LARGE_FILE_THRESHOLD_BYTES = 1024 * 1024 * 1024  # 1GB
HASH_CHUNK_SIZE = 1024 * 1024  # 1MB chunks for fingerprint hashing
