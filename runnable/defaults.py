# mypy: ignore-errors
# The above should be done until https://github.com/python/mypy/issues/8823
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Union

from typing_extensions import TypeAlias

# TODO: This is not the correct way to do this.
try:  # pragma: no cover
    from typing import TypedDict  # type: ignore[unused-ignore]
except ImportError:  # pragma: no cover
    from typing_extensions import TypedDict  # type: ignore[unused-ignore]


NAME = "runnable"
LOGGER_NAME = "runnable"

# CLI settings
LOG_LEVEL = "WARNING"


class EXECUTION_PLAN(Enum):
    """
    The possible execution plans for a runnable job.
    """

    CHAINED = "chained"  #  121 relationship between run log and the dag.
    UNCHAINED = "unchained"  # Only captures execution of steps, no relation.
    INTERACTIVE = "interactive"  # used for interactive sessions


# Type definitions
class ServiceConfig(TypedDict):
    type: str
    config: Mapping[str, Any]


class RunnableConfig(TypedDict, total=False):
    run_log_store: Optional[ServiceConfig]
    secrets: Optional[ServiceConfig]
    catalog: Optional[ServiceConfig]
    executor: Optional[ServiceConfig]
    experiment_tracker: Optional[ServiceConfig]
    pickler: Optional[ServiceConfig]


TypeMapVariable: TypeAlias = Optional[Dict[str, Union[str, int, float]]]


# Config file environment variable
RUNNABLE_CONFIG_FILE = "RUNNABLE_CONFIG_FILE"
RUNNABLE_RUN_TAG = "RUNNABLE_RUN_TAG"

# Interaction settings
TRACK_PREFIX = "RUNNABLE_TRACK_"
STEP_INDICATOR = "_STEP_"
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
NODE_SPEC_FILE = "node_spec.yaml"
COMMAND_FRIENDLY_CHARACTER = "%"
DEFAULT_CONTAINER_CONTEXT_PATH = "/opt/runnable/"
DEFAULT_CONTAINER_DATA_PATH = "data/"
DEFAULT_CONTAINER_OUTPUT_PARAMETERS = "parameters.json"

# Default services
DEFAULT_EXECUTOR = ServiceConfig(type="local", config={})
DEFAULT_RUN_LOG_STORE = ServiceConfig(type="file-system", config={})
DEFAULT_CATALOG = ServiceConfig(type="file-system", config={})
DEFAULT_SECRETS = ServiceConfig(type="do-nothing", config={})
DEFAULT_EXPERIMENT_TRACKER = ServiceConfig(type="do-nothing", config={})
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


# Docker settings
DOCKERFILE_NAME = "Dockerfile"
DOCKERFILE_CONTENT = r"""# Python 3.8 Image without Dependecies
FROM python:3.8

LABEL maintainer="mesanthu@gmail.com"

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

${INSTALL_STYLE}

ENV VIRTUAL_ENV=/opt/venv
RUN python -m virtualenv --python=/usr/local/bin/python $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

${COPY_CONTENT}
WORKDIR /app

${INSTALL_REQUIREMENTS}
"""
GIT_ARCHIVE_NAME = "git_tracked"
LEN_SHA_FOR_TAG = 8


class ENTRYPOINT(Enum):
    """
    The possible container entrypoint types.
    """

    USER = "user"
    SYSTEM = "system"


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
