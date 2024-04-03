# ruff: noqa

# TODO: Might need to add Rich to pyinstaller part
import logging
from logging.config import dictConfig

from runnable import defaults

dictConfig(defaults.LOGGING_CONFIG)
logger = logging.getLogger(defaults.LOGGER_NAME)

from runnable.sdk import (  # noqa
    Catalog,
    Fail,
    Map,
    NotebookTask,
    Parallel,
    Pipeline,
    PythonTask,
    ShellTask,
    Stub,
    Success,
    pickled,
)

# TODO: Think of model registry as a central place to store models.
# TODO: Implement Sagemaker pipelines as a executor.


# TODO: Think of way of generating dag hash without executor configuration
