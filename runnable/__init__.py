# ruff: noqa

# TODO: Might need to add Rich to pyinstaller part
import logging
from logging.config import dictConfig

from rich.console import Console

from runnable import defaults

dictConfig(defaults.LOGGING_CONFIG)
logger = logging.getLogger(defaults.LOGGER_NAME)

console = Console()
console.print(":runner: Lets go!!")

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
    metric,
    pickled,
)

# TODO: Think of model registry as a central place to store models.
# TODO: Implement Sagemaker pipelines as a executor.


# TODO: Think of way of generating dag hash without executor configuration
