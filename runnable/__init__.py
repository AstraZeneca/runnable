# ruff: noqa

# TODO: Might need to add Rich to pyinstaller part
import logging
import os
from logging.config import dictConfig

from rich.console import Console

from runnable import defaults

dictConfig(defaults.LOGGING_CONFIG)
logger = logging.getLogger(defaults.LOGGER_NAME)

console = Console(record=True)
console.print(":runner: Lets go!!")

task_console = Console(record=True)

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

# Needed to disable ploomber telemetry
os.environ["PLOOMBER_STATS_ENABLED"] = "false"
