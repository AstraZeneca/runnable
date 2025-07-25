# ruff: noqa

import os

from rich.console import Console

console = Console(record=True)
console.print(":runner: Lets go!!")

task_console = Console(record=True)

from runnable.sdk import (  # noqa;
    Catalog,
    Conditional,
    Fail,
    Map,
    NotebookJob,
    NotebookTask,
    Parallel,
    Pipeline,
    PythonJob,
    PythonTask,
    ShellJob,
    ShellTask,
    Stub,
    Success,
    metric,
    pickled,
)

# Needed to disable ploomber telemetry
os.environ["PLOOMBER_STATS_ENABLED"] = "false"
