# ruff: noqa

import logging
from rich.logging import RichHandler

# logging.basicConfig(level="NOTSET", format="%(message)s", datefmt="[%X]", handlers=[RichHandler(rich_tracebacks=True)])

magnus_logger = logging.getLogger("magnus")
magnus_logger.setLevel(logging.NOTSET)

# handler
rich_handler = RichHandler(rich_tracebacks=True)

# formatter
formatter = logging.Formatter("%(message)s", datefmt="[%X]")

rich_handler.setFormatter(formatter)
magnus_logger.addHandler(rich_handler)


from magnus.interaction import (
    end_interactive_session,
    get_experiment_tracker_context,
    get_from_catalog,
    get_object,
    get_parameter,
    get_run_id,
    get_secret,
    put_in_catalog,
    put_object,
    start_interactive_session,
    store_parameter,
    track_this,
)  # noqa
from magnus.sdk import AsIs, Pipeline, Task  # noqa

# TODO: Write cleaner and better examples to ship the code.
