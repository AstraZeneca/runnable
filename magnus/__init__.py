# ruff: noqa

# TODO: Might need to add Rich to pyinstaller part

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
from magnus.sdk import Stub, Pipeline, Task, Parallel  # noqa


# TODO: Think of model registry as a central place to store models.
# TODO: Implement Sagemaker pipelines as a executor.


# TODO: Bring this in the mix: https://github.com/mkdocs/mkdocs-click

# TODO: Think of way of generating dag hash without executor configuration
