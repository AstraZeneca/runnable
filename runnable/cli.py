import logging
from enum import Enum
from typing import Annotated

import typer

from runnable import defaults, entrypoints

logger = logging.getLogger(defaults.LOGGER_NAME)


app = typer.Typer(
    help=(
        "Welcome to runnable. Please provide the command that you want to use."
        "All commands have options that you can see by runnable <command> --help"
    ),
)


class LogLevel(str, Enum):
    INFO = "INFO"
    DEBUG = "DEBUG"
    WARNING = "WARNING"
    ERROR = "ERROR"
    FATAL = "FATAL"


class ExecutionMode(str, Enum):
    YAML = "yaml"
    PYTHON = "python"


class FanMode(str, Enum):
    IN = "in"
    OUT = "out"


@app.command()
def execute(
    yaml_file: Annotated[str, typer.Argument(help="The pipeline definition file")],
    config_file: Annotated[
        str,
        typer.Option(
            "--config", "-c", help="The configuration file specifying the services"
        ),
    ] = "",
    parameters_file: Annotated[
        str,
        typer.Option(
            "--parameters",
            "-p",
            help="Parameters, in yaml,  accessible by the application",
        ),
    ] = "",
    log_level: Annotated[
        LogLevel,
        typer.Option(
            "--log-level",
            help="The log level",
            show_default=True,
            case_sensitive=False,
        ),
    ] = LogLevel.WARNING,
    tag: Annotated[str, typer.Option(help="A tag attached to the run")] = "",
    run_id: Annotated[
        str,
        typer.Argument(
            envvar=defaults.ENV_RUN_ID,
            help="An optional run_id, one would be generated if its not provided",
        ),
    ] = "",
):
    """
    Execute a pipeline defined by yaml file.

    The executor is defined by executor block of the configuration file.

    The behavior of this command depends on the executor type:

    -- For local executors (local, local-container), the pipeline is executed in the current environment.

    -- For remote executors (argo, airflow), the pipeline translated to the specification.
    """
    logger.setLevel(log_level.value)

    entrypoints.execute_pipeline_yaml_spec(
        configuration_file=config_file,
        pipeline_file=yaml_file,
        tag=tag,
        run_id=run_id,
        parameters_file=parameters_file,
    )


@app.command(hidden=True)
def execute_single_node(
    run_id: Annotated[
        str,
        typer.Argument(
            help="An optional run_id, one would be generated if its not provided"
        ),
    ],
    yaml_or_python_file: Annotated[
        str, typer.Argument(help="The pipeline definition file")
    ],
    step_name: Annotated[str, typer.Argument(help="The step name to execute")],
    config_file: Annotated[
        str,
        typer.Option(
            "--config", "-c", help="The configuration file specifying the services"
        ),
    ] = "",
    parameters_file: Annotated[
        str,
        typer.Option(
            "--parameters-file",
            "-p",
            help="Parameters, in yaml,  accessible by the application",
        ),
    ] = "",
    log_level: Annotated[
        LogLevel,
        typer.Option(
            "--log-level",
            help="The log level",
            show_default=True,
            case_sensitive=False,
        ),
    ] = LogLevel.INFO,
    tag: Annotated[str, typer.Option(help="A tag attached to the run")] = "",
    mode: Annotated[
        ExecutionMode,
        typer.Option(
            "--mode",
            "-m",
            help="spec in yaml or python sdk",
        ),
    ] = ExecutionMode.YAML,
    map_variable: Annotated[
        str,
        typer.Option(
            "--map-variable",
            help="The map variable dictionary in str",
            show_default=True,
        ),
    ] = "",
):
    logger.setLevel(log_level.value)
    entrypoints.execute_single_node(
        configuration_file=config_file,
        pipeline_file=yaml_or_python_file,
        mode=mode,
        step_name=step_name,
        map_variable=map_variable,
        run_id=run_id,
        tag=tag,
        parameters_file=parameters_file,
    )


@app.command(hidden=True)
def fan(
    run_id: Annotated[str, typer.Argument(help="The run id")],
    step_name: Annotated[str, typer.Argument(help="The step name")],
    python_or_yaml_file: Annotated[
        str, typer.Argument(help="The pipeline definition file")
    ],
    in_or_out: Annotated[str, typer.Argument(help="The fan mode")],
    map_variable: Annotated[
        str,
        typer.Option(
            "--map-variable",
            help="The map variable dictionary in str",
            show_default=True,
        ),
    ] = "",
    config_file: Annotated[
        str,
        typer.Option(
            "--config-file", "-c", help="The configuration file specifying the services"
        ),
    ] = "",
    parameters_file: Annotated[
        str,
        typer.Option(
            "--parameters-file",
            "-p",
            help="Parameters, in yaml,  accessible by the application",
        ),
    ] = "",
    log_level: Annotated[
        LogLevel,
        typer.Option(
            "--log-level",
            help="The log level",
            show_default=True,
            case_sensitive=False,
        ),
    ] = LogLevel.INFO,
    tag: Annotated[str, typer.Option(help="A tag attached to the run")] = "",
    mode: Annotated[
        ExecutionMode,
        typer.Option(
            "--mode",
            "-m",
            help="spec in yaml or python sdk",
        ),
    ] = ExecutionMode.YAML,
):
    logger.setLevel(log_level.value)

    # Fan in or out
    entrypoints.fan(
        configuration_file=config_file,
        pipeline_file=python_or_yaml_file,
        step_name=step_name,
        mode=mode,
        in_or_out=in_or_out,
        map_variable=map_variable,
        run_id=run_id,
        tag=tag,
        parameters_file=parameters_file,
    )


@app.command()
def submit_job(
    job_definition_file: Annotated[
        str,
        typer.Argument(
            help=("The yaml file containing the job definition"),
        ),
    ],
    config_file: Annotated[
        str,
        typer.Option(
            "--config", "-c", help="The configuration file specifying the services"
        ),
    ] = "",
    parameters_file: Annotated[
        str,
        typer.Option(
            "--parameters",
            "-p",
            help="Parameters, in yaml,  accessible by the application",
        ),
    ] = "",
    log_level: Annotated[
        LogLevel,
        typer.Option(
            "--log-level",
            help="The log level",
            show_default=True,
            case_sensitive=False,
        ),
    ] = LogLevel.WARNING,
    tag: Annotated[str, typer.Option(help="A tag attached to the run")] = "",
    run_id: Annotated[
        str,
        typer.Option(
            help="An optional run_id, one would be generated if its not provided"
        ),
    ] = "",
):
    logger.setLevel(log_level.value)

    entrypoints.execute_job_yaml_spec(
        configuration_file=config_file,
        job_definition_file=job_definition_file,
        tag=tag,
        run_id=run_id,
        parameters_file=parameters_file,
    )


@app.command(hidden=True)
def execute_job(
    job_definition_file: Annotated[
        str,
        typer.Argument(
            help=("The yaml file containing the job definition"),
        ),
    ],
    run_id: Annotated[
        str,
        typer.Argument(
            envvar="RUNNABLE_RUN_ID",
            help="An optional run_id, one would be generated if its not provided",
        ),
    ] = "",
    config_file: Annotated[
        str,
        typer.Option(
            "--config", "-c", help="The configuration file specifying the services"
        ),
    ] = "",
    parameters_file: Annotated[
        str,
        typer.Option(
            "--parameters",
            "-p",
            help="Parameters, in yaml,  accessible by the application",
        ),
    ] = "",
    mode: Annotated[
        ExecutionMode,
        typer.Option(
            "--mode",
            "-m",
            help="spec in yaml or python sdk",
        ),
    ] = ExecutionMode.YAML,
    log_level: Annotated[
        LogLevel,
        typer.Option(
            "--log-level",
            help="The log level",
            show_default=True,
            case_sensitive=False,
        ),
    ] = LogLevel.WARNING,
    tag: Annotated[str, typer.Option(help="A tag attached to the run")] = "",
):
    logger.setLevel(log_level.value)

    entrypoints.execute_job_non_local(
        configuration_file=config_file,
        job_definition_file=job_definition_file,
        mode=mode,
        tag=tag,
        run_id=run_id,
        parameters_file=parameters_file,
    )


if __name__ == "__main__":
    app()
