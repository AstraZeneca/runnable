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
        tag=tag,
        run_id=run_id,
        parameters_file=parameters_file,
    )


@app.command()
def visualize(
    run_id: Annotated[str, typer.Argument(help="The run ID to visualize")],
    svg: Annotated[
        bool,
        typer.Option(
            "--svg", help="Generate SVG diagram in addition to console output"
        ),
    ] = False,
    interactive: Annotated[
        bool,
        typer.Option(
            "--interactive",
            "-i",
            help="Generate interactive SVG with clickable metadata (requires web browser)",
        ),
    ] = False,
    html: Annotated[
        bool,
        typer.Option(
            "--html", help="Generate rich HTML dashboard with comprehensive analytics"
        ),
    ] = False,
    open_browser: Annotated[
        bool,
        typer.Option(
            "--open/--no-open",
            help="Automatically open the generated file in default browser",
        ),
    ] = True,
    output: Annotated[
        str,
        typer.Option("--output", "-o", help="Custom output path for SVG file"),
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
):
    """
    Visualize a pipeline execution by run ID.

    This command provides visualization of pipeline structure and execution details
    from run logs. It shows the DAG structure, execution timing, and status information.

    Examples:
        runnable visualize online-lamport-0645
        runnable visualize sophisticated-wright-0644 --svg
        runnable visualize online-lamport --svg --output my_pipeline.svg
        runnable visualize online-lamport --interactive
        runnable visualize online-lamport --html --output dashboard.html
        runnable visualize online-lamport --html --no-open
    """
    logger.setLevel(log_level.value)

    from runnable.viz import visualize_run_by_id_enhanced

    output_file = None
    visualization_type = "console"

    if html:
        visualization_type = "html"
        output_file = output if output else f"{run_id}_dashboard.html"
    elif interactive:
        visualization_type = "interactive"
        output_file = output if output else f"{run_id}_diagram.svg"
    elif svg:
        visualization_type = "svg"
        output_file = output if output else f"{run_id}_diagram.svg"

    visualize_run_by_id_enhanced(run_id, output_file, visualization_type, open_browser)


@app.command()
def analyze(
    run_log_dir: Annotated[
        str,
        typer.Option("--dir", "-d", help="Directory containing run log files"),
    ] = ".run_log_store",
    limit: Annotated[
        int,
        typer.Option("--limit", "-l", help="Maximum number of recent runs to analyze"),
    ] = 10,
    log_level: Annotated[
        LogLevel,
        typer.Option(
            "--log-level",
            help="The log level",
            show_default=True,
            case_sensitive=False,
        ),
    ] = LogLevel.WARNING,
):
    """
    Analyze recent pipeline runs and show summary statistics.

    This command analyzes run logs in the specified directory and provides
    insights into pipeline execution patterns, success rates, and performance.

    Examples:
        runnable analyze
        runnable analyze --limit 20
        runnable analyze --dir /path/to/logs --limit 5
    """
    logger.setLevel(log_level.value)

    from runnable.viz import analyze_run_logs

    analyze_run_logs(run_log_dir, limit)


@app.command()
def timeline(
    run_id: Annotated[
        str, typer.Argument(help="The run ID to visualize as a timeline")
    ],
    output: Annotated[
        str,
        typer.Option("--output", "-o", help="Output HTML file path"),
    ] = "",
    open_browser: Annotated[
        bool,
        typer.Option(
            "--open/--no-open",
            help="Automatically open the generated file in default browser",
        ),
    ] = True,
    log_level: Annotated[
        LogLevel,
        typer.Option(
            "--log-level",
            help="The log level",
            show_default=True,
            case_sensitive=False,
        ),
    ] = LogLevel.WARNING,
):
    """
    Visualize pipeline execution as a Gantt chart timeline.

    This command creates timeline visualizations that are particularly effective
    for understanding composite nodes (parallel, map, conditional) because they
    show temporal relationships and hierarchical structure clearly.

    Examples:
        runnable timeline forgiving-joliot-0645
        runnable timeline parallel-run --output timeline.html
        runnable timeline complex-pipeline --no-open
    """
    logger.setLevel(log_level.value)

    from runnable.gantt import visualize_gantt

    output_file = output if output else f"{run_id}_timeline.html"
    visualize_gantt(run_id, output_file, open_browser)


if __name__ == "__main__":
    app()
