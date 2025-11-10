import logging
import os
from enum import Enum
from pathlib import Path
from typing import Annotated

import typer

from runnable import defaults, entrypoints
from runnable.gantt import SimpleVisualizer, generate_html_timeline, visualize_simple

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
def timeline(
    run_id_or_path: Annotated[
        str, typer.Argument(help="Run ID to visualize, or path to JSON run log file")
    ],
    output: Annotated[
        str,
        typer.Option("--output", "-o", help="Output HTML file path"),
    ] = "",
    console: Annotated[
        bool,
        typer.Option(
            "--console/--no-console",
            help="Show console timeline output (default: true)",
        ),
    ] = True,
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
    Visualize pipeline execution as an interactive timeline.

    This command creates lightweight timeline visualizations that effectively
    show composite nodes (parallel, map, conditional) with hierarchical structure,
    timing information, and execution metadata.

    The new visualization system provides:
    - Clean console output with hierarchical display
    - Interactive HTML with hover tooltips and expandable sections
    - Proper support for all composite pipeline types
    - Rich metadata including commands, parameters, and catalog operations

    By default, shows console output AND generates HTML file with browser opening.

    Input Options:
    - Run ID: Looks up JSON file in .run_log_store/ directory
    - JSON Path: Direct path to run log JSON file (flexible for any config)

    Examples:
        # Using Run ID (looks in .run_log_store/)
        runnable timeline forgiving-joliot-0645                    # Console + HTML + browser
        runnable timeline parallel-run --output custom.html       # Console + custom HTML + browser

        # Using JSON file path (any location)
        runnable timeline /path/to/my-run.json                    # Console + HTML + browser
        runnable timeline ../logs/pipeline-run.json --no-open     # Console + HTML, no browser
        runnable timeline ~/experiments/run.json --no-console     # HTML + browser only

        # Other options
        runnable timeline complex-pipeline --no-open              # Console + HTML, no browser
        runnable timeline simple-run --no-console --no-open       # HTML only, no browser
    """
    logger.setLevel(log_level.value)

    # Determine if input is a file path or run ID
    if os.path.exists(run_id_or_path) or run_id_or_path.endswith(".json"):
        # Input is a file path
        json_file_path = Path(run_id_or_path)
        if not json_file_path.exists():
            print(f"❌ JSON file not found: {json_file_path}")
            return

        # Extract run ID from the file for default naming
        run_id = json_file_path.stem
        mode = "file"
    else:
        # Input is a run ID - use existing behavior
        run_id = run_id_or_path
        json_file_path = None
        mode = "run_id"

    # Default console behavior: always show console output
    show_console = console if console is not None else True

    if output:
        # Generate HTML file with console output
        output_file = output
        print(f"🌐 Generating timeline: {output_file}")

        if show_console:
            # Show console output first, then generate HTML
            if mode == "file":
                _visualize_simple_from_file(json_file_path, show_summary=False)
            else:
                visualize_simple(run_id, show_summary=False)
            print(f"\n🌐 Generating HTML timeline: {output_file}")

        if mode == "file":
            _generate_html_timeline_from_file(json_file_path, output_file, open_browser)
        else:
            generate_html_timeline(run_id, output_file, open_browser)
    else:
        # Default behavior: show console + generate HTML with browser
        if show_console:
            if mode == "file":
                _visualize_simple_from_file(json_file_path, show_summary=False)
            else:
                visualize_simple(run_id, show_summary=False)

        # Always generate HTML file and open browser by default
        output_file = f"{run_id}_timeline.html"
        print(f"\n🌐 Generating HTML timeline: {output_file}")
        if mode == "file":
            _generate_html_timeline_from_file(json_file_path, output_file, open_browser)
        else:
            generate_html_timeline(run_id, output_file, open_browser)


def _visualize_simple_from_file(json_file_path, show_summary: bool = False) -> None:
    """Visualize timeline from JSON file path."""

    try:
        viz = SimpleVisualizer(json_file_path)
        viz.print_simple_timeline()
        if show_summary:
            viz.print_execution_summary()
    except Exception as e:
        print(f"❌ Error reading JSON file: {e}")


def _generate_html_timeline_from_file(
    json_file_path, output_file: str, open_browser: bool = True
) -> None:
    """Generate HTML timeline from JSON file path."""

    try:
        viz = SimpleVisualizer(json_file_path)
        viz.generate_html_timeline(output_file)

        if open_browser:
            import webbrowser

            file_path = Path(output_file).absolute()
            print(f"🌐 Opening timeline in browser: {file_path.name}")
            webbrowser.open(file_path.as_uri())
    except Exception as e:
        print(f"❌ Error generating HTML: {e}")


if __name__ == "__main__":
    app()
