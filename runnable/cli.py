import logging

import click
from click_plugins import with_plugins
from pkg_resources import iter_entry_points

from runnable import defaults, entrypoints

logger = logging.getLogger(defaults.LOGGER_NAME)


@with_plugins(iter_entry_points("runnable.cli_plugins"))
@click.group()
@click.version_option()
def cli():
    """
    Welcome to runnable. Please provide the command that you want to use.
    All commands have options that you can see by runnable <command> --help
    """
    pass


@cli.command("execute", short_help="Execute/translate a pipeline")
@click.option("-f", "--file", default="pipeline.yaml", help="The pipeline definition file", show_default=True)
@click.option(
    "-c", "--config-file", default=None, help="config file, in yaml, to be used for the run", show_default=True
)
@click.option(
    "-p",
    "--parameters-file",
    default=None,
    help="Parameters, in yaml,  accessible by the application",
    show_default=True,
)
@click.option(
    "--log-level",
    default=defaults.LOG_LEVEL,
    help="The log level",
    show_default=True,
    type=click.Choice(["INFO", "DEBUG", "WARNING", "ERROR", "FATAL"]),
)
@click.option("--tag", default="", help="A tag attached to the run")
@click.option("--run-id", help="An optional run_id, one would be generated if not provided")
def execute(file, config_file, parameters_file, log_level, tag, run_id):  # pragma: no cover
    """
    Execute a pipeline

    Usage: runnable execute [OPTIONS]

    Options:
    -f, --file TEXT               The pipeline definition file [default: pipeline.yaml]
    -c, --config-file TEXT        config file, in yaml, to be used for the run [default: None]
    -p, --parameters-file TEXT    Parameters, in yaml,  accessible by the application [default: None]
    --log-level                     One of [INFO|DEBUG|WARNING|ERROR|FATAL]
                                    The log level
                                    [default: INFO]
    --tag TEXT                   A tag attached to the run
                                    [default: ]
    --run-id TEXT                An optional run_id, one would be generated if not
                                    provided
    """
    logger.setLevel(log_level)
    entrypoints.execute(
        configuration_file=config_file,
        pipeline_file=file,
        tag=tag,
        run_id=run_id,
        parameters_file=parameters_file,
    )


@cli.command("execute_single_node", short_help="Internal entry point to execute a single node", hidden=True)
@click.argument("run_id")
@click.argument("step_name")
@click.option("--map-variable", default="", help="The map variable dictionary in str", show_default=True)
@click.option("-f", "--file", default="", help="The pipeline definition file", show_default=True)
@click.option(
    "-c", "--config-file", default=None, help="config file, in yaml, to be used for the run", show_default=True
)
@click.option(
    "-p",
    "--parameters-file",
    default=None,
    help="Parameters, in yaml,  accessible by the application",
    show_default=True,
)
@click.option(
    "--log-level",
    default=defaults.LOG_LEVEL,
    help="The log level",
    show_default=True,
    type=click.Choice(["INFO", "DEBUG", "WARNING", "ERROR", "FATAL"]),
)
@click.option("--tag", default="", help="A tag attached to the run")
def execute_single_node(run_id, step_name, map_variable, file, config_file, parameters_file, log_level, tag):
    """
    Internal entrypoint for runnable to execute a single node.

    Other than local executor, every other executor uses this entry point to execute a step in the context of runnable.
    Only chained executions should use this method. Unchained executions should use execute_
    """
    logger.setLevel(log_level)

    # Execute the node as part of the graph execution.
    entrypoints.execute_single_node(
        configuration_file=config_file,
        pipeline_file=file,
        step_name=step_name,
        map_variable=map_variable,
        run_id=run_id,
        tag=tag,
        parameters_file=parameters_file,
    )


@cli.command("execute_notebook", short_help="Entry point to execute a notebook")
@click.argument("filename")
@click.option("--entrypoint", default=defaults.ENTRYPOINT.USER.value, hidden=True)
@click.option(
    "-c", "--config-file", default=None, help="config file, in yaml, to be used for the run", show_default=True
)
@click.option(
    "-p",
    "--parameters-file",
    default=None,
    help="Parameters, in yaml,  accessible by the application",
    show_default=True,
)
@click.option(
    "--log-level",
    default=defaults.LOG_LEVEL,
    help="The log level",
    show_default=True,
    type=click.Choice(["INFO", "DEBUG", "WARNING", "ERROR", "FATAL"]),
)
@click.option("--data-folder", "-d", default="data/", help="The catalog data folder")
@click.option("--put-in-catalog", "-put", default=None, multiple=True, help="The data to put from the catalog")
@click.option("--notebook-output-path", default="", help="The output path for the notebook")
@click.option("--tag", help="A tag attached to the run")
@click.option("--run-id", help="An optional run_id, one would be generated if not provided")
def execute_notebook(
    filename,
    entrypoint,
    config_file,
    parameters_file,
    log_level,
    data_folder,
    put_in_catalog,
    notebook_output_path,
    tag,
    run_id,
):
    """
    External entry point to execute a Jupyter notebook in isolation.

    The notebook would be executed in the environment defined by the config file or default if none.
    The execution plan is unchained.
    """
    logger.setLevel(log_level)
    catalog_config = {"compute_data_folder": data_folder, "put": list(put_in_catalog) if put_in_catalog else None}
    if not filename.endswith(".ipynb"):
        raise Exception("A notebook should always have ipynb as the extension")

    entrypoints.execute_notebook(
        entrypoint=entrypoint,
        notebook_file=filename,
        catalog_config=catalog_config,
        configuration_file=config_file,
        parameters_file=parameters_file,
        notebook_output_path=notebook_output_path,
        tag=tag,
        run_id=run_id,
    )


@cli.command("execute_function", short_help="Entry point to execute a python function")
@click.argument("command")
@click.option("--entrypoint", default=defaults.ENTRYPOINT.USER.value, hidden=True)
@click.option(
    "-c", "--config-file", default=None, help="config file, in yaml, to be used for the run", show_default=True
)
@click.option(
    "-p",
    "--parameters-file",
    default=None,
    help="Parameters, in yaml,  accessible by the application",
    show_default=True,
)
@click.option(
    "--log-level",
    default=defaults.LOG_LEVEL,
    help="The log level",
    show_default=True,
    type=click.Choice(["INFO", "DEBUG", "WARNING", "ERROR", "FATAL"]),
)
@click.option("--data-folder", "-d", default="data/", help="The catalog data folder")
@click.option("--put-in-catalog", "-put", default=None, multiple=True, help="The data to put from the catalog")
@click.option("--tag", help="A tag attached to the run")
@click.option("--run-id", help="An optional run_id, one would be generated if not provided")
def execute_function(
    command, entrypoint, config_file, parameters_file, log_level, data_folder, put_in_catalog, tag, run_id
):
    """
    External entry point to execute a python function in isolation.

    The function would be executed in the environment defined by the config file or default if none.
    The execution plan is unchained.
    """
    logger.setLevel(log_level)
    catalog_config = {"compute_data_folder": data_folder, "put": list(put_in_catalog) if put_in_catalog else None}
    entrypoints.execute_function(
        entrypoint=entrypoint,
        command=command,
        catalog_config=catalog_config,
        configuration_file=config_file,
        parameters_file=parameters_file,
        tag=tag,
        run_id=run_id,
    )


@cli.command("fan", short_help="Internal entry point to fan in or out a composite node", hidden=True)
@click.argument("run_id")
@click.argument("step_name")
@click.option("-m", "--mode", help="fan in or fan out", required=True, type=click.Choice(["in", "out"]))
@click.option("--map-variable", default="", help="The map variable dictionary in str", show_default=True)
@click.option("-f", "--file", default="", help="The pipeline definition file", show_default=True)
@click.option(
    "-c", "--config-file", default=None, help="config file, in yaml, to be used for the run", show_default=True
)
@click.option(
    "-p",
    "--parameters-file",
    default=None,
    help="Parameters, in yaml,  accessible by the application",
    show_default=True,
)
@click.option(
    "--log-level",
    default=defaults.LOG_LEVEL,
    help="The log level",
    show_default=True,
    type=click.Choice(["INFO", "DEBUG", "WARNING", "ERROR", "FATAL"]),
)
@click.option("--tag", default="", help="A tag attached to the run")
def fan(run_id, step_name, mode, map_variable, file, config_file, parameters_file, log_level, tag):
    """
    Internal entrypoint for runnable to fan in or out a composite node.

    Only 3rd party orchestrators should use this entry point.
    """
    logger.setLevel(log_level)

    # Fan in or out
    entrypoints.fan(
        configuration_file=config_file,
        pipeline_file=file,
        step_name=step_name,
        mode=mode,
        map_variable=map_variable,
        run_id=run_id,
        tag=tag,
        parameters_file=parameters_file,
    )


# Needed for the binary creation
if __name__ == "__main__":
    cli()
