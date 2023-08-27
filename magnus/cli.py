import logging
from logging.config import fileConfig

import click
from click_plugins import with_plugins
from pkg_resources import iter_entry_points, resource_filename

from magnus import defaults, docker_utils, pipeline

fileConfig(resource_filename(__name__, "log_config.ini"))
logger = logging.getLogger(defaults.NAME)


@with_plugins(iter_entry_points("magnus.cli_plugins"))
@click.group()
@click.version_option()
def cli():
    """
    Welcome to magnus. Please provide the command that you want to use.
    All commands have options that you can see by magnus <command> --help
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
@click.option("--tag", help="A tag attached to the run")
@click.option("--run-id", help="An optional run_id, one would be generated if not provided")
@click.option("--use-cached", help="Provide the previous run_id to re-run.", show_default=True)
def execute(file, config_file, parameters_file, log_level, tag, run_id, use_cached):  # pragma: no cover
    """
    External entry point to executing a pipeline. This command is most commonly used
    either to execute a pipeline or to translate the pipeline definition to another language.

    You can re-run an older run by providing the run_id of the older run in --use-cached.
    Ensure that the catalogs and run logs are accessible by the present configuration.
    """
    logger.setLevel(log_level)
    pipeline.execute(
        configuration_file=config_file,
        pipeline_file=file,
        tag=tag,
        run_id=run_id,
        use_cached=use_cached,
        parameters_file=parameters_file,
    )


@cli.command("execute_step", short_help="Execute a single step of the pipeline")
@click.argument("step_name")
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
@click.option("--tag", help="A tag attached to the run")
@click.option("--run-id", help="An optional run_id, one would be generated if not provided")
@click.option("--use-cached", help="Provide the previous run_id to re-run.", show_default=True)
def execute_step(step_name, file, config_file, parameters_file, log_level, tag, run_id, use_cached):  # pragma: no cover
    """
    External entry point to executing a single step of the pipeline.

    This command is helpful to run only one step of the pipeline in isolation.
    Only the steps of the parent dag could be invoked using this method.

    You can re-run an older run by providing the run_id of the older run in --use-cached.
    Ensure that the catalogs and run logs are accessible by the present configuration.

    When running map states, ensure that the parameter to iterate on is available in parameter space.
    """
    logger.setLevel(log_level)
    pipeline.execute_single_step(
        configuration_file=config_file,
        pipeline_file=file,
        step_name=step_name,
        tag=tag,
        run_id=run_id,
        parameters_file=parameters_file,
        use_cached=use_cached,
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
    Internal entrypoint for magnus to execute a single node.

    Other than local executor, every other executor uses this entry point to execute a step in the context of magnus.
    Only chained executions should use this method. Unchained executions should use execute_
    """
    logger.setLevel(log_level)

    # Execute the node as part of the graph execution.
    pipeline.execute_single_node(
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

    pipeline.execute_notebook(
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
    pipeline.execute_function(
        entrypoint=entrypoint,
        command=command,
        catalog_config=catalog_config,
        configuration_file=config_file,
        parameters_file=parameters_file,
        tag=tag,
        run_id=run_id,
    )


@cli.command("execute_container", short_help="Entry point to execute a container")
@click.argument("image")
@click.option("--entrypoint", default=defaults.ENTRYPOINT.USER.value, hidden=True)
@click.option("--command", default="", help="The command to execute. Defaults to CMD of image")
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
@click.option(
    "--context-path",
    default=defaults.DEFAULT_CONTAINER_CONTEXT_PATH,
    help="The context path for data and parameter files",
)
@click.option(
    "--data-folder",
    default=defaults.DEFAULT_CONTAINER_DATA_PATH,
    help="The catalog data folder relative to context",
)
@click.option(
    "--output-parameters-file", default=defaults.DEFAULT_CONTAINER_OUTPUT_PARAMETERS, help="The output parameters file"
)
@click.option("--experiment-tracking-file", default="", help="The output experiment tracking file")
@click.option("--put-in-catalog", "-put", default=None, multiple=True, help="The data to put from the catalog")
@click.option("--expose-secret", default=None, multiple=True, help="The secret to expose to the container")
@click.option("--tag", help="A tag attached to the run")
@click.option("--run-id", help="An optional run_id, one would be generated if not provided")
def execute_container(
    image,
    entrypoint,
    command,
    config_file,
    parameters_file,
    log_level,
    context_path,
    data_folder,
    output_parameters_file,
    experiment_tracking_file,
    put_in_catalog,
    expose_secret,
    tag,
    run_id,
):
    """
    External entry point to execute a container in isolation.

    The container would be executed in the environment defined by the config file or default if none.
    The execution plan is unchained.
    """
    logger.setLevel(log_level)
    catalog_config = {"compute_data_folder": data_folder, "put": list(put_in_catalog) if put_in_catalog else None}
    expose_secrets = list(expose_secret) if expose_secret else []
    pipeline.execute_container(
        image=image,
        entrypoint=entrypoint,
        command=command,
        configuration_file=config_file,
        parameters_file=parameters_file,
        context_path=context_path,
        catalog_config=catalog_config,
        output_parameters_file=output_parameters_file,
        experiment_tracking_file=experiment_tracking_file,
        expose_secrets=expose_secrets,
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
    Internal entrypoint for magnus to fan in or out a composite node.

    Only 3rd party orchestrators should use this entry point.
    """
    logger.setLevel(log_level)

    # Fan in or out
    pipeline.fan(
        configuration_file=config_file,
        pipeline_file=file,
        step_name=step_name,
        mode=mode,
        map_variable=map_variable,
        run_id=run_id,
        tag=tag,
        parameters_file=parameters_file,
    )


@cli.command("wrap_around_container", short_help="Internal entry point to sync data/parameters in and out", hidden=True)
@click.argument("run_id")
@click.argument("step_identifier")
@click.option("--map-variable", default="", help="The map variable dictionary in str", show_default=True)
@click.option(
    "-m", "--mode", help="pre or post execution of the container", required=True, type=click.Choice(["pre", "post"])
)
def wrap_around_container(run_id: str, step_identifier: str, map_variable: str, mode: str):
    """
    Internal entrypoint for magnus to sync data/parameters in and out.

    Only 3rd party orchestrators using containers as command types should use this entry point.

    mode:
    pre would be called prior the execution of the container.
        - Create the step log
        - It should read the step config from environmental variables and resolve it with the executor config.
        - sync catalog/parameters and send it in.
    post would be called after the execution of the container.
        - Update the step log
        - Sync back the catalog/parameters and send it to central storage.


    Args:
        run_id (str): The run_id to identify parameters/run log/catalog information
        step_identifier (str): A unique identifier to retrieve the step configuration
        mode (str): Pre or post processing of the container execution
    """


@cli.command("build_docker", short_help="Utility tool to build docker images")
@click.argument("image_name")
@click.option("-f", "--docker-file", default=None, help="The dockerfile to be used. If None, we generate one")
@click.option("-s", "--style", default="poetry", help="The method used to get requirements", show_default=True)
@click.option("-t", "--tag", default="latest", help="The tag assigned to the image", show_default=True)
@click.option(
    "-c",
    "--commit-tag",
    is_flag=True,
    default=False,
    help="Use commit id as tag. Over-rides tag option",
    show_default=True,
)
@click.option(
    "-d", "--dry-run", is_flag=True, default=False, help="Generate the dockerfile, but NOT the image", show_default=True
)
@click.option(
    "--git-tracked/--all",
    default=True,
    help="Controls what should be added to image. All vs git-tracked",
    show_default=True,
)
def build_docker(image_name, docker_file, style, tag, commit_tag, dry_run, git_tracked):
    """
    A utility function to create docker images from the existing codebase.

    It is advised to provide your own dockerfile as much as possible. If you do not have one handy, you can use
    --dry-run functionality to see if the auto-generated one suits your needs.

    If you are auto-generating the dockerfile:
    BEWARE!! Over-riding the default options assumes you know what you are doing! BEWARE!!

    1). By default, only git tracked files are added to the docker image.

    2). The auto-generated dockerfile uses, python 3.8 as the default image and adds the current folder.
    """
    docker_utils.build_docker(
        image_name=image_name,
        docker_file=docker_file,
        style=style,
        tag=tag,
        commit_tag=commit_tag,
        dry_run=dry_run,
        git_tracked=git_tracked,
    )
