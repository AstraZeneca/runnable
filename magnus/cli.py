import logging
from logging.config import fileConfig

import click
from click_plugins import with_plugins
from pkg_resources import iter_entry_points, resource_filename

from magnus import defaults, docker_utils, pipeline

fileConfig(resource_filename(__name__, 'log_config.ini'))
logger = logging.getLogger(defaults.NAME)


@with_plugins(iter_entry_points('magnus.cli_plugins'))
@click.group()
@click.version_option()
def cli():
    """
    Welcome to magnus. Please provide the command that you want to use.
    All commands have options that you can see by magnus <command> --help
    """
    pass


@cli.command('execute', short_help="Execute/translate a pipeline")
@click.option('-f', '--file', default='pipeline.yaml', help='The pipeline definition file', show_default=True)
@click.option('-v', '--var-file', default=None, help='Variables used in pipeline definition', show_default=True)
@click.option('-c', '--config-file', default=None,
              help="config file, in yaml, to be used for the run", show_default=True)
@click.option('-p', '--parameters-file', default=None,
              help="Parameters, in yaml,  accessible by the application", show_default=True)
@click.option('--log-level', default=defaults.LOG_LEVEL, help='The log level', show_default=True,
              type=click.Choice(['INFO', 'DEBUG', 'WARNING', 'ERROR', 'FATAL']))
@click.option('--tag', help='A tag attached to the run')
@click.option('--run-id', help='An optional run_id, one would be generated if not provided')
@click.option('--use-cached', help='Provide the previous run_id to re-run.', show_default=True)
def execute(file, var_file, config_file, parameters_file, log_level, tag, run_id, use_cached):  # pragma: no cover
    """
    Entry point to executing a pipeline. This command is most commonly used
    either to execute a pipeline or to translate the pipeline definition to another language.

    You can re-run an older run by providing the run_id of the older run in --use-cached.
    Ensure that the catalogs and run logs are accessible by the present configuration.
    """
    logger.setLevel(log_level)
    pipeline.execute(
        variables_file=var_file, configuration_file=config_file, pipeline_file=file, tag=tag,
        run_id=run_id, use_cached=use_cached, parameters_file=parameters_file)


@cli.command('execute_step', short_help="Execute a single step of the pipeline")
@click.argument('step_name')
@click.option('-f', '--file', default='pipeline.yaml', help='The pipeline definition file', show_default=True)
@click.option('-v', '--var-file', default=None, help='Variables used in pipeline definition', show_default=True)
@click.option('-c', '--config-file', default=None,
              help="config file, in yaml, to be used for the run", show_default=True)
@click.option('-p', '--parameters-file', default=None,
              help="Parameters, in yaml,  accessible by the application", show_default=True)
@click.option('--log-level', default=defaults.LOG_LEVEL, help='The log level', show_default=True,
              type=click.Choice(['INFO', 'DEBUG', 'WARNING', 'ERROR', 'FATAL']))
@click.option('--tag', help='A tag attached to the run')
@click.option('--run-id', help='An optional run_id, one would be generated if not provided')
@click.option('--use-cached', help='Provide the previous run_id to re-run.', show_default=True)
def execute_step(step_name, file, var_file, config_file, parameters_file, log_level, tag, run_id, use_cached):  # pragma: no cover
    """
    Entry point to executing a single step of the pipeline.

    This command is helpful to run only one step of the pipeline in isolation.
    Only the steps of the parent dag could be invoked using this method.

    You can re-run an older run by providing the run_id of the older run in --use-cached.
    Ensure that the catalogs and run logs are accessible by the present configuration.

    When running map states, ensure that the parameter to iterate on is available in parameter space.
    """
    logger.setLevel(log_level)
    pipeline.execute_single_step(
        variables_file=var_file, configuration_file=config_file, pipeline_file=file, step_name=step_name, tag=tag,
        run_id=run_id, parameters_file=parameters_file, use_cached=use_cached)


@cli.command("execute_single_node", short_help="Internal entry point to execute a single node", hidden=True)
@click.argument('run_id')
@click.argument('step_name')
@click.option('--map-variable', default='', help='The map variable dictionary in str', show_default=True)
@click.option('-f', '--file', default='pipeline.yaml', help='The pipeline definition file', show_default=True)
@click.option('-v', '--var-file', default=None, help='Variables used in pipeline definition', show_default=True)
@click.option('-c', '--config-file', default=None,
              help="config file, in yaml, to be used for the run", show_default=True)
@click.option('-p', '--parameters-file', default=None,
              help="Parameters, in yaml,  accessible by the application", show_default=True)
@click.option('--log-level', default=defaults.LOG_LEVEL, help='The log level', show_default=True,
              type=click.Choice(['INFO', 'DEBUG', 'WARNING', 'ERROR', 'FATAL']))
@click.option('--tag', default='', help='A tag attached to the run')
def execute_single_node(run_id, step_name, map_variable, file, var_file, config_file, parameters_file, log_level, tag):
    logger.setLevel(log_level)

    pipeline.execute_single_node(variables_file=var_file,
                                 configuration_file=config_file, pipeline_file=file,
                                 step_name=step_name, map_variable=map_variable,
                                 run_id=run_id, tag=tag, parameters_file=parameters_file)


@cli.command("execute_single_branch", short_help="Internal entry point to execute a single branch", hidden=True)
@click.argument('run_id')
@click.argument('branch_name')
@click.option('--map-variable', default='', help='The map variable dictionary in str', show_default=True)
@click.option('-f', '--file', default='pipeline.yaml', help='The pipeline definition file', show_default=True)
@click.option('-v', '--var-file', default=None, help='Variables used in pipeline definition', show_default=True)
@click.option('-c', '--config-file', default=None,
              help="config file, in yaml, to be used for the run", show_default=True)
@click.option('--log-level', default=defaults.LOG_LEVEL, help='The log level', show_default=True,
              type=click.Choice(['INFO', 'DEBUG', 'WARNING', 'ERROR', 'FATAL']))
def execute_single_branch(run_id, branch_name, map_variable, file, var_file, config_file, log_level):
    logger.setLevel(log_level)

    pipeline.execute_single_brach(variables_file=var_file, configuration_file=config_file,
                                  pipeline_file=file, branch_name=branch_name, map_variable=map_variable,
                                  run_id=run_id)


@cli.command('build_docker', short_help="Utility tool to build docker images")
@click.argument('image_name')
@click.option('-f', '--docker-file', default=None, help="The dockerfile to be used. If None, we generate one")
@click.option('-s', '--style', default="poetry", help="The method used to get requirements", show_default=True)
@click.option('-t', "--tag", default="latest", help="The tag assigned to the image", show_default=True)
@click.option('-c', '--commit-tag', is_flag=True, default=False, help="Use commit id as tag. Over-rides tag option",
              show_default=True)
@click.option('-d', '--dry-run', is_flag=True, default=False, help="Generate the dockerfile, but NOT the image",
              show_default=True)
@click.option('--git-tracked/--all', default=True, help="Controls what should be added to image. All vs git-tracked",
              show_default=True)
def build_docker(image_name, docker_file, style, tag, commit_tag, dry_run, git_tracked):
    """
    A utility function to create docker images from the existing codebase.

    It is advised to provide your own dockerfile as much as possible. If you do not have one handy, you can use
    --dry-run functionality to see if the auto-generated one suits your needs.

    If you are auto-generating the dockerfile:
    BEWARE!! Over-riding the default options assumes you know what you are doing! BEWARE!!

    1). By default, only git tracked files are added to the docker image.

    2). The auto-generated dockerfile uses, python 3.7 as the default image and adds the current folder.
    """
    docker_utils.build_docker(image_name=image_name, docker_file=docker_file, style=style,
                              tag=tag, commit_tag=commit_tag, dry_run=dry_run, git_tracked=git_tracked)
