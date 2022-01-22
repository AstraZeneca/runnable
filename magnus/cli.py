import argparse
import logging
import sys
from logging.config import fileConfig

from pkg_resources import resource_filename

from magnus import defaults, pipeline

fileConfig(resource_filename(__name__, 'log_config.ini'))
logger = logging.getLogger(defaults.NAME)


class MagnusCLI:
    """
    Dispatcher pattern class for the CLI
    """

    def __init__(self):
        parser = argparse.ArgumentParser(
            description='Magnus CLI',
            usage='''pipeline <command> [<args>]

Welcome to magnus. Please choose the command you want to run.

The available commands are:
   execute   executes the pipeline
''')
        parser.add_argument('command', help='Subcommand to run')
        # parse_args defaults to [1:] for args, but you need to
        # exclude the rest of the args too, or validation will fail
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            parser.print_help()
            sys.exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()

    @staticmethod
    def _resolve_args(sys_args, parser):
        """
        Method to resolve known args and unknown args.
        """
        args, unknown = parser.parse_known_args(sys_args[2:])
        kwargs = {}
        for arg in unknown:
            if arg.startswith(("-", "--")):
                parser.add_argument(arg.split('=')[0])
                kwargs[arg.split('=')[0].replace('-', '')] = None

        args = parser.parse_args(sys.argv[2:])
        for kwarg in kwargs:
            kwargs[kwarg] = vars(args)[kwarg]

        return args, {k.lower(): v for k, v in kwargs.items()}

    def execute(self):
        """
        Entry point to interactive modules
        """
        parser = argparse.ArgumentParser(
            description='Execute the pipeline based on the definition')
        # prefixing the argument with -- means it's optional
        parser.add_argument('-f', '--file', default='pipeline.yaml', help='The pipeline definition location')
        parser.add_argument('-v', '--var-file', default=None, help='Variables used in pipeline')
        parser.add_argument('-c', '--config-file', default=None,
                            help="Provide a config file which over-rides everything")
        parser.add_argument('--log-level', default=defaults.LOG_LEVEL, help='The logger level')
        parser.add_argument('--tag', help='A tag attached to the run')
        parser.add_argument('--run-id', help='An optional run_id, one would be generated if not provided')
        parser.add_argument('--use-cached', help='Provide the previous run_id to re-run.')

        # now that we're inside a subcommand, ignore the first
        # TWO argvs, ie the command (git) and the subcommand (commit)
        args, kwargs = self._resolve_args(sys.argv, parser)

        logger.setLevel(args.log_level)
        pipeline.execute(
            variables_file=args.var_file, configuration_file=args.config_file, pipeline_file=args.file, tag=args.tag,
            run_id=args.run_id, use_cached=args.use_cached, **kwargs)

    def execute_single_node(self):
        """
        Entry point for orchestrated modes.
        Interactive modes can use this when graph traversal and actual execution are in different environments
        """
        parser = argparse.ArgumentParser(
            description='Execute a single node of the pipeline')
        # prefixing the argument with -- means it's optional
        parser.add_argument('run_id', help='Run id of the run')
        parser.add_argument('step_name', help='The internal step name of the pipeline to run')
        parser.add_argument('--map-variable', default='', help='The map iterator variable dictionary')
        parser.add_argument('-f', '--file', default='pipeline.yaml', help='The pipeline definition location')
        parser.add_argument('-v', '--var-file', default=None, help='Variables used in pipeline')
        parser.add_argument('-c', '--config-file', default=None,
                            help="Provide a config file which over-rides everything")
        parser.add_argument('--log-level', default=defaults.LOG_LEVEL, help='The logger level')
        parser.add_argument('--tag', help='A tag attached to the run')

        # now that we're inside a subcommand, ignore the first
        # TWO argvs, ie the command (git) and the subcommand (commit)
        args, kwargs = self._resolve_args(sys.argv, parser)
        logger.setLevel(args.log_level)

        pipeline.execute_single_node(
            variables_file=args.var_file, configuration_file=args.config_file, pipeline_file=args.file,
            step_name=args.step_name, map_variable=args.map_variable, run_id=args.run_id, tag=args.tag, **kwargs)

    def execute_single_branch(self):
        """
        Entry point for executing a branch.
        Mostly used by interactive modes where parallelization of branch is enabled
        """
        parser = argparse.ArgumentParser(
            description='Execute a single node of the pipeline')
        # prefixing the argument with -- means it's optional
        parser.add_argument('run_id', help='Run id of the run')
        parser.add_argument('branch_name', help='The internal branch name of the pipeline to run')
        parser.add_argument('--map-variable', default='', help='The map iterator variable')
        parser.add_argument('-f', '--file', default='pipeline.yaml', help='The pipeline definition location')
        parser.add_argument('-v', '--var-file', default=None, help='Variables used in pipeline')
        parser.add_argument('-c', '--config-file', default=None,
                            help="Provide a config file which over-rides everything")
        parser.add_argument('--log-level', default=defaults.LOG_LEVEL, help='The logger level')
        parser.add_argument('--tag', help='A tag attached to the run')

        # now that we're inside a subcommand, ignore the first
        # TWO argvs, ie the command (git) and the subcommand (commit)
        args, kwargs = self._resolve_args(sys.argv, parser)
        logger.setLevel(args.log_level)

        pipeline.execute_single_brach(
            variables_file=args.var_file, configuration_file=args.config_file, pipeline_file=args.file,
            branch_name=args.branch_name, map_variable=args.map_variable, run_id=args.run_id, tag=args.tag, **kwargs)


def main():  # pragma: no cover
    """
    Required for the setup.py
    """
    MagnusCLI()
