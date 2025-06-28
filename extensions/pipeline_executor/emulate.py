import logging
import shlex
import subprocess
import sys

from pydantic import PrivateAttr

from extensions.pipeline_executor import GenericPipelineExecutor
from runnable import defaults
from runnable.defaults import MapVariableType
from runnable.nodes import BaseNode

logger = logging.getLogger(defaults.LOGGER_NAME)


class Emulator(GenericPipelineExecutor):
    """
    In the mode of local execution, we run everything on the local computer.

    This has some serious implications on the amount of time it would take to complete the run.
    Also ensure that the local compute is good enough for the compute to happen of all the steps.

    Example config:

    ```yaml
    pipeline-executor:
      type: local
    ```

    """

    service_name: str = "emulator"

    _should_setup_run_log_at_traversal: bool = PrivateAttr(default=True)

    def trigger_node_execution(
        self, node: BaseNode, map_variable: MapVariableType = None
    ):
        """
        In this mode of execution, we prepare for the node execution and execute the node

        Args:
            node (BaseNode): [description]
            map_variable (str, optional): [description]. Defaults to ''.
        """
        command = self._context.get_node_callable_command(
            node, map_variable=map_variable
        )

        self.run_click_command(command)
        # execute the command in a forked process

        step_log = self._context.run_log_store.get_step_log(
            node._get_step_log_name(map_variable), self._context.run_id
        )
        if step_log.status != defaults.SUCCESS:
            msg = "Node execution inside the emulate failed. Please check the logs.\n"
            logger.error(msg)
            step_log.status = defaults.FAIL
            self._context.run_log_store.add_step_log(step_log, self._context.run_id)

    def execute_node(self, node: BaseNode, map_variable: MapVariableType = None):
        """
        For local execution, we just execute the node.

        Args:
            node (BaseNode): _description_
            map_variable (dict[str, str], optional): _description_. Defaults to None.
        """
        self._execute_node(node=node, map_variable=map_variable)

    def run_click_command(self, command: str) -> str:
        """
        Execute a Click-based CLI command in the current virtual environment.

        Args:
            args: List of Click command arguments (including subcommands and options)

        Returns:
            Combined stdout/stderr output as string
        """
        # For Click commands installed via setup.py entry_points
        # command = [sys.executable, '-m', 'your_package.cli'] + args

        # For direct module execution
        sub_command = [sys.executable, "-m", "runnable.cli"] + shlex.split(command)[1:]

        process = subprocess.Popen(
            sub_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
        )

        output = []
        try:
            while True:
                line = process.stdout.readline()  # type: ignore
                if not line and process.poll() is not None:
                    break
                print(line, end="")
                output.append(line)
        finally:
            process.stdout.close()  # type: ignore

        if process.returncode != 0:
            raise subprocess.CalledProcessError(
                process.returncode, command, "".join(output)
            )

        return "".join(output)
