import logging
import shlex
import subprocess
import sys
from typing import List, Optional

from extensions.job_executor import GenericJobExecutor
from runnable import context, defaults
from runnable.tasks import BaseTaskType

logger = logging.getLogger(defaults.LOGGER_NAME)


class EmulatorJobExecutor(GenericJobExecutor):
    """
    The EmulatorJobExecutor is a job executor that emulates the job execution.
    """

    service_name: str = "emulator"

    def submit_job(self, job: BaseTaskType, catalog_settings=Optional[List[str]]):
        """
        This method gets invoked by the CLI.
        """
        self._set_up_run_log()

        # Call the container job
        job_log = self._context.run_log_store.create_job_log()
        self._context.run_log_store.add_job_log(
            run_id=self._context.run_id, job_log=job_log
        )
        self.run_click_command()

    def run_click_command(self) -> str:
        """
        Execute a Click-based CLI command in the current virtual environment.

        Args:
            command: Click command to execute
        """
        assert isinstance(self._context, context.JobContext)
        command = self._context.get_job_callable_command()

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
