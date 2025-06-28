import logging
import shlex
import subprocess
import sys
from typing import List, Optional


from extensions.job_executor import GenericJobExecutor
from runnable import console, context, defaults
from runnable.datastore import DataCatalog
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

    def execute_job(self, job: BaseTaskType, catalog_settings=Optional[List[str]]):
        """
        Focusses on execution of the job.
        """
        logger.info("Trying to execute job")

        job_log = self._context.run_log_store.get_job_log(run_id=self._context.run_id)
        self.add_code_identities(job_log)

        attempt_log = job.execute_command()

        job_log.status = attempt_log.status
        job_log.attempts.append(attempt_log)

        allow_file_not_found_exc = True
        if job_log.status == defaults.SUCCESS:
            allow_file_not_found_exc = False

        data_catalogs_put: Optional[List[DataCatalog]] = self._sync_catalog(
            catalog_settings=catalog_settings,
            allow_file_not_found_exc=allow_file_not_found_exc,
        )

        logger.debug(f"data_catalogs_put: {data_catalogs_put}")

        job_log.add_data_catalogs(data_catalogs_put or [])

        console.print("Summary of job")
        console.print(job_log.get_summary())

        self._context.run_log_store.add_job_log(
            run_id=self._context.run_id, job_log=job_log
        )

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
