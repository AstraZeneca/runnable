import logging
from typing import List, Optional

from extensions.job_executor import GenericJobExecutor
from runnable import console, defaults
from runnable.datastore import DataCatalog
from runnable.tasks import BaseTaskType

logger = logging.getLogger(defaults.LOGGER_NAME)


class LocalJobExecutor(GenericJobExecutor):
    """
    The LocalJobExecutor is a job executor that runs the job locally.
    """

    service_name: str = "local"
    mock: bool = False

    def submit_job(self, job: BaseTaskType, catalog_settings=Optional[List[str]]):
        """
        This method gets invoked by the CLI.
        """
        self._set_up_run_log()

        job_log = self._context.run_log_store.create_job_log()
        self._context.run_log_store.add_job_log(
            run_id=self._context.run_id, job_log=job_log
        )

        self.execute_job(job, catalog_settings=catalog_settings)

    def execute_job(self, job: BaseTaskType, catalog_settings=Optional[List[str]]):
        """
        Focusses on execution of the job.
        """
        logger.info("Trying to execute job")

        job_log = self._context.run_log_store.get_job_log(run_id=self._context.run_id)
        self.add_code_identities(job_log)

        attempt_log = job.execute_command(
            attempt_number=self.step_attempt_number,
            mock=self.mock,
        )

        job_log.status = attempt_log.status
        job_log.attempts.append(attempt_log)

        data_catalogs_put: Optional[List[DataCatalog]] = self._sync_catalog(
            catalog_settings=catalog_settings
        )
        logger.debug(f"data_catalogs_put: {data_catalogs_put}")

        job_log.add_data_catalogs(data_catalogs_put or [])

        console.print("Summary of job")
        console.print(job_log.get_summary())

        self._context.run_log_store.add_job_log(
            run_id=self._context.run_id, job_log=job_log
        )
