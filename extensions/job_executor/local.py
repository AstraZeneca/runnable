import logging

from extensions.job_executor import GenericJobExecutor
from runnable import console, defaults
from runnable.tasks import BaseTaskType

logger = logging.getLogger(defaults.LOGGER_NAME)


class LocalJobExecutor(GenericJobExecutor):
    """
    The LocalJobExecutor is a job executor that runs the job locally.
    """

    service_name: str = "local"
    mock: bool = False

    def submit_job(self, job: BaseTaskType):
        """
        This method gets invoked by the CLI.
        """
        self.pre_job_execution(job)
        self.execute_job(job)

    def pre_job_execution(self, job: BaseTaskType):
        """
        This method is called before the job execution.
        """
        job_log = self._context.run_log_store.create_job_log()
        self._context.run_log_store.add_job_log(
            run_id=self._context.run_id, job_log=job_log
        )

    def execute_job(self, job: BaseTaskType):
        """
        Focusses on execution of the job.
        """
        self.prepare_for_execution()
        logger.info("Trying to execute job")

        job_log = self._context.run_log_store.get_job_log(run_id=self._context.run_id)

        attempt_log = job.execute_command(
            attempt_number=self.step_attempt_number,
            mock=self.mock,
        )

        job_log.status = attempt_log.status
        job_log.attempts.append(attempt_log)

        # data_catalogs_put: Optional[List[DataCatalog]] = self._sync_catalog(stage="put")
        # logger.debug(f"data_catalogs_put: {data_catalogs_put}")

        # step_log.add_data_catalogs(data_catalogs_put or [])

        console.print("Summary of job")
        console.print(job_log.get_summary())

        self._context.run_log_store.add_job_log(
            run_id=self._context.run_id, job_log=job_log
        )

    def post_job_execution(self, job: BaseTaskType):
        """
        This method is called after the job execution.
        """
        ...
