import logging
from typing import List, Optional

from extensions.job_executor import GenericJobExecutor
from runnable import defaults
from runnable.tasks import BaseTaskType

logger = logging.getLogger(defaults.LOGGER_NAME)


class LocalJobExecutor(GenericJobExecutor):
    """
    The LocalJobExecutor is a job executor that runs the job locally.

    Configuration:

    pipeline-executor:
        type: local

    """

    service_name: str = "local"

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
