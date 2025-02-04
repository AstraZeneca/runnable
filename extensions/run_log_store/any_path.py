import logging
from abc import abstractmethod
from typing import Any, Dict

from runnable import defaults, exceptions
from runnable.datastore import BaseRunLogStore, RunLog

logger = logging.getLogger(defaults.LOGGER_NAME)


class AnyPathRunLogStore(BaseRunLogStore):
    """
    In this type of Run Log store, we use a file system to store the JSON run log.

    Every single run is stored as a different file which makes it compatible across other store types.

    When to use:
        When locally testing a pipeline and have the need to compare across runs.
        Its fully featured and perfectly fine if your local environment is where you would do everything.

    Do not use:
        If you need parallelization on local, this run log would not support it.

    Example config:

    run_log:
      type: file-system
      config:
        log_folder: The folder to out the logs. Defaults to .run_log_store

    """

    service_name: str = "file-system"
    log_folder: str = defaults.LOG_LOCATION_FOLDER

    def get_summary(self) -> Dict[str, Any]:
        summary = {"Type": self.service_name, "Location": self.log_folder}

        return summary

    @abstractmethod
    def write_to_path(self, run_log: RunLog): ...

    @abstractmethod
    def read_from_path(self, run_id: str) -> RunLog: ...

    def create_run_log(
        self,
        run_id: str,
        dag_hash: str = "",
        use_cached: bool = False,
        tag: str = "",
        original_run_id: str = "",
        status: str = defaults.CREATED,
    ) -> RunLog:
        """
        # Creates a Run log
        # Adds it to the db
        """

        try:
            self.get_run_log_by_id(run_id=run_id, full=False)
            raise exceptions.RunLogExistsError(run_id=run_id)
        except exceptions.RunLogNotFoundError:
            pass

        logger.info(f"{self.service_name} Creating a Run Log for : {run_id}")
        run_log = RunLog(
            run_id=run_id,
            dag_hash=dag_hash,
            tag=tag,
            status=status,
        )
        self.write_to_path(run_log)
        return run_log

    def get_run_log_by_id(
        self,
        run_id: str,
        full: bool = False,
    ) -> RunLog:
        """
        # Returns the run_log defined by id
        # Raises Exception if not found
        """
        try:
            logger.info(f"{self.service_name} Getting a Run Log for : {run_id}")
            run_log = self.read_from_path(run_id)
            return run_log
        except FileNotFoundError as e:
            raise exceptions.RunLogNotFoundError(run_id) from e

    def put_run_log(self, run_log: RunLog):
        """
        # Puts the run_log into the database
        """
        logger.info(
            f"{self.service_name} Putting the run log in the DB: {run_log.run_id}"
        )
        self.write_to_path(run_log)
