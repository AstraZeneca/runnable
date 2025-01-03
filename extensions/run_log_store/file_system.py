import json
import logging
from pathlib import Path
from typing import Any, Dict

from runnable import defaults, exceptions, utils
from runnable.datastore import BaseRunLogStore, RunLog

logger = logging.getLogger(defaults.LOGGER_NAME)


class FileSystemRunLogstore(BaseRunLogStore):
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

    @property
    def log_folder_name(self):
        return self.log_folder

    def get_summary(self) -> Dict[str, Any]:
        summary = {"Type": self.service_name, "Location": self.log_folder}

        return summary

    def write_to_folder(self, run_log: RunLog):
        """
        Write the run log to the folder

        Args:
            run_log (RunLog): The run log to be added to the database
        """
        write_to = self.log_folder_name
        utils.safe_make_dir(write_to)

        write_to_path = Path(write_to)
        run_id = run_log.run_id
        json_file_path = write_to_path / f"{run_id}.json"

        with json_file_path.open("w") as fw:
            json.dump(run_log.model_dump(), fw, ensure_ascii=True, indent=4)  # pylint: disable=no-member

    def get_from_folder(self, run_id: str) -> RunLog:
        """
        Look into the run log folder for the run log for the run id.

        If the run log does not exist, raise an exception. If it does, decode it
        as a RunLog and return it

        Args:
            run_id (str): The requested run id to retrieve the run log store

        Raises:
            FileNotFoundError: If the Run Log has not been found.

        Returns:
            RunLog: The decoded Run log
        """
        write_to = self.log_folder_name

        read_from_path = Path(write_to)
        json_file_path = read_from_path / f"{run_id}.json"

        if not json_file_path.exists():
            raise FileNotFoundError(f"Expected {json_file_path} is not present")

        with json_file_path.open("r") as fr:
            json_str = json.load(fr)
            run_log = RunLog(**json_str)  # pylint: disable=no-member
        return run_log

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
        self.write_to_folder(run_log)
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
            run_log = self.get_from_folder(run_id)
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
        self.write_to_folder(run_log)
