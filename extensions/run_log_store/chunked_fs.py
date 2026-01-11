import json
import logging
from pathlib import Path
from typing import Any, Dict, Union

from extensions.run_log_store.generic_chunked import ChunkedRunLogStore
from runnable import defaults, utils

logger = logging.getLogger(defaults.LOGGER_NAME)


class ChunkedFileSystemRunLogStore(ChunkedRunLogStore):
    """
    File system run log store but chunks the run log into thread safe chunks.
    This enables executions to be parallel.
    """

    service_name: str = "chunked-fs"
    log_folder: str = defaults.LOG_LOCATION_FOLDER

    def get_summary(self) -> Dict[str, Any]:
        summary = {"Type": self.service_name, "Location": self.log_folder}

        return summary

    def _exists(self, run_id: str, name: str) -> bool:
        """
        Check if a file exists in the log folder.

        Args:
            run_id (str): The run id
            name (str): The exact file name to check

        Returns:
            bool: True if file exists, False otherwise
        """
        log_folder = self.log_folder_with_run_id(run_id=run_id)
        file_path = log_folder / self.safe_suffix_json(name)
        return file_path.exists()

    def _list_branch_logs(self, run_id: str) -> list[str]:
        """
        List all branch log file names for a run_id.

        Args:
            run_id (str): The run id

        Returns:
            list[str]: List of branch log file names without .json extension
        """
        log_folder = self.log_folder_with_run_id(run_id=run_id)
        if not log_folder.exists():
            return []

        # Find all files starting with "BranchLog-"
        branch_files = list(log_folder.glob("BranchLog-*.json"))
        # Return file names without path and without .json extension
        return [f.stem for f in branch_files]

    def log_folder_with_run_id(self, run_id: str) -> Path:
        """
        Utility function to get the log folder for a run id.

        Args:
            run_id (str): The run id

        Returns:
            Path: The path to the log folder with the run id
        """
        return Path(self.log_folder) / run_id

    def safe_suffix_json(self, name: Union[Path, str]) -> str:
        """
        Safely attach a suffix to a json file.

        Args:
            name (Path): The name of the file with or without suffix of json

        Returns:
            str : The name of the file with .json
        """
        if str(name).endswith("json"):
            return str(name)

        return str(name) + ".json"

    def _store(self, run_id: str, contents: dict, name: str, insert=False):
        """
        Store the contents against the name in the folder.

        Args:
            run_id (str): The run id
            contents (dict): The dict to store
            name (str): The name to store as (without path)
            insert (bool): Whether this is a new insert (unused, kept for compatibility)
        """
        log_folder_with_run_id = self.log_folder_with_run_id(run_id=run_id)
        file_path = log_folder_with_run_id / name

        utils.safe_make_dir(log_folder_with_run_id)

        with open(self.safe_suffix_json(file_path), "w") as fw:
            json.dump(contents, fw, ensure_ascii=True, indent=4)

    def _retrieve(self, run_id: str, name: str) -> dict:
        """
        Does the job of retrieving from the folder.

        Args:
            run_id (str): The run id
            name (str): the name of the file to retrieve (without path)

        Returns:
            dict: The contents
        """
        log_folder_with_run_id = self.log_folder_with_run_id(run_id=run_id)
        file_path = log_folder_with_run_id / name

        with open(self.safe_suffix_json(file_path), "r") as fr:
            contents = json.load(fr)

        return contents
