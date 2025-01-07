import json
import logging
from pathlib import Path
from string import Template
from typing import Any, Dict, Optional, Sequence, Union

from extensions.run_log_store.generic_chunked import ChunkedRunLogStore
from runnable import defaults, utils

logger = logging.getLogger(defaults.LOGGER_NAME)

T = Union[str, Path]


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

    def get_matches(
        self, run_id: str, name: str, multiple_allowed: bool = False
    ) -> Optional[Union[Sequence[T], T]]:
        """
        Get contents of files matching the pattern name*

        Args:
            run_id (str): The run id
            name (str): The suffix of the file name to check in the run log store.
        """
        log_folder = self.log_folder_with_run_id(run_id=run_id)
        sub_name = Template(name).safe_substitute({"creation_time": ""})

        matches = list(log_folder.glob(f"{sub_name}*"))

        if matches:
            if not multiple_allowed:
                if len(matches) > 1:
                    msg = f"Multiple matches found for {name} while multiple is not allowed"
                    raise Exception(msg)
                return matches[0]
            return matches

        return None

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

    def _store(self, run_id: str, contents: dict, name: Union[Path, str], insert=False):
        """
        Store the contents against the name in the folder.

        Args:
            run_id (str): The run id
            contents (dict): The dict to store
            name (str): The name to store as
        """
        if insert:
            name = self.log_folder_with_run_id(run_id=run_id) / name

        utils.safe_make_dir(self.log_folder_with_run_id(run_id=run_id))

        with open(self.safe_suffix_json(name), "w") as fw:
            json.dump(contents, fw, ensure_ascii=True, indent=4)

    def _retrieve(self, name: Union[str, Path]) -> dict:
        """
        Does the job of retrieving from the folder.

        Args:
            name (str): the name of the file to retrieve

        Returns:
            dict: The contents
        """
        contents: dict = {}

        with open(self.safe_suffix_json(name), "r") as fr:
            contents = json.load(fr)

        return contents
