import json
import logging
from functools import lru_cache
from typing import Any, Dict

from cloudpathlib import S3Client, S3Path
from pydantic import Field, SecretStr

from extensions.run_log_store.generic_chunked import ChunkedRunLogStore
from runnable import defaults

logger = logging.getLogger(defaults.LOGGER_NAME)


@lru_cache
def get_minio_client(
    endpoint_url: str, aws_access_key_id: str, aws_secret_access_key: str
) -> S3Client:
    return S3Client(
        endpoint_url=endpoint_url,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )


class ChunkedMinioRunLogStore(ChunkedRunLogStore):
    """
    File system run log store but chunks the run log into thread safe chunks.
    This enables executions to be parallel.
    """

    service_name: str = "chunked-minio"
    endpoint_url: str = Field(default="http://localhost:9002")
    aws_access_key_id: SecretStr = SecretStr(secret_value="minioadmin")
    aws_secret_access_key: SecretStr = SecretStr(secret_value="minioadmin")
    bucket: str = Field(default="runnable/run-logs")

    def get_summary(self) -> Dict[str, Any]:
        summary = {
            "Type": self.service_name,
            "Location": f"{self.endpoint_url}/{self.bucket}",
        }

        return summary

    def get_run_log_bucket(self) -> S3Path:
        run_id = self._context.run_id

        return S3Path(
            f"s3://{self.bucket}/{run_id}/",
            client=get_minio_client(
                self.endpoint_url,
                self.aws_access_key_id.get_secret_value(),
                self.aws_secret_access_key.get_secret_value(),
            ),
        )

    def _exists(self, run_id: str, name: str) -> bool:
        """
        Check if a file exists in the Minio bucket.

        Args:
            run_id (str): The run id
            name (str): The exact file name to check

        Returns:
            bool: True if file exists, False otherwise
        """
        run_log_bucket = self.get_run_log_bucket()
        file_path = run_log_bucket / name
        return file_path.exists()

    def _list_branch_logs(self, run_id: str) -> list[str]:
        """
        List all branch log file names for a run_id.

        Args:
            run_id (str): The run id

        Returns:
            list[str]: List of branch log file names (e.g., ["BranchLog-map.1", "BranchLog-map.2"])
        """
        run_log_bucket = self.get_run_log_bucket()
        if not run_log_bucket.exists():
            return []

        # Find all files starting with "BranchLog-"
        branch_files = list(run_log_bucket.glob("BranchLog-*"))
        # Return file names without path (just the name)
        return [f.name for f in branch_files]

    def _store(self, run_id: str, contents: dict, name: str, insert=False):
        """
        Store the contents against the name in the folder.

        Args:
            run_id (str): The run id
            contents (dict): The dict to store
            name (str): The name to store as (without path)
            insert (bool): Whether this is a new insert (unused, kept for compatibility)
        """
        run_log_bucket = self.get_run_log_bucket()
        run_log_bucket.mkdir(parents=True, exist_ok=True)

        file_path = str(run_log_bucket / name)
        obj = S3Path(
            file_path,
            client=get_minio_client(
                self.endpoint_url,
                self.aws_access_key_id.get_secret_value(),
                self.aws_secret_access_key.get_secret_value(),
            ),
        )

        obj.write_text(json.dumps(contents, ensure_ascii=True, indent=4))

    def _retrieve(self, run_id: str, name: str) -> dict:
        """
        Does the job of retrieving from the folder.

        Args:
            run_id (str): The run id
            name (str): the name of the file to retrieve (without path)

        Returns:
            dict: The contents
        """
        run_log_bucket = self.get_run_log_bucket()
        file_path = str(run_log_bucket / name)

        obj = S3Path(
            file_path,
            client=get_minio_client(
                self.endpoint_url,
                self.aws_access_key_id.get_secret_value(),
                self.aws_secret_access_key.get_secret_value(),
            ),
        )

        run_log_text = json.loads(obj.read_text())
        return run_log_text
