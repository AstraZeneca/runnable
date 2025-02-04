import json
import logging
from functools import lru_cache
from typing import Any, Dict

from cloudpathlib import S3Client, S3Path
from pydantic import Field, SecretStr

from extensions.run_log_store.any_path import AnyPathRunLogStore
from runnable import defaults
from runnable.datastore import RunLog

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


class MinioRunLogStore(AnyPathRunLogStore):
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

    service_name: str = "minio"

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

    def write_to_path(self, run_log: RunLog):
        """
        Write the run log to the folder

        Args:
            run_log (RunLog): The run log to be added to the database
        """
        run_log_bucket = self.get_run_log_bucket()
        run_log_bucket.mkdir(parents=True, exist_ok=True)

        run_log_object = run_log_bucket / f"{run_log.run_id}.json"
        run_log_object.write_text(
            json.dumps(run_log.model_dump_json(), ensure_ascii=True, indent=4)
        )

    def read_from_path(self, run_id: str) -> RunLog:
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
        run_log_bucket = self.get_run_log_bucket()

        run_log_object = run_log_bucket / f"{run_id}.json"

        run_log_text = json.loads(run_log_object.read_text())
        run_log = RunLog(**json.loads(run_log_text))

        return run_log
