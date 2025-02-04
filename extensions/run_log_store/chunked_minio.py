import json
import logging
from functools import lru_cache
from string import Template
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

    def get_matches(
        self, run_id: str, name: str, multiple_allowed: bool = False
    ) -> None | str | list[str]:
        """
        Get contents of files matching the pattern name*

        Args:
            run_id (str): The run id
            name (str): The suffix of the file name to check in the run log store.
        """
        run_log_bucket = self.get_run_log_bucket()
        run_log_bucket.mkdir(parents=True, exist_ok=True)

        sub_name = Template(name).safe_substitute({"creation_time": ""})
        matches = list(run_log_bucket.glob(f"{sub_name}*"))

        if matches:
            if not multiple_allowed:
                if len(matches) > 1:
                    msg = f"Multiple matches found for {name} while multiple is not allowed"
                    raise Exception(msg)
                return str(matches[0])
            return [str(match) for match in matches]

        return None

    def _store(self, run_id: str, contents: dict, name: str, insert=False):
        """
        Store the contents against the name in the folder.

        Args:
            run_id (str): The run id
            contents (dict): The dict to store
            name (str): The name to store as
        """

        if insert:
            name = str(self.get_run_log_bucket() / name)

        self.get_run_log_bucket().mkdir(parents=True, exist_ok=True)
        obj = S3Path(
            name,
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
            name (str): the name of the file to retrieve

        Returns:
            dict: The contents
        """

        obj = S3Path(
            name,
            client=get_minio_client(
                self.endpoint_url,
                self.aws_access_key_id.get_secret_value(),
                self.aws_secret_access_key.get_secret_value(),
            ),
        )

        run_log_text = json.loads(obj.read_text())
        return run_log_text
