import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

from cloudpathlib import CloudPath, S3Client, S3Path
from pydantic import Field, SecretStr

from extensions.catalog.any_path import AnyPathCatalog
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


class MinioCatalog(AnyPathCatalog):
    service_name: str = "minio"

    endpoint_url: str = Field(default="http://localhost:9002")
    aws_access_key_id: SecretStr = SecretStr(secret_value="minioadmin")
    aws_secret_access_key: SecretStr = SecretStr(secret_value="minioadmin")
    bucket: str = "runnable"

    def get_additional_identifiers(self, file: Path | CloudPath) -> dict[str, str]:
        """
        Get the implementation specific id for the file
        """
        assert isinstance(file, S3Path)

        return {"etag": file.etag or "None"}

    def get_summary(self) -> dict[str, Any]:
        return {
            "service_name": self.service_name,
            "compute_data_folder": self.compute_data_folder,
            "endpoint_url": self.endpoint_url,
            "bucket": self.bucket,
        }

    def get_log_location(self) -> Path | CloudPath:
        run_id = self._context.run_id

        return S3Path(
            f"s3://{self.bucket}/{run_id}/{self.compute_data_folder}".strip("."),
            client=get_minio_client(
                self.endpoint_url,
                self.aws_access_key_id.get_secret_value(),
                self.aws_secret_access_key.get_secret_value(),
            ),
        )

    def get_catalog_location(self) -> S3Path:
        if not self.store_copy:
            return S3Path(
                f"s3://{self.bucket}".strip("."),
                client=get_minio_client(
                    self.endpoint_url,
                    self.aws_access_key_id.get_secret_value(),
                    self.aws_secret_access_key.get_secret_value(),
                ),
            )

        run_id = self._context.run_id

        return S3Path(
            f"s3://{self.bucket}/{run_id}/{self.compute_data_folder}".strip("."),
            client=get_minio_client(
                self.endpoint_url,
                self.aws_access_key_id.get_secret_value(),
                self.aws_secret_access_key.get_secret_value(),
            ),
        )

    def download_from_catalog(self, file: Path | CloudPath) -> CloudPath:
        assert isinstance(file, S3Path)

        relative_file_path = file.relative_to(self.get_catalog_location())

        file_to_download = Path(self.compute_data_folder) / relative_file_path
        file_to_download.parent.mkdir(parents=True, exist_ok=True)

        file.download_to(file_to_download)

        return file

    def upload_log_file(self, file: Path) -> None:
        run_catalog = self.get_log_location()
        run_catalog.mkdir(parents=True, exist_ok=True)
        file_in_cloud = run_catalog / file
        assert isinstance(file_in_cloud, S3Path)
        file_in_cloud.upload_from(file)

    def upload_to_catalog(self, file: Path) -> CloudPath:
        run_catalog = self.get_catalog_location()

        relative_file_path = file.relative_to(self.compute_data_folder)
        (run_catalog / relative_file_path.parent).mkdir(parents=True, exist_ok=True)

        file_in_cloud = run_catalog / file
        assert isinstance(file_in_cloud, S3Path)
        file_in_cloud.upload_from(file)

        return file_in_cloud
