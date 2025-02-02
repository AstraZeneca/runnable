from cloudpathlib import S3Client, S3Path

from extensions.catalog.any_path import AnyPathCatalog


class MinioCatalog(AnyPathCatalog):
    service_name: str = "minio"
    bucket: str = "runnable"

    def get_path(self, path: str) -> S3Path:
        # TODO: Might need to assert the credentials are set
        client = S3Client(
            endpoint_url="http://localhost:9002",
            aws_access_key_id="minioadmin",
            aws_secret_access_key="minioadmin",
        )
        return client.CloudPath(f"s3://{self.bucket}/{path}")
