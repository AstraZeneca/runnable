from cloudpathlib import S3Path

from extensions.catalog.any_path import AnyPathCatalog


class S3Catalog(AnyPathCatalog):
    service_name: str = "s3"

    def get_path(self, path: str) -> S3Path:
        # TODO: Might need to assert the credentials are set
        return S3Path(path)
