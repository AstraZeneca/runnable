from typing import Type

from cloudpathlib import S3Path

from extensions.catalog.any_path import AnyPathCatalog


class S3Catalog(AnyPathCatalog):
    service_name: str = "s3"

    def get_path_client(self) -> Type[S3Path]:
        return S3Path
