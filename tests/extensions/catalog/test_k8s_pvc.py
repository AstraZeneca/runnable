from runnable.extensions.catalog.k8s_pvc.implementation import K8sPVCatalog


def test_get_catalog_location_returns_location_relative_to_mount_path():
    test_catalog = K8sPVCatalog(
        catalog_location="test_location",
        mount_path="/here",
        persistent_volume_name="test",
    )

    assert test_catalog.get_catalog_location() == "/here/test_location"
