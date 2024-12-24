from runnable.catalog import is_catalog_out_of_sync


def test_is_catalog_out_of_sync_returns_true_for_empty_synced_catalogs():
    assert is_catalog_out_of_sync(1, []) is True


def test_is_catalog_out_of_sync_returns_false_for_same_objects():
    class MockCatalog:
        catalog_relative_path = None
        data_hash = None

    catalog_item = MockCatalog()
    catalog_item.catalog_relative_path = "path"
    catalog_item.data_hash = "hash"

    synced_catalog = [catalog_item]
    assert is_catalog_out_of_sync(catalog_item, synced_catalog) is False


def test_is_catalog_out_of_sync_returns_true_for_different_hash():
    class MockCatalog:
        catalog_relative_path = None
        data_hash = None

    catalog_item1 = MockCatalog()
    catalog_item1.catalog_relative_path = "path"
    catalog_item1.data_hash = "hash"

    catalog_item2 = MockCatalog()
    catalog_item2.catalog_relative_path = "path"
    catalog_item2.data_hash = "not-hash"

    synced_catalog = [catalog_item1]
    assert is_catalog_out_of_sync(catalog_item2, synced_catalog) is True


def test_is_catalog_out_of_sync_returns_true_for_different_paths():
    class MockCatalog:
        catalog_relative_path = None
        data_hash = None

    catalog_item1 = MockCatalog()
    catalog_item1.catalog_relative_path = "path"
    catalog_item1.data_hash = "hash"

    catalog_item2 = MockCatalog()
    catalog_item2.catalog_relative_path = "path1"
    catalog_item2.data_hash = "hash"

    synced_catalog = [catalog_item1]
    assert is_catalog_out_of_sync(catalog_item2, synced_catalog) is True
