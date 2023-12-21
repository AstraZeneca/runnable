import logging

from magnus.extensions.executor.argo import integration


def test_file_system_run_log_store_is_allowed_with_a_warning(caplog):
    test_integration = integration.FileSystemRunLogStore(executor="test", integration_service="test")

    with caplog.at_level(logging.WARNING, logger="magnus"):
        test_integration.validate()

    assert "Argo cannot run work with file-system run log store. Unless you " in caplog.text


def test_chunked_file_system_run_log_store_is_allowed_with_a_warning(caplog):
    test_integration = integration.ChunkedFileSystemRunLogStore(executor="test", integration_service="test")

    with caplog.at_level(logging.WARNING, logger="magnus"):
        test_integration.validate()

    assert "Argo cannot run work with chunked file-system run log store" in caplog.text


def test_file_sytem_catalog_is_allowed_with_a_warning(caplog):
    test_integration = integration.FileSystemCatalog(executor="test", integration_service="test")

    with caplog.at_level(logging.WARNING, logger="magnus"):
        test_integration.validate()

    assert "Argo cannot run work with file-system run log store. Unless you " in caplog.text
