from pathlib import Path

from magnus.extensions.catalog.file_system import integration


def test_configure_for_traversal_populates_volumes(mocker, monkeypatch):
    mock_local_container = mocker.MagicMock()
    monkeypatch.setattr(integration, "LocalContainerExecutor", mock_local_container)

    mock_executor = mocker.MagicMock()
    mock_executor._volumes = {}
    mock_executor._container_catalog_location = "this_location"

    mock_fs_catalog = mocker.MagicMock()
    mock_fs_catalog.catalog_location = "catalog_location"

    test_integration = integration.LocalContainerComputeFileSystemCatalog(mock_executor, mock_fs_catalog)
    test_integration.configure_for_traversal()

    assert mock_executor._volumes == {str(Path("catalog_location").resolve()): {"bind": "this_location", "mode": "rw"}}


def test_configure_for_execution_assigns_catalog_location_within_container(mocker, monkeypatch):
    mock_local_container = mocker.MagicMock()
    monkeypatch.setattr(integration, "LocalContainerExecutor", mock_local_container)

    mock_executor = mocker.MagicMock()
    mock_executor._container_catalog_location = "this_location"

    mock_fs_catalog = mocker.MagicMock()

    test_integration = integration.LocalContainerComputeFileSystemCatalog(mock_executor, mock_fs_catalog)
    test_integration.configure_for_execution()

    assert mock_fs_catalog.catalog_location == "this_location"
