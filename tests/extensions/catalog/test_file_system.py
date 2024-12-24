import os
import tempfile

import pytest

from extensions.catalog import file_system as module
from extensions.catalog.file_system import FileSystemCatalog
from runnable import defaults


def test_file_system_catalog_inits_default_values_if_none_config():
    catalog_handler = FileSystemCatalog()
    assert catalog_handler.compute_data_folder == defaults.COMPUTE_DATA_FOLDER
    assert catalog_handler.catalog_location == defaults.CATALOG_LOCATION_FOLDER


def test_file_system_catalog_get_catalog_location_defaults_if_location_not_provided(
    monkeypatch, mocker
):
    catalog_handler = FileSystemCatalog()

    assert catalog_handler.catalog_location == defaults.CATALOG_LOCATION_FOLDER


def test_file_system_catalog_catalog_location_returns_config_catalog_location_if_provided(
    monkeypatch, mocker
):
    catalog_handler = FileSystemCatalog(catalog_location="this")

    assert catalog_handler.catalog_location == "this"


def test_file_system_catalog_get_raises_exception_if_catalog_does_not_exist(
    monkeypatch, mocker
):
    def mock_does_dir_exist(dir_name):
        if dir_name == "this_compute_folder":
            return True
        return False

    monkeypatch.setattr(module.utils, "does_dir_exist", mock_does_dir_exist)

    catalog_handler = FileSystemCatalog(catalog_location="this_location")
    with pytest.raises(Exception, match="Expected Catalog to be present at"):
        catalog_handler.get(
            "testing", run_id="dummy_run_id", compute_data_folder="this_compute_folder"
        )


def test_file_system_catalog_get_copies_files_from_catalog_to_compute_folder_with_all(
    mocker, monkeypatch
):
    mock_run_store = mocker.MagicMock()
    mock_context = mocker.MagicMock()
    mock_context.run_log_store = mock_run_store

    mocker.patch(
        "runnable.catalog.BaseCatalog._context",
        new_callable=mocker.PropertyMock,
        return_value=mock_context,
    )

    with tempfile.TemporaryDirectory() as catalog_location:
        with tempfile.TemporaryDirectory(dir=".") as compute_folder:
            catalog_location_path = module.Path(catalog_location)
            run_id = "testing"
            module.Path(catalog_location_path / run_id / compute_folder).mkdir(
                parents=True
            )
            with open(
                module.Path(catalog_location)
                / run_id
                / compute_folder
                / "catalog_file",
                "w",
            ) as fw:
                fw.write("hello")

            catalog_handler = FileSystemCatalog()
            catalog_handler.catalog_location = catalog_location

            catalog_handler.get(name="**/*", run_id=run_id)

            _, _, files = next(os.walk(compute_folder))

            assert len(list(files)) == 1


def test_file_system_catalog_get_copies_files_from_catalog_to_compute_folder_with_pattern(
    mocker, monkeypatch
):
    mock_run_store = mocker.MagicMock()
    mock_context = mocker.MagicMock()
    mock_context.run_log_store = mock_run_store

    mocker.patch(
        "runnable.catalog.BaseCatalog._context",
        new_callable=mocker.PropertyMock,
        return_value=mock_context,
    )

    with tempfile.TemporaryDirectory() as catalog_location:
        with tempfile.TemporaryDirectory(dir=".") as compute_folder:
            catalog_location_path = module.Path(catalog_location)
            run_id = "testing"
            module.Path(catalog_location_path / run_id / compute_folder).mkdir(
                parents=True
            )
            with open(
                module.Path(catalog_location)
                / run_id
                / compute_folder
                / "catalog_file",
                "w",
            ) as fw:
                fw.write("hello")

            with open(
                module.Path(catalog_location) / run_id / compute_folder / "not_catalog",
                "w",
            ) as fw:
                fw.write("hello")

            catalog_handler = FileSystemCatalog(catalog_location=catalog_location)
            catalog_handler.get(name="**/catalog*", run_id=run_id)

            _, _, files = next(os.walk(compute_folder))

            assert len(list(files)) == 1


def test_file_system_catalog_put_copies_files_from_compute_folder_to_catalog_if_synced_changed_all(
    mocker, monkeypatch
):
    monkeypatch.setattr(
        module, "is_catalog_out_of_sync", mocker.MagicMock(return_value=True)
    )
    mock_run_store = mocker.MagicMock()
    mock_context = mocker.MagicMock()
    mock_context.run_log_store = mock_run_store

    mocker.patch(
        "runnable.catalog.BaseCatalog._context",
        new_callable=mocker.PropertyMock,
        return_value=mock_context,
    )

    with tempfile.TemporaryDirectory() as catalog_location:
        with tempfile.TemporaryDirectory(dir=".") as compute_folder:
            catalog_location_path = module.Path(catalog_location)
            run_id = "testing"
            module.Path(catalog_location_path / run_id).mkdir(parents=True)

            with open(module.Path(compute_folder) / "catalog_file", "w") as fw:
                fw.write("hello")

            catalog_handler = FileSystemCatalog(catalog_location=catalog_location)
            catalog_handler.put(name=str(compute_folder) + "/*", run_id=run_id)

            _, _, files = next(os.walk(catalog_location_path / run_id / compute_folder))

            assert len(list(files)) == 1


def test_file_system_catalog_put_copies_files_from_compute_folder_to_catalog_if_synced_changed_pattern(
    mocker, monkeypatch
):
    monkeypatch.setattr(
        module, "is_catalog_out_of_sync", mocker.MagicMock(return_value=True)
    )
    mock_run_store = mocker.MagicMock()
    mock_context = mocker.MagicMock()
    mock_context.run_log_store = mock_run_store

    mocker.patch(
        "runnable.catalog.BaseCatalog._context",
        new_callable=mocker.PropertyMock,
        return_value=mock_context,
    )
    with tempfile.TemporaryDirectory() as catalog_location:
        with tempfile.TemporaryDirectory(dir=".") as compute_folder:
            catalog_location_path = module.Path(catalog_location)
            run_id = "testing"
            module.Path(catalog_location_path / run_id).mkdir(parents=True)
            with open(module.Path(compute_folder) / "catalog_file", "w") as fw:
                fw.write("hello")

            with open(module.Path(compute_folder) / "not_catalog_file", "w") as fw:
                fw.write("hello")

            catalog_handler = FileSystemCatalog(catalog_location=catalog_location)

            catalog_handler.put(name=str(compute_folder) + "/catalog*", run_id=run_id)

            _, _, files = next(os.walk(catalog_location_path / run_id / compute_folder))

            assert len(list(files)) == 1


def test_file_system_catalog_put_copies_files_from_compute_folder_to_catalog_if_synced_true(
    mocker, monkeypatch
):
    monkeypatch.setattr(
        module, "is_catalog_out_of_sync", mocker.MagicMock(return_value=False)
    )
    mock_run_store = mocker.MagicMock()
    mock_context = mocker.MagicMock()
    mock_context.run_log_store = mock_run_store

    mocker.patch(
        "runnable.catalog.BaseCatalog._context",
        new_callable=mocker.PropertyMock,
        return_value=mock_context,
    )

    with tempfile.TemporaryDirectory() as catalog_location:
        with tempfile.TemporaryDirectory(dir=".") as compute_folder:
            catalog_location_path = module.Path(catalog_location)
            run_id = "testing"
            module.Path(catalog_location_path / run_id).mkdir(parents=True)
            with open(module.Path(compute_folder) / "catalog_file", "w") as fw:
                fw.write("hello")

            with open(module.Path(compute_folder) / "not_catalog_file", "w") as fw:
                fw.write("hello")

            catalog_handler = FileSystemCatalog(catalog_location=catalog_location)

            catalog_handler.put(name=str(compute_folder) + "/*", run_id=run_id)

            with pytest.raises(FileNotFoundError):
                _ = os.listdir(catalog_location_path / run_id / compute_folder)
                assert True


def test_file_system_catalog_put_uses_compute_folder_by_default(monkeypatch, mocker):
    mock_safe_make_dir = mocker.MagicMock()
    monkeypatch.setattr(module.utils, "safe_make_dir", mock_safe_make_dir)

    mock_does_dir_exist = mocker.MagicMock(side_effect=Exception())
    monkeypatch.setattr(module.utils, "does_dir_exist", mock_does_dir_exist)

    catalog_handler = FileSystemCatalog(catalog_location="this_location")
    with pytest.raises(Exception):
        catalog_handler.put("testing", run_id="dummy_run_id")

    mock_does_dir_exist.assert_called_once_with(module.Path("."))


def test_file_system_catalog_put_uses_compute_folder_provided(monkeypatch, mocker):
    mock_safe_make_dir = mocker.MagicMock()
    monkeypatch.setattr(module.utils, "safe_make_dir", mock_safe_make_dir)

    mock_does_dir_exist = mocker.MagicMock(side_effect=Exception())
    monkeypatch.setattr(module.utils, "does_dir_exist", mock_does_dir_exist)

    catalog_handler = FileSystemCatalog(catalog_location="this_location")
    with pytest.raises(Exception):
        catalog_handler.put(
            "testing", run_id="dummy_run_id", compute_data_folder="not_data"
        )

    mock_does_dir_exist.assert_called_once_with(module.Path("not_data"))


def test_file_system_catalog_put_raises_exception_if_compute_data_folder_does_not_exist(
    monkeypatch, mocker
):
    mock_safe_make_dir = mocker.MagicMock()
    monkeypatch.setattr(module.utils, "safe_make_dir", mock_safe_make_dir)

    mock_does_dir_exist = mocker.MagicMock(return_value=False)
    monkeypatch.setattr(module.utils, "does_dir_exist", mock_does_dir_exist)

    catalog_handler = FileSystemCatalog(catalog_location="this_location")
    with pytest.raises(Exception):
        catalog_handler.put(
            "testing", run_id="dummy_run_id", compute_data_folder="this_compute_folder"
        )


def test_file_system_catalog_put_creates_catalog_location_using_run_id(
    monkeypatch, mocker
):
    mock_safe_make_dir = mocker.MagicMock()
    monkeypatch.setattr(module.utils, "safe_make_dir", mock_safe_make_dir)

    mock_does_dir_exist = mocker.MagicMock(side_effect=Exception())
    monkeypatch.setattr(module.utils, "does_dir_exist", mock_does_dir_exist)

    catalog_handler = FileSystemCatalog(catalog_location="this_location")

    with pytest.raises(Exception):
        catalog_handler.put("testing", run_id="dummy_run_id")

    mock_safe_make_dir.assert_called_once_with(
        module.Path("this_location") / "dummy_run_id"
    )


def test_file_system_sync_between_runs_raises_exception_if_previous_catalog_does_not_exist(
    monkeypatch, mocker
):
    mock_safe_make_dir = mocker.MagicMock()
    monkeypatch.setattr(module.utils, "safe_make_dir", mock_safe_make_dir)

    mock_does_dir_exist = mocker.MagicMock(return_value=False)
    monkeypatch.setattr(module.utils, "does_dir_exist", mock_does_dir_exist)

    catalog_handler = FileSystemCatalog(catalog_location="this_location")
    with pytest.raises(Exception):
        catalog_handler.sync_between_runs("previous", "current")
