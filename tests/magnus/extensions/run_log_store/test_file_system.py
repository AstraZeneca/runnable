import pytest

from magnus.extensions.run_log_store.file_system.implementation import FileSystemRunLogstore
import magnus.extensions.run_log_store.file_system.implementation as implementation
from magnus import defaults
from magnus import exceptions


def test_file_system_run_log_store_log_folder_name_defaults_if_not_provided():
    run_log_store = FileSystemRunLogstore()

    assert run_log_store.log_folder_name == defaults.LOG_LOCATION_FOLDER


def test_file_system_run_log_store_log_folder_name_if__provided():
    run_log_store = FileSystemRunLogstore(log_folder="test")

    assert run_log_store.log_folder_name == "test"


def test_file_system_run_log_store_write_to_folder_makes_dir_if_not_present(mocker, monkeypatch):
    mock_safe_make_dir = mocker.MagicMock()
    monkeypatch.setattr(implementation.utils, "safe_make_dir", mock_safe_make_dir)

    mock_json = mocker.MagicMock()
    mock_path = mocker.MagicMock()
    monkeypatch.setattr(implementation, "json", mock_json)
    monkeypatch.setattr(implementation, "Path", mock_path)

    mock_run_log = mocker.MagicMock()
    mock_dict = mocker.MagicMock()
    mock_run_log.dict = mock_dict

    run_log_store = FileSystemRunLogstore()
    run_log_store.write_to_folder(run_log=mock_run_log)

    mock_safe_make_dir.assert_called_once_with(run_log_store.log_folder_name)
    assert mock_dict.call_count == 1


def test_file_system_run_log_store_get_from_folder_raises_exception_if_folder_not_present(mocker, monkeypatch):
    mock_path = mocker.MagicMock()
    monkeypatch.setattr(implementation, "Path", mocker.MagicMock(return_value=mock_path))

    mock_path.__truediv__.return_value = mock_path

    mock_path.exists.return_value = False

    run_log_store = implementation.FileSystemRunLogstore()

    with pytest.raises(FileNotFoundError):
        run_log_store.get_from_folder(run_id="test")


def test_file_system_run_log_store_get_from_folder_returns_run_log_from_file_contents(mocker, monkeypatch):
    mock_path = mocker.MagicMock()
    monkeypatch.setattr(implementation, "Path", mocker.MagicMock(return_value=mock_path))

    mock_path.__truediv__.return_value = mock_path
    mock_path.exists.return_value = True

    mock_json = mocker.MagicMock()
    monkeypatch.setattr(implementation, "json", mock_json)
    mock_json.load.return_value = {"run_id": "test"}

    run_log_store = implementation.FileSystemRunLogstore()
    run_log = run_log_store.get_from_folder(run_id="does not matter")

    assert run_log.run_id == "test"


def test_file_system_run_log_store_create_run_log_writes_to_folder(mocker, monkeypatch):
    mock_write_to_folder = mocker.MagicMock()

    monkeypatch.setattr(implementation.FileSystemRunLogstore, "write_to_folder", mock_write_to_folder)

    run_log_store = implementation.FileSystemRunLogstore()
    run_log = run_log_store.create_run_log(run_id="test random")

    mock_write_to_folder.assert_called_once_with(run_log)

    assert run_log.run_id == "test random"


def test_file_system_run_log_store_create_run_log_raises_exception_if_present(mocker, monkeypatch):
    mock_write_to_folder = mocker.MagicMock()
    mock_get_run_log_by_id = mocker.MagicMock(return_value="existing")

    monkeypatch.setattr(implementation.FileSystemRunLogstore, "write_to_folder", mock_write_to_folder)
    monkeypatch.setattr(implementation.FileSystemRunLogstore, "get_run_log_by_id", mock_get_run_log_by_id)

    run_log_store = implementation.FileSystemRunLogstore()
    with pytest.raises(exceptions.RunLogExistsError):
        run_log_store.create_run_log(run_id="test random")


def test_file_system_run_log_store_get_run_log_by_id_raises_exception_if_get_from_folder_fails(mocker, monkeypatch):
    mock_get_from_folder = mocker.MagicMock()
    mock_get_from_folder.side_effect = FileNotFoundError()

    monkeypatch.setattr(implementation.FileSystemRunLogstore, "get_from_folder", mock_get_from_folder)

    run_log_store = implementation.FileSystemRunLogstore()
    with pytest.raises(exceptions.RunLogNotFoundError):
        run_log_store.get_run_log_by_id(run_id="should fail")


def test_file_system_run_log_store_get_run_log_by_id_returns_run_log_from_get_from_folder(mocker, monkeypatch):
    mock_get_from_folder = mocker.MagicMock()
    mock_get_from_folder.return_value = "I am a run log"

    monkeypatch.setattr(implementation.FileSystemRunLogstore, "get_from_folder", mock_get_from_folder)

    run_log_store = implementation.FileSystemRunLogstore()

    run_log = run_log_store.get_run_log_by_id(run_id="test")

    assert run_log == "I am a run log"


def test_file_system_run_log_store_put_run_log_writes_to_folder(mocker, monkeypatch):
    mock_write_to_folder = mocker.MagicMock()

    monkeypatch.setattr(implementation.FileSystemRunLogstore, "write_to_folder", mock_write_to_folder)

    run_log_store = implementation.FileSystemRunLogstore()
    mock_run_log = mocker.MagicMock()
    run_log_store.put_run_log(run_log=mock_run_log)

    mock_write_to_folder.assert_called_once_with(mock_run_log)
