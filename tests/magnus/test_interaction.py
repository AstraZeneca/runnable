import os
import json
import logging

import pytest

from magnus import (
    defaults,  # pylint: disable=import-error
    exceptions,  # pylint: disable=import-error
    interaction,  # pylint: disable=import-error
)


@pytest.fixture(autouse=True)
def mock_context(monkeypatch, mocker, request):
    if "noautofixt" in request.keywords:
        yield None
        return
    mc = mocker.MagicMock()
    monkeypatch.setattr(interaction, "context", mc)
    yield


def test_track_this_adds_values_to_environ():
    interaction.track_this(a="b")
    assert defaults.TRACK_PREFIX + "a" in os.environ
    del os.environ[defaults.TRACK_PREFIX + "a"]


def test_track_this_adds_multiple_values_to_environ():
    interaction.track_this(a="b", b="a")
    assert defaults.TRACK_PREFIX + "a" in os.environ
    assert defaults.TRACK_PREFIX + "b" in os.environ
    del os.environ[defaults.TRACK_PREFIX + "a"]
    del os.environ[defaults.TRACK_PREFIX + "b"]


def test_track_this_ignores_step_if_zero():
    interaction.track_this(a="b", b="a")
    assert defaults.TRACK_PREFIX + "a" in os.environ
    assert defaults.TRACK_PREFIX + "b" in os.environ
    del os.environ[defaults.TRACK_PREFIX + "a"]
    del os.environ[defaults.TRACK_PREFIX + "b"]


def test_track_this_adds_step_if_non_zero():
    interaction.track_this(a="b", b="a", step=1)
    assert defaults.TRACK_PREFIX + "1_" + "a" in os.environ
    assert defaults.TRACK_PREFIX + "1_" + "b" in os.environ
    del os.environ[defaults.TRACK_PREFIX + "1_" + "a"]
    del os.environ[defaults.TRACK_PREFIX + "1_" + "b"]


def test_store_paramenter_adds_values_to_environ():
    interaction.store_parameter(a="b")
    assert defaults.PARAMETER_PREFIX + "a" in os.environ
    del os.environ[defaults.PARAMETER_PREFIX + "a"]


def test_store_parameter_adds_multiple_values_to_environ():
    interaction.store_parameter(a="b", b="a")
    assert defaults.PARAMETER_PREFIX + "a" in os.environ
    assert defaults.PARAMETER_PREFIX + "b" in os.environ
    del os.environ[defaults.PARAMETER_PREFIX + "a"]
    del os.environ[defaults.PARAMETER_PREFIX + "b"]


def test_store_parameter_updates_if_present_and_asked():
    os.environ[defaults.PARAMETER_PREFIX + "a"] = "b"
    os.environ[defaults.PARAMETER_PREFIX + "b"] = "a"
    interaction.store_parameter(a="c", b="d")
    assert json.loads(os.environ[defaults.PARAMETER_PREFIX + "a"]) == "c"
    assert json.loads(os.environ[defaults.PARAMETER_PREFIX + "b"]) == "d"

    del os.environ[defaults.PARAMETER_PREFIX + "a"]
    del os.environ[defaults.PARAMETER_PREFIX + "b"]


def test_get_parameter_returns_all_parameters_if_no_key_provided(monkeypatch, mocker):
    monkeypatch.setattr(interaction.parameters, "get_user_set_parameters", mocker.MagicMock(return_value="this"))

    assert interaction.get_parameter() == "this"


def test_get_parameter_returns_parameters_if_provided(mocker, monkeypatch):
    monkeypatch.setattr(
        interaction.parameters, "get_user_set_parameters", mocker.MagicMock(return_value={"this": "that"})
    )

    assert interaction.get_parameter("this") == "that"


def test_get_parameter_returns_parameters_raises_exception_if_key_not_found(mocker, monkeypatch):
    monkeypatch.setattr(
        interaction.parameters, "get_user_set_parameters", mocker.MagicMock(return_value={"this": "that"})
    )

    with pytest.raises(Exception):
        interaction.get_parameter("this1")


def test_get_secret_delegates_to_secrets_handler_get(mocker, monkeypatch):
    mock_context = mocker.MagicMock()
    mock_secrets_handler = mocker.MagicMock()

    mock_context.run_context.secrets_handler = mock_secrets_handler

    monkeypatch.setattr(interaction, "context", mock_context)

    mock_secrets_handler.get.return_value = "test"

    assert interaction.get_secret("secret") == "test"


def test_get_secret_raises_exception_if_secrets_handler_raises(mocker, monkeypatch):
    mock_context = mocker.MagicMock()
    mock_secrets_handler = mocker.MagicMock()

    mock_context.run_context.secrets_handler = mock_secrets_handler

    monkeypatch.setattr(interaction, "context", mock_context)

    mock_secrets_handler.get.side_effect = exceptions.SecretNotFoundError("test", "test")
    with pytest.raises(exceptions.SecretNotFoundError):
        assert interaction.get_secret("secret")


def test_get_from_catalog_delegates_to_catalog_handler(mocker, monkeypatch):
    mock_context = mocker.MagicMock()
    mock_catalog_handler = mocker.MagicMock()

    mock_context.run_context.catalog_handler = mock_catalog_handler

    mock_catalog_handler_get = mocker.MagicMock()
    mock_catalog_handler.get = mock_catalog_handler_get
    mock_context.run_context.run_id = "RUN_ID"

    mock_catalog_handler.compute_data_folder = "compute_folder"
    monkeypatch.setattr(interaction, "context", mock_context)

    interaction.get_from_catalog("this")

    mock_catalog_handler_get.assert_called_once_with("this", compute_data_folder="compute_folder", run_id="RUN_ID")


def test_get_from_catalog_uses_destination_folder(mocker, monkeypatch):
    mock_context = mocker.MagicMock()
    mock_catalog_handler = mocker.MagicMock()

    mock_context.run_context.catalog_handler = mock_catalog_handler

    mock_catalog_handler_get = mocker.MagicMock()
    mock_catalog_handler.get = mock_catalog_handler_get
    mock_context.run_context.run_id = "RUN_ID"

    mock_catalog_handler.compute_data_folder = "compute_folder"
    monkeypatch.setattr(interaction, "context", mock_context)

    interaction.get_from_catalog("this", destination_folder="use_this_folder")

    mock_catalog_handler_get.assert_called_once_with("this", compute_data_folder="use_this_folder", run_id="RUN_ID")


def test_get_from_catalog_raises_warning_if_no_context_step_log(mocker, monkeypatch, caplog):
    mock_context = mocker.MagicMock()
    mock_catalog_handler = mocker.MagicMock()

    mock_context.run_context.catalog_handler = mock_catalog_handler
    mock_context.run_context.executor._context_step_log = None

    mock_catalog_handler_get = mocker.MagicMock()
    mock_catalog_handler.get = mock_catalog_handler_get
    mock_context.run_context.run_id = "RUN_ID"

    mock_catalog_handler.compute_data_folder = "compute_folder"
    monkeypatch.setattr(interaction, "context", mock_context)

    with caplog.at_level(logging.WARNING):
        interaction.get_from_catalog("this")

    assert "Step log context was not found during interaction" in caplog.text

    mock_catalog_handler_get.assert_called_once_with("this", compute_data_folder="compute_folder", run_id="RUN_ID")


def test_put_in_catalog_raises_warning_if_no_catalog_was_obtained(mocker, monkeypatch, caplog):
    mock_context = mocker.MagicMock()
    mock_catalog_handler = mocker.MagicMock()

    mock_context.run_context.catalog_handler = mock_catalog_handler

    mock_catalog_handler_put = mocker.MagicMock()
    mock_catalog_handler_put.return_value = None
    mock_catalog_handler.put = mock_catalog_handler_put
    mock_context.run_context.run_id = "RUN_ID"

    mock_catalog_handler.compute_data_folder = "compute_folder"
    monkeypatch.setattr(interaction, "context", mock_context)

    mock_file_path = mocker.MagicMock()
    mock_path = mocker.MagicMock(return_value=mock_file_path)
    mock_file_path.name = "file_name"
    mock_file_path.parent = "in_this_folder"
    monkeypatch.setattr(interaction, "Path", mock_path)

    with caplog.at_level(logging.WARNING):
        interaction.put_in_catalog("this_file")

    assert "No catalog was done by the this_file" in caplog.text

    mock_catalog_handler_put.assert_called_once_with("file_name", compute_data_folder="in_this_folder", run_id="RUN_ID")


def test_put_in_catalog_raises_warning_if_no_context_step_log(mocker, monkeypatch, caplog):
    mock_context = mocker.MagicMock()
    mock_catalog_handler = mocker.MagicMock()

    mock_context.run_context.catalog_handler = mock_catalog_handler
    mock_context.run_context.executor._context_step_log = None

    mock_catalog_handler_put = mocker.MagicMock()
    mock_catalog_handler_put.return_value = None
    mock_catalog_handler.put = mock_catalog_handler_put
    mock_context.run_context.run_id = "RUN_ID"

    mock_catalog_handler.compute_data_folder = "compute_folder"
    monkeypatch.setattr(interaction, "context", mock_context)

    mock_file_path = mocker.MagicMock()
    mock_path = mocker.MagicMock(return_value=mock_file_path)
    mock_file_path.name = "file_name"
    mock_file_path.parent = "in_this_folder"
    monkeypatch.setattr(interaction, "Path", mock_path)

    with caplog.at_level(logging.WARNING):
        interaction.put_in_catalog("this_file")

    assert "Step log context was not found during interaction" in caplog.text

    mock_catalog_handler_put.assert_called_once_with("file_name", compute_data_folder="in_this_folder", run_id="RUN_ID")


def test_put_in_catalog_delegates_to_catalog_handler(mocker, monkeypatch):
    mock_context = mocker.MagicMock()
    mock_catalog_handler = mocker.MagicMock()

    mock_context.run_context.catalog_handler = mock_catalog_handler

    mock_catalog_handler_put = mocker.MagicMock()
    mock_catalog_handler.put = mock_catalog_handler_put
    mock_context.run_context.run_id = "RUN_ID"

    mock_catalog_handler.compute_data_folder = "compute_folder"
    monkeypatch.setattr(interaction, "context", mock_context)

    mock_file_path = mocker.MagicMock()
    mock_path = mocker.MagicMock(return_value=mock_file_path)
    mock_file_path.name = "file_name"
    mock_file_path.parent = "in_this_folder"
    monkeypatch.setattr(interaction, "Path", mock_path)

    interaction.put_in_catalog("this_file")

    mock_catalog_handler_put.assert_called_once_with("file_name", compute_data_folder="in_this_folder", run_id="RUN_ID")


@pytest.mark.noautofixt
def test_get_run_id_returns_from_context(monkeypatch, mocker):
    mock_context = mocker.MagicMock()
    mock_context.run_context.run_id = "1234"
    monkeypatch.setattr(interaction, "context", mock_context)

    assert interaction.get_run_id() == "1234"


@pytest.mark.noautofixt
def test_get_tag_raises_exception_if_no_executor(monkeypatch, mocker):
    mock_context = mocker.MagicMock()
    mock_context.run_context.executor = None
    monkeypatch.setattr(interaction, "context", mock_context)

    with pytest.raises(Exception, match="Please raise a bug report"):
        assert interaction.get_tag() == "1234"


@pytest.mark.noautofixt
def test_get_tag_gets_tag_from_context(monkeypatch, mocker):
    mock_context = mocker.MagicMock()
    mock_context.run_context.tag = "1234"
    monkeypatch.setattr(interaction, "context", mock_context)

    assert interaction.get_tag() == "1234"


@pytest.mark.noautofixt
def test_get_experiment_context_raises_exception_if_no_executor(monkeypatch, mocker):
    mock_context = mocker.MagicMock()
    mock_context.run_context.executor = None
    monkeypatch.setattr(interaction, "context", mock_context)

    with pytest.raises(Exception, match="Please raise a bug report"):
        interaction.get_experiment_tracker_context()


@pytest.mark.noautofixt
def test_get_experiment_context_returns_client_context(monkeypatch, mocker):
    mock_context = mocker.MagicMock()
    mock_experiment_tracker = mocker.MagicMock()
    mock_client_context = mocker.MagicMock()

    mock_experiment_tracker.client_context = mock_client_context

    mock_context.run_context.experiment_tracker = mock_experiment_tracker
    monkeypatch.setattr(interaction, "context", mock_context)

    assert interaction.get_experiment_tracker_context() == mock_client_context


def test_put_object_calls_put_in_catalog(monkeypatch, mocker):
    mock_dump = mocker.MagicMock()
    mock_put_in_catalog = mocker.MagicMock()
    mock_os_remove = mocker.MagicMock()

    monkeypatch.setattr(interaction, "put_in_catalog", mock_put_in_catalog)
    monkeypatch.setattr(interaction.pickler.NativePickler, "dump", mock_dump)
    monkeypatch.setattr(interaction.os, "remove", mock_os_remove)

    interaction.put_object("imdata", "iamsam")

    mock_dump.assert_called_once_with(data="imdata", path="iamsam")
    mock_put_in_catalog.assert_called_once_with(f"iamsam.pickle")
    mock_os_remove.assert_called_once_with(f"iamsam.pickle")


def test_get_object_calls_get_from_catalog(monkeypatch, mocker):
    mock_load = mocker.MagicMock()
    mock_get_from_catalog = mocker.MagicMock()
    mock_os_remove = mocker.MagicMock()

    monkeypatch.setattr(interaction, "get_from_catalog", mock_get_from_catalog)
    monkeypatch.setattr(interaction.pickler.NativePickler, "load", mock_load)
    monkeypatch.setattr(interaction.os, "remove", mock_os_remove)

    interaction.get_object("iamsam")

    mock_load.assert_called_once_with("iamsam")
    mock_get_from_catalog.assert_called_once_with(name="iamsam.pickle", destination_folder=".")
    mock_os_remove.assert_called_once_with("iamsam.pickle")


def test_get_object_raises_exception_if_file_not_found(monkeypatch, mocker):
    mock_load = mocker.MagicMock(side_effect=FileNotFoundError())
    mock_get_from_catalog = mocker.MagicMock()
    mock_os_remove = mocker.MagicMock()

    monkeypatch.setattr(interaction, "get_from_catalog", mock_get_from_catalog)
    monkeypatch.setattr(interaction.pickler.NativePickler, "load", mock_load)
    monkeypatch.setattr(interaction.os, "remove", mock_os_remove)

    with pytest.raises(FileNotFoundError):
        interaction.get_object("iamsam")
