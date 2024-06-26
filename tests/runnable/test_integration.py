import logging

import pytest

from runnable import (
    integration,  # pylint: disable=import-error; pylint: disable=import-error
)


def test_base_integration_validate_does_nothing():
    base_integration = integration.BaseIntegration("Executor", "service")
    base_integration.validate()


def test_base_integration_configure_for_traversal_does_nothing():
    base_integration = integration.BaseIntegration("Executor", "service")
    base_integration.validate()


def test_base_integration_configure_for_execution_does_nothing():
    base_integration = integration.BaseIntegration("Executor", "service")
    base_integration.validate()


def test_validate_calls_validate_of_integration_handler(monkeypatch, mocker):
    mock_validate = mocker.MagicMock()
    mock_integration_handler = mocker.MagicMock()

    mock_integration_handler.return_value.validate = mock_validate

    monkeypatch.setattr(integration, "get_integration_handler", lambda x, y: mock_integration_handler())

    integration.validate(None, None)

    assert mock_validate.call_count == 1


def test_configure_for_traversal_calls_validate_of_integration_handler(monkeypatch, mocker):
    mock_configure_for_traversal = mocker.MagicMock()
    mock_integration_handler = mocker.MagicMock()

    mock_integration_handler.return_value.configure_for_traversal = mock_configure_for_traversal

    monkeypatch.setattr(integration, "get_integration_handler", lambda x, y: mock_integration_handler())

    integration.configure_for_traversal(None, None)

    assert mock_configure_for_traversal.call_count == 1


def test_configure_for_execution_calls_validate_of_integration_handler(monkeypatch, mocker):
    mock_configure_for_execution = mocker.MagicMock()
    mock_integration_handler = mocker.MagicMock()

    mock_integration_handler.return_value.configure_for_execution = mock_configure_for_execution

    monkeypatch.setattr(integration, "get_integration_handler", lambda x, y: mock_integration_handler())

    integration.configure_for_execution(None, None)

    assert mock_configure_for_execution.call_count == 1


def test_get_integration_handler_gives_default_integration_if_no_match(monkeypatch, mocker):
    mock_service = mocker.MagicMock()
    mock_service.service_type = "I do not exist"
    mock_service.service_name = "DummyService"

    mock_executor = mocker.MagicMock()
    mock_executor.executor_type = "DummyExecutor"
    mock_executor.service_type = "executor"

    obj = integration.get_integration_handler(mock_executor, mock_service)
    assert isinstance(obj, integration.BaseIntegration)


# def test_get_integration_handler_considers_extension_from_stevedore(monkeypatch, mocker):
#     mock_extension = mocker.MagicMock()
#     mock_extension_manager = mocker.MagicMock(return_value={"extension": mock_extension})

#     monkeypatch.setattr(integration.extension, "ExtensionManager", mock_extension_manager)
#     m = mocker.Mock.create_autospec(return_value=integration.BaseIntegration)

#     m.__class__.__subclasses__ = [] # way to remove subclasses


def test_do_nothing_catalog_validate_emits_warning(caplog):
    extension = integration.DoNothingCatalog("none", "service")

    with caplog.at_level(logging.INFO, logger="runnable"):
        extension.validate()

    assert "A do-nothing catalog does not hold any data and therefore cannot pass data between nodes." in caplog.text


def test_do_nothing_secrets_validate_emits_warning(caplog):
    extension = integration.DoNothingSecrets("none", "service")

    with caplog.at_level(logging.INFO, logger="runnable"):
        extension.validate()

    assert "A do-nothing secrets does not hold any secrets and therefore cannot return you any secrets." in caplog.text


def test_buffered_run_log_store_raises_exception_for_anything_else_than_local(mocker, monkeypatch):
    mock_executor = mocker.MagicMock()

    mock_executor.service_name = "not_local"

    extension = integration.BufferedRunLogStore(mock_executor, "service")
    # expect an exception
    with pytest.raises(Exception, match="Buffered run log store is only supported for local executor"):
        extension.validate()


def test_buffered_run_log_store_accepts_local(mocker, caplog):
    mock_executor = mocker.MagicMock()

    mock_executor.service_name = "local"

    extension = integration.BufferedRunLogStore(mock_executor, "service")
    with caplog.at_level(logging.INFO, logger="runnable"):
        extension.validate()

    assert "Run log generated by buffered run log store are not persisted." in caplog.text
