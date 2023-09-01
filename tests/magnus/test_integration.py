import pytest

from magnus import (
    integration,  # pylint: disable=import-error; pylint: disable=import-error
)


def test_base_integration_validate_does_nothing():
    base_integration = integration.BaseIntegration(None, None)
    base_integration.validate()


def test_base_integration_configure_for_traversal_does_nothing():
    base_integration = integration.BaseIntegration(None, None)
    base_integration.validate()


def test_base_integration_configure_for_execution_does_nothing():
    base_integration = integration.BaseIntegration(None, None)
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
