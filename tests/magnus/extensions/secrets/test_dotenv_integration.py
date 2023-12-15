import logging
from pathlib import Path

from magnus.extensions.secrets.dotenv import integration


def test_validate_issues_warning(mocker, caplog):
    mock_executor = mocker.MagicMock()
    mock_dot_env_secrets = mocker.MagicMock()

    test_integration = integration.LocalContainerComputeDotEnvSecrets(mock_executor, mock_dot_env_secrets)

    with caplog.at_level(logging.WARNING):
        test_integration.validate()

    assert "Using dot env for non local deployments is not ideal" in caplog.text


def test_configure_for_traversal_populates_volumes(mocker, monkeypatch):
    mock_local_container = mocker.MagicMock()
    monkeypatch.setattr(integration, "LocalContainerExecutor", mock_local_container)

    mock_executor = mocker.MagicMock()
    mock_executor._volumes = {}
    mock_executor._container_secrets_location = "this_location"

    mock_dot_env_secrets = mocker.MagicMock()
    mock_dot_env_secrets.secrets_location = "secrets_location"

    test_integration = integration.LocalContainerComputeDotEnvSecrets(mock_executor, mock_dot_env_secrets)
    test_integration.configure_for_traversal()

    assert mock_executor._volumes == {str(Path("secrets_location").resolve()): {"bind": "this_location", "mode": "ro"}}


def test_configure_for_execution_assigns_secrets_location_within_container(mocker, monkeypatch):
    mock_local_container = mocker.MagicMock()
    monkeypatch.setattr(integration, "LocalContainerExecutor", mock_local_container)

    mock_executor = mocker.MagicMock()
    mock_executor._container_secrets_location = "this_location"

    mock_dot_env_secrets = mocker.MagicMock()

    test_integration = integration.LocalContainerComputeDotEnvSecrets(mock_executor, mock_dot_env_secrets)
    test_integration.configure_for_execution()

    assert mock_dot_env_secrets.location == "this_location"
