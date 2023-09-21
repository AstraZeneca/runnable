import pytest

from magnus.extensions.secrets.env_secrets import integration


def test_local_container_integration_raises_exception(caplog, mocker):
    mock_executor = mocker.MagicMock()

    test_integration = integration.LocalContainerComputeEnvSecretsManager(mock_executor, "service")

    with pytest.raises(Exception, match="Local container executions cannot be used with environment secrets manager"):
        test_integration.validate()
