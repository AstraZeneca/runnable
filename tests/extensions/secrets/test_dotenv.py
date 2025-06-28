import pytest

from extensions.secrets.dotenv import DotEnvSecrets
from runnable import defaults, exceptions


def test_dot_env_secrets_defaults_to_default_location_if_none_provided(
    mocker, monkeypatch
):
    dot_env_secret = DotEnvSecrets()
    assert dot_env_secret.secrets_location == defaults.DOTENV_FILE_LOCATION


def test_dot_env_secrets_usees_location_if_provided(mocker, monkeypatch):
    dot_env_secret = DotEnvSecrets(location="here")
    assert dot_env_secret.location == "here"


def test_secrets_location_default():
    secrets = DotEnvSecrets()
    assert secrets.secrets_location == secrets.location


def test_load_secrets(mocker):
    secrets = DotEnvSecrets()
    mocker.patch("extensions.secrets.dotenv.dotenv_values", return_value={"FOO": "BAR"})
    secrets._load_secrets()
    assert secrets.secrets == {"FOO": "BAR"}


def test_get_secret_found(mocker):
    secrets = DotEnvSecrets()
    secrets.secrets = {"API_KEY": "123"}
    assert secrets.get("API_KEY") == "123"


def test_get_secret_loads_if_empty(mocker):
    secrets = DotEnvSecrets()
    mocker.patch.object(secrets, "_load_secrets")
    secrets.secrets = {}
    # Simulate _load_secrets populating secrets
    secrets._load_secrets.side_effect = lambda: secrets.secrets.update({"X": "Y"})
    assert secrets.get("X") == "Y"
    secrets._load_secrets.assert_called_once()


def test_get_secret_not_found(mocker):
    secrets = DotEnvSecrets()
    secrets.secrets = {"FOO": "BAR"}
    with pytest.raises(exceptions.SecretNotFoundError) as exc:
        secrets.get("MISSING")
    assert "No secret found by name:MISSING" in str(exc.value)
