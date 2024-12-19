import pytest

from runnable import defaults, exceptions

from runnable.extensions.secrets.dotenv.implementation import DotEnvSecrets
import runnable.extensions.secrets.dotenv.implementation as implementation


def test_dot_env_secrets_defaults_to_default_location_if_none_provided(
    mocker, monkeypatch
):
    dot_env_secret = DotEnvSecrets()
    assert dot_env_secret.secrets_location == defaults.DOTENV_FILE_LOCATION


def test_dot_env_secrets_usees_location_if_provided(mocker, monkeypatch):
    dot_env_secret = DotEnvSecrets(location="here")
    assert dot_env_secret.location == "here"


def test_dot_env_secrets_get_returns_secret_if_one_provided(mocker, monkeypatch):
    dot_env_secret = DotEnvSecrets(location="here")
    dot_env_secret.secrets["give"] = "this"

    assert dot_env_secret.get("give") == "this"


def test_dot_env_secrets_raises_exception_if_secret_not_found(mocker, monkeypatch):
    monkeypatch.setattr(DotEnvSecrets, "_load_secrets", mocker.MagicMock())

    dot_env_secret = DotEnvSecrets(location="here")
    dot_env_secret.secrets["give"] = "this"

    with pytest.raises(expected_exception=exceptions.SecretNotFoundError):
        dot_env_secret.get("give1")


def test_dot_env_load_secrets_raises_exception_if_file_does_not_exist(
    mocker, monkeypatch
):
    monkeypatch.setattr(
        implementation.utils, "does_file_exist", mocker.MagicMock(return_value=False)
    )

    dot_env_secret = DotEnvSecrets(location="here")

    with pytest.raises(Exception, match="Did not find the secrets file"):
        dot_env_secret._load_secrets()


def test_dot_env_load_secrets_raises_exception_if_secret_formatting_is_invalid(
    mocker, monkeypatch
):
    monkeypatch.setattr(
        implementation.utils, "does_file_exist", mocker.MagicMock(return_value=True)
    )

    dot_env_secret = DotEnvSecrets(location="here")

    with pytest.raises(
        Exception, match="A secret should be of format, secret_name=secret_value"
    ):
        mocker.patch("builtins.open", mocker.mock_open(read_data="data"))
        dot_env_secret._load_secrets()


def test_dot_env_load_secrets_raises_exception_if_secret_formatting_is_invalid_ge_2(
    mocker, monkeypatch
):
    monkeypatch.setattr(
        implementation.utils, "does_file_exist", mocker.MagicMock(return_value=True)
    )

    dot_env_secret = DotEnvSecrets(location="here")

    with pytest.raises(
        Exception, match="A secret should be of format, secret_name=secret_value"
    ):
        mocker.patch("builtins.open", mocker.mock_open(read_data="data=data1="))
        dot_env_secret._load_secrets()


def test_dot_env_load_secrets_populates_correct_secrets_if_valid(mocker, monkeypatch):
    monkeypatch.setattr(
        implementation.utils, "does_file_exist", mocker.MagicMock(return_value=True)
    )

    dot_env_secret = DotEnvSecrets(location="here")

    mocker.patch("builtins.open", mocker.mock_open(read_data="data=data1\n"))
    dot_env_secret._load_secrets()
    assert dot_env_secret.secrets == {"data": "data1"}
