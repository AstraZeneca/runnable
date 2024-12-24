from extensions.secrets.dotenv import DotEnvSecrets
from runnable import defaults


def test_dot_env_secrets_defaults_to_default_location_if_none_provided(
    mocker, monkeypatch
):
    dot_env_secret = DotEnvSecrets()
    assert dot_env_secret.secrets_location == defaults.DOTENV_FILE_LOCATION


def test_dot_env_secrets_usees_location_if_provided(mocker, monkeypatch):
    dot_env_secret = DotEnvSecrets(location="here")
    assert dot_env_secret.location == "here"


# TODO: dotenv testing
