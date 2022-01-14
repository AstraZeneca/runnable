from pathlib import Path
import pytest

from magnus import secrets  # pylint: disable=import-error
from magnus import defaults  # pylint: disable=import-error


def test_base_secrets_init_config_empty_dict():
    base_secret = secrets.BaseSecrets(config=None)

    assert base_secret.config == {}


def test_base_secrets_get_raises_not_implemented_error():
    base_secret = secrets.BaseSecrets(config=None)

    with pytest.raises(NotImplementedError):
        base_secret.get()


def test_do_nothing_secrets_handler_returns_none_if_name_provided(mocker, monkeypatch):
    mock_base_secret = mocker.MagicMock()

    monkeypatch.setattr(secrets, 'BaseSecrets', mock_base_secret)

    dummy_secret = secrets.DoNothingSecretManager(config=None)
    assert dummy_secret.get('I dont exist') is ''


def test_do_nothing__handler_returns_empty_dict_if_name_not_provided(mocker, monkeypatch):
    mock_base_secret = mocker.MagicMock()

    monkeypatch.setattr(secrets, 'BaseSecrets', mock_base_secret)

    dummy_secret = secrets.DoNothingSecretManager(config=None)
    assert dummy_secret.get() == {}


def test_dot_env_secrets_defaults_to_default_location_if_none_provided(mocker, monkeypatch):
    mock_base_secret = mocker.MagicMock()

    monkeypatch.setattr(secrets, 'BaseSecrets', mock_base_secret)

    dot_env_secret = secrets.DotEnvSecrets(config=None)
    assert dot_env_secret.get_secrets_location() == defaults.DOTENV_FILE_LOCATION


def test_dot_env_secrets_usees_location_if_provided(mocker, monkeypatch):
    mock_base_secret = mocker.MagicMock()

    monkeypatch.setattr(secrets, 'BaseSecrets', mock_base_secret)

    dot_env_secret = secrets.DotEnvSecrets(config={'location': 'here'})
    assert dot_env_secret.get_secrets_location() == 'here'


def test_dot_env_secrets_get_returns_all_secrets_if_no_name_provided(mocker, monkeypatch):
    mock_base_secret = mocker.MagicMock()

    monkeypatch.setattr(secrets, 'BaseSecrets', mock_base_secret)
    monkeypatch.setattr(secrets.DotEnvSecrets, 'load_secrets', mocker.MagicMock())

    dot_env_secret = secrets.DotEnvSecrets(config={'location': 'here'})
    dot_env_secret.secrets = 'TopSecret'

    assert dot_env_secret.get() == 'TopSecret'


def test_dot_env_secrets_get_returns_secret_if_one_provided(mocker, monkeypatch):
    monkeypatch.setattr(secrets, 'BaseSecrets', mocker.MagicMock())
    monkeypatch.setattr(secrets.DotEnvSecrets, 'load_secrets', mocker.MagicMock())

    dot_env_secret = secrets.DotEnvSecrets(config={'location': 'here'})
    dot_env_secret.secrets['give'] = 'this'

    assert dot_env_secret.get('give') == 'this'


def test_dot_env_secrets_raises_exception_if_secret_not_found(mocker, monkeypatch):
    mock_base_secret = mocker.MagicMock()

    monkeypatch.setattr(secrets, 'BaseSecrets', mock_base_secret)
    monkeypatch.setattr(secrets.DotEnvSecrets, 'load_secrets', mocker.MagicMock())

    dot_env_secret = secrets.DotEnvSecrets(config={'location': 'here'})
    dot_env_secret.secrets['give'] = 'this'

    with pytest.raises(Exception):
        dot_env_secret.get('give1')


def test_dot_env_load_secrets_raises_exception_if_file_does_not_exist(mocker, monkeypatch):
    monkeypatch.setattr(secrets, 'lru_cache', mocker.MagicMock())
    monkeypatch.setattr(secrets.utils, 'does_file_exist', mocker.MagicMock(return_value=False))

    monkeypatch.setattr(secrets, 'BaseSecrets', mocker.MagicMock())
    dot_env_secret = secrets.DotEnvSecrets(config={'location': 'here'})

    with pytest.raises(Exception):
        dot_env_secret.load_secrets()


def test_dot_env_load_secrets_raises_exception_if_secret_formatting_is_invalid(mocker, monkeypatch):
    monkeypatch.setattr(secrets, 'lru_cache', mocker.MagicMock())
    monkeypatch.setattr(secrets.utils, 'does_file_exist', mocker.MagicMock(return_value=True))

    monkeypatch.setattr(secrets, 'BaseSecrets', mocker.MagicMock())
    dot_env_secret = secrets.DotEnvSecrets(config={'location': 'here'})

    with pytest.raises(Exception):
        mocker.patch('builtins.open', mocker.mock_open(read_data='data'))
        dot_env_secret.load_secrets()


def test_dot_env_load_secrets_raises_exception_if_secret_formatting_is_invalid_ge_2(mocker, monkeypatch):
    monkeypatch.setattr(secrets, 'lru_cache', mocker.MagicMock())
    monkeypatch.setattr(secrets.utils, 'does_file_exist', mocker.MagicMock(return_value=True))

    monkeypatch.setattr(secrets, 'BaseSecrets', mocker.MagicMock())
    dot_env_secret = secrets.DotEnvSecrets(config={'location': 'here'})

    with pytest.raises(Exception):
        mocker.patch('builtins.open', mocker.mock_open(read_data=['data=data1=']))
        dot_env_secret.load_secrets()


def test_dot_env_load_secrets_populates_correct_secrets_if_valid(mocker, monkeypatch):
    monkeypatch.setattr(secrets, 'lru_cache', mocker.MagicMock())
    monkeypatch.setattr(secrets.utils, 'does_file_exist', mocker.MagicMock(return_value=True))

    monkeypatch.setattr(secrets, 'BaseSecrets', mocker.MagicMock())
    dot_env_secret = secrets.DotEnvSecrets(config={'location': 'here'})

    mocker.patch('builtins.open', mocker.mock_open(read_data='data=data1\n'))
    dot_env_secret.load_secrets()
    assert dot_env_secret.secrets == {'data': 'data1'}
