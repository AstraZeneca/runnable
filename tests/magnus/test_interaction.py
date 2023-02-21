import os

import pytest

import magnus
from magnus import defaults  # pylint: disable=import-error
from magnus import exceptions  # pylint: disable=import-error
from magnus import interaction  # pylint: disable=import-error


def test_track_this_adds_values_to_environ(monkeypatch, mocker):
    mock_executor = mocker.MagicMock()
    monkeypatch.setattr(magnus.pipeline, 'global_executor', mock_executor)
    interaction.track_this(a='b')
    assert defaults.TRACK_PREFIX + 'a' in os.environ
    del os.environ[defaults.TRACK_PREFIX + 'a']


def test_track_this_adds_multiple_values_to_environ(mocker, monkeypatch):
    mock_executor = mocker.MagicMock()
    monkeypatch.setattr(magnus.pipeline, 'global_executor', mock_executor)
    interaction.track_this(a='b', b='a')
    assert defaults.TRACK_PREFIX + 'a' in os.environ
    assert defaults.TRACK_PREFIX + 'b' in os.environ
    del os.environ[defaults.TRACK_PREFIX + 'a']
    del os.environ[defaults.TRACK_PREFIX + 'b']


def test_store_paramenter_adds_values_to_environ():
    interaction.store_parameter(a='b')
    assert defaults.PARAMETER_PREFIX + 'a' in os.environ
    del os.environ[defaults.PARAMETER_PREFIX + 'a']


def test_store_parameter_adds_multiple_values_to_environ():
    interaction.store_parameter(a='b', b='a')
    assert defaults.PARAMETER_PREFIX + 'a' in os.environ
    assert defaults.PARAMETER_PREFIX + 'b' in os.environ
    del os.environ[defaults.PARAMETER_PREFIX + 'a']
    del os.environ[defaults.PARAMETER_PREFIX + 'b']


def test_get_parameter_returns_all_parameters_if_no_key_provided(mocker, monkeypatch):
    monkeypatch.setattr(interaction.utils, 'get_user_set_parameters', mocker.MagicMock(return_value='this'))

    assert interaction.get_parameter() == 'this'


def test_get_parameter_returns_parameters_if_provided(mocker, monkeypatch):
    monkeypatch.setattr(interaction.utils, 'get_user_set_parameters', mocker.MagicMock(return_value={'this': 'that'}))

    assert interaction.get_parameter('this') == 'that'


def test_get_parameter_returns_parameters_raises_exception_if_key_not_found(mocker, monkeypatch):
    monkeypatch.setattr(interaction.utils, 'get_user_set_parameters', mocker.MagicMock(return_value={'this': 'that'}))

    with pytest.raises(Exception):
        interaction.get_parameter('this1')


def test_get_secret_delegates_to_secrets_handler_get(mocker, monkeypatch):
    mock_global_exec = mocker.MagicMock()
    import magnus.pipeline  # pylint: disable=import-error
    magnus.pipeline.global_executor = mock_global_exec

    mock_secrets_handler = mocker.MagicMock()
    mock_global_exec.secrets_handler = mock_secrets_handler

    mock_secrets_handler.get.return_value = 'test'

    assert interaction.get_secret('secret') == 'test'


def test_get_secret_raises_exception_if_secrets_handler_raises(mocker, monkeypatch):
    mock_global_exec = mocker.MagicMock()
    import magnus.pipeline  # pylint: disable=import-error
    magnus.pipeline.global_executor = mock_global_exec

    # monkeypatch.setattr(magnus.pipeline, 'global_executor', mock_global_exec)

    mock_secrets_handler = mocker.MagicMock()
    mock_global_exec.secrets_handler = mock_secrets_handler

    mock_secrets_handler.get.side_effect = exceptions.SecretNotFoundError('test', 'test')
    with pytest.raises(exceptions.SecretNotFoundError):
        assert interaction.get_secret('secret')


def test_get_from_catalog_delegates_to_catalog_handler(mocker, monkeypatch):
    import magnus.pipeline  # pylint: disable=import-error
    mock_global_exec = mocker.MagicMock()
    magnus.pipeline.global_executor = mock_global_exec

    mock_catalog_handler_get = mocker.MagicMock()
    mock_global_exec.catalog_handler.get = mock_catalog_handler_get
    mock_global_exec.run_id = 'RUN_ID'

    mock_global_exec.catalog_handler.compute_data_folder = 'compute_folder'

    interaction.get_from_catalog('this')

    mock_catalog_handler_get.assert_called_once_with('this', compute_data_folder='compute_folder', run_id='RUN_ID')


def test_get_from_catalog_uses_destination_folder(mocker, monkeypatch):
    import magnus.pipeline  # pylint: disable=import-error
    mock_global_exec = mocker.MagicMock()
    magnus.pipeline.global_executor = mock_global_exec

    mock_catalog_handler_get = mocker.MagicMock()
    mock_global_exec.catalog_handler.get = mock_catalog_handler_get
    mock_global_exec.run_id = 'RUN_ID'

    mock_global_exec.catalog_handler.compute_data_folder = 'compute_folder_not_used'

    interaction.get_from_catalog('this', destination_folder='use_this_folder')

    mock_catalog_handler_get.assert_called_once_with('this', compute_data_folder='use_this_folder', run_id='RUN_ID')


def test_put_in_catalog_delegates_to_catalog_handler(mocker, monkeypatch):
    import magnus.pipeline  # pylint: disable=import-error
    mock_global_exec = mocker.MagicMock()
    magnus.pipeline.global_executor = mock_global_exec

    mock_catalog_handler_put = mocker.MagicMock()
    mock_global_exec.catalog_handler.put = mock_catalog_handler_put
    mock_global_exec.run_id = 'RUN_ID'

    mock_file_path = mocker.MagicMock()
    mock_path = mocker.MagicMock(return_value=mock_file_path)
    mock_file_path.name = 'file_name'
    mock_file_path.parent = 'in_this_folder'
    monkeypatch.setattr(magnus.interaction, 'Path', mock_path)

    interaction.put_in_catalog('this_file')

    mock_catalog_handler_put.assert_called_once_with('file_name', compute_data_folder='in_this_folder', run_id='RUN_ID')
