import pytest

from runnable import (
    secrets,  # pylint: disable=import-error
)


@pytest.fixture(autouse=True)
def instantiable_base_class(monkeypatch):
    monkeypatch.setattr(secrets.BaseSecrets, "__abstractmethods__", set())
    yield


def test_base_secrets_context_refers_to_run_context(
    mocker, monkeypatch, instantiable_base_class
):
    mock_run_context = mocker.Mock()

    monkeypatch.setattr(secrets.context, "run_context", mock_run_context)

    assert secrets.BaseSecrets()._context == mock_run_context


def test_base_secrets_get_raises_not_implemented_error(instantiable_base_class):
    base_secret = secrets.BaseSecrets()

    with pytest.raises(NotImplementedError):
        base_secret.get(name="secret")


def test_do_nothing_secrets_handler_returns_none_if_name_provided(mocker, monkeypatch):
    dummy_secret = secrets.DoNothingSecretManager()
    assert dummy_secret.get("I dont exist") == ""
