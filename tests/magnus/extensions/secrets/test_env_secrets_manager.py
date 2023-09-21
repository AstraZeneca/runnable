import pytest
import os

from magnus.extensions.secrets.env_secrets.implementation import EnvSecretsManager
from magnus import exceptions


def test_env_secrets_manager_raises_error_if_name_provided_and_not_present():
    manager = EnvSecretsManager()

    with pytest.raises(exceptions.SecretNotFoundError):
        manager.get("environment")


def test_env_secrets_returns_secret_if_present_in_environment(monkeypatch):
    monkeypatch.setenv("TEST_SECRET", "test_secret")

    manager = EnvSecretsManager()
    assert manager.get("TEST_SECRET") == "test_secret"


def test_env_secrets_returns_secret_if_present_in_environment_with_prefix(monkeypatch):
    monkeypatch.setenv("PREFIX_TEST_SECRET", "test_secret")

    manager = EnvSecretsManager(prefix="PREFIX_")
    assert manager.get("TEST_SECRET") == "test_secret"


def test_env_secrets_returns_secret_if_present_in_environment_with_suffix(monkeypatch):
    monkeypatch.setenv("TEST_SECRET_SUFFIX", "test_secret")

    manager = EnvSecretsManager(suffix="_SUFFIX")
    assert manager.get("TEST_SECRET") == "test_secret"


def test_env_secrets_returns_secret_if_present_in_environment_with_suffix_and_prefix(monkeypatch):
    monkeypatch.setenv("PREFIX_TEST_SECRET_SUFFIX", "test_secret")

    manager = EnvSecretsManager(suffix="_SUFFIX", prefix="PREFIX_")
    assert manager.get("TEST_SECRET") == "test_secret"


def test_env_secrets_returns_matched_secrets_with_suffix(monkeypatch):
    monkeypatch.setenv("TEST_SECRET_SUFFIX", "test_secret")

    manager = EnvSecretsManager(suffix="_SUFFIX")

    assert manager.get() == {"TEST_SECRET": "test_secret"}


def test_env_secrets_returns_matched_secrets_with_prefix(monkeypatch):
    monkeypatch.setenv("PREFIX_TEST_SECRET", "test_secret")

    manager = EnvSecretsManager(prefix="PREFIX_")

    assert manager.get() == {"TEST_SECRET": "test_secret"}


def test_env_secrets_returns_matched_secrets_with_prefix_and_suffix(monkeypatch):
    monkeypatch.setenv("PREFIX_TEST_SECRET_SUFFIX", "test_secret")

    manager = EnvSecretsManager(prefix="PREFIX_", suffix="_SUFFIX")

    assert manager.get() == {"TEST_SECRET": "test_secret"}


def test_env_secrets_returns_os_environ_if_no_prefix_or_suffix():
    manager = EnvSecretsManager()

    assert manager.get() == os.environ
