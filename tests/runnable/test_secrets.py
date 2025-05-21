import os

import pytest

from runnable import exceptions
from runnable.secrets import BaseSecrets, DoNothingSecretManager, EnvSecretsManager


class TestSecretManager(BaseSecrets):
    """A concrete implementation of BaseSecrets for testing"""

    service_name: str = "test-secrets"

    def get(self, name: str) -> str:
        return f"secret-{name}"


@pytest.fixture
def clean_env():
    """Clean environment variables before and after tests"""
    # Save original env vars
    original_env = dict(os.environ)

    # Clean env vars
    os.environ.clear()

    yield

    # Restore original env vars
    os.environ.clear()
    os.environ.update(original_env)


def test_base_secrets_initialization():
    """Test basic initialization of BaseSecrets implementation"""
    secret_manager = TestSecretManager()
    assert secret_manager.service_name == "test-secrets"
    assert secret_manager.service_type == "secrets"


def test_do_nothing_secret_manager():
    """Test DoNothingSecretManager functionality"""
    manager = DoNothingSecretManager()

    # Should return empty string for any secret name
    assert manager.get("any_secret") == ""
    assert manager.get("another_secret") == ""
    assert manager.service_name == "do-nothing"


def test_env_secrets_manager_existing_secret(clean_env):
    """Test EnvSecretsManager with existing secret"""
    manager = EnvSecretsManager()

    # Set a test secret in environment
    os.environ["TEST_SECRET"] = "secret_value"

    # Should return the secret value
    assert manager.get("TEST_SECRET") == "secret_value"
    assert manager.service_name == "env-secrets"


def test_env_secrets_manager_missing_secret(clean_env):
    """Test EnvSecretsManager with non-existent secret"""
    manager = EnvSecretsManager()

    with pytest.raises(exceptions.SecretNotFoundError) as exc_info:
        manager.get("NONEXISTENT_SECRET")

    assert exc_info.value.secret_name == "NONEXISTENT_SECRET"
    assert exc_info.value.secret_setting == "environment variables"


def test_env_secrets_manager_multiple_secrets(clean_env):
    """Test EnvSecretsManager with multiple secrets"""
    manager = EnvSecretsManager()

    # Set multiple test secrets
    secrets = {"SECRET1": "value1", "SECRET2": "value2", "SECRET3": "value3"}

    for name, value in secrets.items():
        os.environ[name] = value

    # Should return correct values for all secrets
    for name, expected in secrets.items():
        assert manager.get(name) == expected


def test_base_secrets_abstract_methods():
    """Test that BaseSecrets cannot be instantiated directly"""
    with pytest.raises(TypeError):
        BaseSecrets()


def test_secrets_manager_extra_fields():
    """Test that extra fields are not allowed"""
    with pytest.raises(ValueError):
        TestSecretManager(extra_field="not_allowed")
