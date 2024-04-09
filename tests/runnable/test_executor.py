import pytest

from runnable import executor, defaults


@pytest.fixture(autouse=True)
def instantiable_base_class(monkeypatch):
    monkeypatch.setattr(executor.BaseExecutor, "__abstractmethods__", set())
    yield


def test_base_executor_context_refers_to_global_run_context(mocker, monkeypatch):
    mock_run_context = mocker.MagicMock()
    monkeypatch.setattr(executor.context, "run_context", mock_run_context)

    base_executor = executor.BaseExecutor()
    assert base_executor._context is mock_run_context


def test_step_attempt_number_defaults_to_1():
    base_executor = executor.BaseExecutor()

    assert base_executor.step_attempt_number == 1


def test_step_attempt_number_looks_up_environment(monkeypatch):
    monkeypatch.setenv(defaults.ATTEMPT_NUMBER, "12345")
    base_executor = executor.BaseExecutor()

    assert base_executor.step_attempt_number == 12345
