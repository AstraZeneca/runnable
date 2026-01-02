import pytest

from runnable import executor, defaults


@pytest.fixture(autouse=True)
def instantiable_base_class(monkeypatch):
    monkeypatch.setattr(executor.BaseExecutor, "__abstractmethods__", set())
    yield


def test_base_executor_context_refers_to_global_run_context(mocker, monkeypatch):
    mock_run_context = mocker.MagicMock()
    monkeypatch.setattr(executor.context, "get_run_context", lambda: mock_run_context)

    base_executor = executor.BaseExecutor()
    assert base_executor._context is mock_run_context


def test_base_executor_no_step_attempt_number():
    """Test that BaseExecutor no longer has step_attempt_number property"""
    from runnable.executor import BaseExecutor

    # This should not exist anymore
    assert not hasattr(BaseExecutor, 'step_attempt_number')
