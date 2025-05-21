import os
from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from runnable import defaults, exceptions
from runnable.datastore import (
    JsonParameter,
    MetricParameter,
    ObjectParameter,
    StepAttempt,
)
from runnable.tasks import (
    BaseTaskType,
    NotebookTaskType,
    PythonTaskType,
    ShellTaskType,
    TaskReturns,
    task_return_to_parameter,
)


# Test Implementation of BaseTaskType
class TestTaskType(BaseTaskType):
    """Concrete implementation of BaseTaskType for testing"""

    task_type: str = "test"
    command: str = "test_command"

    def execute_command(self, map_variable=None) -> StepAttempt:
        return StepAttempt(status=defaults.SUCCESS, start_time=str(datetime.now()))


# Fixtures
@pytest.fixture
def mock_context(mocker):
    """Mock runnable.context at module level"""
    # Create mock context objects
    mock_ctx = Mock()
    mock_ctx.run_id = "test_run"
    mock_ctx.secrets = Mock()
    mock_ctx.run_log_store = Mock()
    mock_ctx.catalog = Mock()

    # Setup mock progress
    mock_progress = Mock()
    mock_progress.stop = Mock()
    mock_progress.start = Mock()

    # Setup get_parameters to return an iterable dictionary
    parameters_dict = {
        "param1": JsonParameter(kind="json", value="value1"),
        "x": JsonParameter(kind="json", value=42),
    }
    mock_ctx.run_log_store.get_parameters.return_value = parameters_dict.copy()

    # Patch both run_context and progress at module level
    mocker.patch("runnable.context.run_context", mock_ctx)
    mocker.patch("runnable.context.progress", mock_progress)

    # Return both mocks for test use
    return mock_ctx, mock_progress


@pytest.fixture
def clean_env():
    """Clean environment before and after tests"""
    original_env = dict(os.environ)
    os.environ.clear()
    yield
    os.environ.clear()
    os.environ.update(original_env)


# Tests for BaseTaskType
def test_base_task_initialization():
    """Test basic task initialization"""
    task = TestTaskType()
    assert task.task_type == "test"
    assert task.command == "test_command"
    assert isinstance(task.secrets, list)
    assert isinstance(task.returns, list)


def test_base_task_secrets_handling(mock_context, clean_env):
    """Test secrets handling in BaseTaskType"""
    # Setup mock secrets behavior
    mock_ctx, _ = mock_context
    mock_ctx.secrets.get.side_effect = lambda x: f"value_of_{x}"

    task = TestTaskType(secrets=["SECRET1", "SECRET2"])

    # Test setting secrets
    task.set_secrets_as_env_variables()
    assert os.environ["SECRET1"] == "value_of_SECRET1"
    assert os.environ["SECRET2"] == "value_of_SECRET2"

    # Test cleaning up secrets
    task.delete_secrets_from_env_variables()
    assert "SECRET1" not in os.environ
    assert "SECRET2" not in os.environ


# Tests for PythonTaskType
def test_python_task_execution(mock_context):
    """Test Python task execution"""
    mock_ctx, mock_progress = mock_context
    task = PythonTaskType(
        command="math.sqrt", returns=[TaskReturns(name="result", kind="json")]
    )

    with patch("runnable.tasks.importlib.import_module") as mock_import:
        mock_module = Mock()
        mock_module.sqrt = lambda x: x**0.5
        mock_import.return_value = mock_module

        attempt_log = task.execute_command()
        assert attempt_log.status == defaults.SUCCESS
        assert mock_progress.stop.called
        assert mock_progress.start.called


def test_python_task_with_returns(mock_context):
    """Test Python task with return values"""
    task = PythonTaskType(
        command="test_module.test_func",
        returns=[
            TaskReturns(name="result1", kind="json"),
            TaskReturns(name="metric1", kind="metric"),
        ],
    )

    with patch("runnable.tasks.importlib.import_module") as mock_import:
        mock_module = Mock()
        mock_module.test_func = lambda: (42, 0.95)
        mock_import.return_value = mock_module

        attempt_log = task.execute_command()
        assert attempt_log.status == defaults.SUCCESS
        assert "result1" in attempt_log.output_parameters
        assert "metric1" in attempt_log.user_defined_metrics


# Tests for NotebookTaskType
def test_notebook_task_validation():
    """Test notebook task validation"""
    # Should raise error for non-ipynb file
    with pytest.raises(Exception):
        NotebookTaskType(command="test.py")

    # Should accept ipynb file
    task = NotebookTaskType(command="test.ipynb")
    assert task.command == "test.ipynb"


def test_notebook_output_path():
    """Test notebook output path generation"""
    task = NotebookTaskType(command="test.ipynb")

    # Test without map variable
    path = task.get_notebook_output_path()
    assert path.endswith("_out.ipynb")

    # Test with map variable
    path = task.get_notebook_output_path({"var": "value"})
    assert "varvalue" in path
    assert path.endswith("_out.ipynb")


# Tests for ShellTaskType
def test_shell_task_execution(mock_context, clean_env):
    """Test shell task execution"""
    task = ShellTaskType(
        command="echo 'test'", returns=[TaskReturns(name="OUTPUT", kind="json")]
    )

    attempt_log = task.execute_command()
    assert attempt_log.status == defaults.SUCCESS


def test_shell_task_with_environment(mock_context, clean_env):
    """Test shell task with environment variables"""
    task = ShellTaskType(
        command="echo $TEST_VAR", returns=[TaskReturns(name="OUTPUT", kind="json")]
    )

    os.environ["TEST_VAR"] = "test_value"
    attempt_log = task.execute_command()
    assert attempt_log.status == defaults.SUCCESS


def test_shell_task_error_handling(mock_context):
    """Test shell task error handling"""
    task = ShellTaskType(command="nonexistent_command")

    attempt_log = task.execute_command()
    assert attempt_log.status == defaults.FAIL


# Tests for TaskReturns and task_return_to_parameter
def test_task_returns_conversion(mock_context):
    """Test converting task returns to parameters"""
    # Test JSON return
    json_return = TaskReturns(name="json_val", kind="json")
    json_param = task_return_to_parameter(json_return, {"key": "value"})
    assert isinstance(json_param, JsonParameter)

    # Test Metric return
    metric_return = TaskReturns(name="metric_val", kind="metric")
    metric_param = task_return_to_parameter(metric_return, 0.95)
    assert isinstance(metric_param, MetricParameter)


def test_task_execution_context(mock_context):
    """Test task execution context manager"""
    mock_ctx, _ = mock_context
    task = TestTaskType()

    with task.execution_context() as params:
        assert isinstance(params, dict)
        assert "param1" in params
        assert params["param1"].value == "value1"
        assert "x" in params
        assert params["x"].value == "42"

        # Add new parameter
        params["param2"] = JsonParameter(kind="json", value="value2")

    # Verify set_parameters was called with updated parameters
    mock_ctx.run_log_store.set_parameters.assert_called_once()
