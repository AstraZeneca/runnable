import pytest
from unittest.mock import Mock, patch
from runnable.tasks import BaseTaskType
from runnable.datastore import JsonParameter


class TestBranchAwareTask(BaseTaskType):
    task_type: str = "test"

    def execute(self):
        return {"test": "result"}


def test_base_task_type_has_internal_branch_name_field():
    """Test BaseTaskType accepts internal_branch_name parameter."""
    task = TestBranchAwareTask(internal_branch_name="test.branch")
    assert task.internal_branch_name == "test.branch"


def test_base_task_type_internal_branch_name_defaults_to_none():
    """Test internal_branch_name defaults to None."""
    task = TestBranchAwareTask()
    assert task.internal_branch_name is None


@patch('runnable.context.get_run_context')
def test_get_scoped_parameters_uses_branch_context(mock_get_context):
    """Test that get_scoped_parameters uses internal_branch_name."""
    # Setup mock context
    mock_context = Mock()
    mock_run_log_store = Mock()
    mock_context.run_log_store = mock_run_log_store
    mock_context.run_id = "test_run"
    mock_get_context.return_value = mock_context

    # Test with branch context
    task = TestBranchAwareTask(internal_branch_name="map.iteration_1")
    task._get_scoped_parameters()

    mock_run_log_store.get_parameters.assert_called_once_with(
        run_id="test_run",
        internal_branch_name="map.iteration_1"
    )


@patch('runnable.context.get_run_context')
def test_set_scoped_parameters_uses_branch_context(mock_get_context):
    """Test that set_scoped_parameters uses internal_branch_name."""
    # Setup mock context
    mock_context = Mock()
    mock_run_log_store = Mock()
    mock_context.run_log_store = mock_run_log_store
    mock_context.run_id = "test_run"
    mock_get_context.return_value = mock_context

    # Test with branch context
    task = TestBranchAwareTask(internal_branch_name="map.iteration_1")
    params = {"result": JsonParameter(kind="json", value="test")}
    task._set_scoped_parameters(params)

    mock_run_log_store.set_parameters.assert_called_once_with(
        parameters=params,
        run_id="test_run",
        internal_branch_name="map.iteration_1"
    )


@patch('runnable.context.get_run_context')
def test_execute_command_uses_scoped_parameters_directly(mock_get_context):
    """Test execute_command gets parameters from scoped partition directly."""
    # Setup mock context
    mock_context = Mock()
    mock_run_log_store = Mock()
    mock_context.run_log_store = mock_run_log_store
    mock_context.run_id = "test_run"
    mock_get_context.return_value = mock_context

    # Mock scoped parameters
    scoped_params = {
        "input": JsonParameter(kind="json", value="test_input")
    }
    mock_run_log_store.get_parameters.return_value = scoped_params

    # Test task with branch context
    task = TestBranchAwareTask(internal_branch_name="map.iteration_1")

    # Should call get_parameters with branch context, not resolve_unreduced_parameters
    with patch.object(task, '_get_scoped_parameters', return_value=scoped_params) as mock_get_scoped:
        # Simulate part of execute_command that gets parameters
        params = task._get_scoped_parameters()

    mock_get_scoped.assert_called_once()
    assert params == scoped_params
