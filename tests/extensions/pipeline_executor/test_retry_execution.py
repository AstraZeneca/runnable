import pytest
from unittest.mock import Mock, patch, PropertyMock
from runnable import defaults, exceptions
from runnable.datastore import RunLog, StepLog, StepAttempt, JsonParameter
from runnable.nodes import BaseNode
from extensions.pipeline_executor import GenericPipelineExecutor


# Create a concrete test implementation
class ConcreteGenericPipelineExecutor(GenericPipelineExecutor):
    """Concrete implementation of GenericPipelineExecutor with abstract methods implemented for testing"""

    def execute_node(self, node, map_variable=None, **kwargs):
        pass  # Mock implementation for testing

    def trigger_node_execution(self, node, map_variable=None):
        pass  # Mock implementation for testing

    def add_code_identities(self, node, step_log):
        pass  # Mock implementation for testing


@pytest.fixture
def mock_context():
    """Create a mock context for testing"""
    context = Mock()
    context.run_log_store = Mock()
    return context


def test_should_skip_successful_step_in_retry(mock_context):
    """Test that successful steps are skipped during retry"""
    executor = ConcreteGenericPipelineExecutor()

    # Mock context for retry
    mock_context.is_retry = True
    mock_context.run_id = "retry-run-123"

    # Mock node
    mock_node = Mock(spec=BaseNode)
    mock_node._get_step_log_name.return_value = "successful_step"
    mock_node.internal_name = "successful_step"

    # Mock step log with successful last attempt
    step_log = StepLog(
        name="successful_step",
        internal_name="successful_step",
        attempts=[
            StepAttempt(attempt_number=1, status=defaults.SUCCESS)
        ]
    )
    mock_context.run_log_store.get_step_log.return_value = step_log

    with patch.object(ConcreteGenericPipelineExecutor, '_context', new_callable=PropertyMock) as mock_property:
        mock_property.return_value = mock_context

        should_skip = executor._should_skip_step_in_retry(mock_node)

    assert should_skip is True


def test_should_not_skip_failed_step_in_retry(mock_context):
    """Test that failed steps are not skipped during retry"""
    executor = ConcreteGenericPipelineExecutor()

    # Mock context for retry
    mock_context.is_retry = True
    mock_context.run_id = "retry-run-123"

    # Mock node
    mock_node = Mock(spec=BaseNode)
    mock_node._get_step_log_name.return_value = "failed_step"
    mock_node.internal_name = "failed_step"

    # Mock step log with failed last attempt
    step_log = StepLog(
        name="failed_step",
        internal_name="failed_step",
        attempts=[
            StepAttempt(attempt_number=1, status=defaults.FAIL)
        ]
    )
    mock_context.run_log_store.get_step_log.return_value = step_log

    with patch.object(ConcreteGenericPipelineExecutor, '_context', new_callable=PropertyMock) as mock_property:
        mock_property.return_value = mock_context

        should_skip = executor._should_skip_step_in_retry(mock_node)

    assert should_skip is False


def test_should_not_skip_never_executed_step_in_retry(mock_context):
    """Test that never executed steps are not skipped during retry"""
    executor = ConcreteGenericPipelineExecutor()

    # Mock context for retry
    mock_context.is_retry = True
    mock_context.run_id = "retry-run-123"

    # Mock node
    mock_node = Mock(spec=BaseNode)
    mock_node._get_step_log_name.return_value = "never_executed_step"
    mock_node.internal_name = "never_executed_step"

    # Mock step log not found (never executed)
    mock_context.run_log_store.get_step_log.side_effect = exceptions.StepLogNotFoundError(
        run_id="retry-run-123", step_name="never_executed_step"
    )

    with patch.object(ConcreteGenericPipelineExecutor, '_context', new_callable=PropertyMock) as mock_property:
        mock_property.return_value = mock_context

        should_skip = executor._should_skip_step_in_retry(mock_node)

    assert should_skip is False


def test_should_not_skip_step_in_normal_run(mock_context):
    """Test that no steps are skipped during normal (non-retry) runs"""
    executor = ConcreteGenericPipelineExecutor()

    # Mock context for normal run
    mock_context.is_retry = False

    # Mock node
    mock_node = Mock(spec=BaseNode)
    mock_node.internal_name = "any_step"

    with patch.object(ConcreteGenericPipelineExecutor, '_context', new_callable=PropertyMock) as mock_property:
        mock_property.return_value = mock_context

        should_skip = executor._should_skip_step_in_retry(mock_node)

    assert should_skip is False


def test_execute_from_graph_calls_should_skip_step_in_retry(mock_context):
    """Test that execute_from_graph calls the skip logic method"""
    executor = ConcreteGenericPipelineExecutor()

    # Mock node
    mock_node = Mock(spec=BaseNode)
    mock_node.internal_name = "test_step"

    # Track calls to the skip method
    with patch.object(executor, '_should_skip_step_in_retry', return_value=True) as mock_skip:
        with patch.object(ConcreteGenericPipelineExecutor, '_context', new_callable=PropertyMock) as mock_property:
            mock_property.return_value = mock_context

            # Call execute_from_graph
            result = executor.execute_from_graph(mock_node)

    # Should have called the skip method with correct parameters
    mock_skip.assert_called_once_with(mock_node, None)

    # Method should return None when skipped
    assert result is None


def test_execute_from_graph_skips_when_should_skip_returns_true(mock_context):
    """Test that execute_from_graph skips execution when _should_skip_step_in_retry returns True"""
    executor = ConcreteGenericPipelineExecutor()

    # Mock node
    mock_node = Mock(spec=BaseNode)
    mock_node.internal_name = "successful_step"

    call_count = 0

    def mock_create_step_log(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return Mock()

    mock_context.run_log_store.create_step_log.side_effect = mock_create_step_log

    # Mock should_skip to return True (step should be skipped)
    with patch.object(executor, '_should_skip_step_in_retry', return_value=True):
        with patch.object(ConcreteGenericPipelineExecutor, '_context', new_callable=PropertyMock) as mock_property:
            mock_property.return_value = mock_context

            result = executor.execute_from_graph(mock_node)

    # Should not call create_step_log since method returns early on skip
    assert call_count == 0
    mock_context.run_log_store.create_step_log.assert_not_called()

    # Method should return None (void method)
    assert result is None


def test_set_up_run_log_calls_retry_validation_for_retry_runs(mock_context):
    """Test that _set_up_run_log calls retry validation for retry runs and returns early"""
    executor = ConcreteGenericPipelineExecutor()

    # Mock context for retry
    mock_context.is_retry = True
    mock_context.run_id = "retry-run-123"

    # Mock run log store methods (should not be called for retry)
    mock_context.run_log_store.get_run_log_by_id = Mock()
    mock_context.run_log_store.create_run_log = Mock()
    mock_context.run_log_store.set_parameters = Mock()
    mock_context.run_log_store.set_run_config = Mock()

    with patch.object(ConcreteGenericPipelineExecutor, '_context', new_callable=PropertyMock) as mock_property:
        mock_property.return_value = mock_context

        with patch.object(executor, '_validate_retry_prerequisites') as mock_validate:
            executor._set_up_run_log()

    # Should call validation
    mock_validate.assert_called_once()

    # Should NOT call any run log store methods (returns early)
    mock_context.run_log_store.get_run_log_by_id.assert_not_called()
    mock_context.run_log_store.create_run_log.assert_not_called()
    mock_context.run_log_store.set_parameters.assert_not_called()
    mock_context.run_log_store.set_run_config.assert_not_called()


def test_set_up_run_log_works_normally_for_non_retry_runs(mock_context):
    """Test that _set_up_run_log works normally for non-retry runs"""
    executor = ConcreteGenericPipelineExecutor()

    # Mock context for normal run
    mock_context.is_retry = False
    mock_context.run_id = "normal-run-123"
    mock_context.tag = None
    mock_context.dag_hash = "test-hash"
    mock_context.model_dump.return_value = {"test": "config"}

    # Mock run log store for normal flow
    mock_context.run_log_store.get_run_log_by_id.side_effect = exceptions.RunLogNotFoundError(run_id="normal-run-123")
    mock_context.run_log_store.create_run_log = Mock()
    mock_context.run_log_store.set_parameters = Mock()
    mock_context.run_log_store.set_run_config = Mock()

    test_params = {"param1": JsonParameter(value="test", kind="json")}

    with patch.object(ConcreteGenericPipelineExecutor, '_context', new_callable=PropertyMock) as mock_property:
        mock_property.return_value = mock_context

        with patch.object(executor, '_get_parameters', return_value=test_params):
            with patch.object(executor, '_validate_retry_prerequisites') as mock_validate:
                executor._set_up_run_log()

    # Should NOT call retry validation for normal runs
    mock_validate.assert_not_called()

    # Should create run log normally
    mock_context.run_log_store.create_run_log.assert_called_once_with(
        run_id="normal-run-123",
        tag=None,
        status=defaults.PROCESSING,
        dag_hash="test-hash"
    )
    mock_context.run_log_store.set_parameters.assert_called_once_with(
        run_id="normal-run-123", parameters=test_params
    )
    mock_context.run_log_store.set_run_config.assert_called_once_with(
        run_id="normal-run-123", run_config={"test": "config"}
    )


def test_set_up_run_log_propagates_retry_validation_errors(mock_context):
    """Test that _set_up_run_log propagates RetryValidationError from validation"""
    executor = ConcreteGenericPipelineExecutor()

    # Mock context for retry
    mock_context.is_retry = True
    mock_context.run_id = "retry-run-123"

    validation_error = exceptions.RetryValidationError(
        "DAG structure has changed", run_id="retry-run-123"
    )

    with patch.object(ConcreteGenericPipelineExecutor, '_context', new_callable=PropertyMock) as mock_property:
        mock_property.return_value = mock_context

        with patch.object(executor, '_validate_retry_prerequisites', side_effect=validation_error):
            with pytest.raises(exceptions.RetryValidationError, match="DAG structure has changed"):
                executor._set_up_run_log()
