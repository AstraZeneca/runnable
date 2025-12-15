import pytest
from unittest.mock import Mock, patch, PropertyMock
from runnable import defaults, exceptions
from runnable.datastore import RunLog, StepLog, StepAttempt
from extensions.pipeline_executor import GenericPipelineExecutor


# Create a concrete test implementation
class ConcreteGenericPipelineExecutor(GenericPipelineExecutor):
    """Concrete implementation of GenericPipelineExecutor with abstract methods implemented for testing"""

    def execute_node(self, node, map_variable=None, **kwargs):
        pass  # Mock implementation for testing


@pytest.fixture
def mock_context():
    """Create a mock context for testing"""
    context = Mock()
    context.run_log_store = Mock()
    return context


def test_validate_retry_run_missing_run_log(mock_context):
    """Test that retry validation fails when original run log doesn't exist"""
    executor = ConcreteGenericPipelineExecutor()

    # Mock context to simulate retry scenario
    mock_context.is_retry = True
    mock_context.run_id = "retry-run-123"
    mock_context.dag_hash = "current-hash-456"

    # Mock run_log_store to raise exception for missing run log
    mock_context.run_log_store.get_run_log_by_id.side_effect = exceptions.RunLogNotFoundError(
        run_id="retry-run-123"
    )

    with patch.object(ConcreteGenericPipelineExecutor, '_context', new_callable=PropertyMock) as mock_property:
        mock_property.return_value = mock_context
        with pytest.raises(exceptions.RetryValidationError, match="Original run log not found"):
            executor._validate_retry_prerequisites()


def test_validate_retry_run_dag_hash_mismatch(mock_context):
    """Test that retry validation fails when DAG hashes don't match"""
    executor = ConcreteGenericPipelineExecutor()

    # Mock context
    mock_context.is_retry = True
    mock_context.run_id = "retry-run-123"
    mock_context.dag_hash = "current-hash-456"

    # Mock original run log with different DAG hash
    original_run_log = RunLog(
        run_id="retry-run-123",
        dag_hash="original-hash-789",
        status="FAILED"
    )
    mock_context.run_log_store.get_run_log_by_id.return_value = original_run_log

    with patch.object(ConcreteGenericPipelineExecutor, '_context', new_callable=PropertyMock) as mock_property:
        mock_property.return_value = mock_context
        with pytest.raises(exceptions.RetryValidationError, match="DAG structure has changed"):
            executor._validate_retry_prerequisites()


def test_validate_retry_run_success(mock_context):
    """Test successful retry validation"""
    executor = ConcreteGenericPipelineExecutor()

    # Mock context
    mock_context.is_retry = True
    mock_context.run_id = "retry-run-123"
    mock_context.dag_hash = "matching-hash-456"

    # Mock original run log with matching DAG hash
    original_run_log = RunLog(
        run_id="retry-run-123",
        dag_hash="matching-hash-456",
        status="FAILED"
    )
    mock_context.run_log_store.get_run_log_by_id.return_value = original_run_log

    with patch.object(ConcreteGenericPipelineExecutor, '_context', new_callable=PropertyMock) as mock_property:
        mock_property.return_value = mock_context
        # Should not raise any exception
        executor._validate_retry_prerequisites()
