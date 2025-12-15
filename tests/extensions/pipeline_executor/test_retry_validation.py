import pytest
from unittest.mock import Mock, patch, PropertyMock
from runnable import defaults, exceptions
from runnable.datastore import RunLog, StepLog, StepAttempt, JsonParameter
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


def test_get_parameters_for_retry_loads_original_parameters(mock_context):
    """Test loading parameters from original run during retry"""
    executor = ConcreteGenericPipelineExecutor()

    # Mock context for retry
    mock_context.is_retry = True
    mock_context.run_id = "retry-run-123"
    mock_context.parameters_file = "new-params.yaml"  # User tries to provide new params

    # Mock original run log with parameters
    original_parameters = {
        "param1": JsonParameter(value="original_value1", kind="json"),
        "param2": JsonParameter(value=42, kind="json")
    }
    original_run_log = RunLog(
        run_id="retry-run-123",
        dag_hash="hash-123",
        status="FAILED",
        parameters=original_parameters
    )
    mock_context.run_log_store.get_run_log_by_id.return_value = original_run_log

    with patch.object(ConcreteGenericPipelineExecutor, '_context', new_callable=PropertyMock) as mock_property:
        mock_property.return_value = mock_context

        # Mock _get_parameters to return new parameters (should be ignored in retry)
        with patch.object(executor, '_get_parameters', return_value={"param1": JsonParameter(value="new_value", kind="json")}):
            params = executor._get_parameters_for_retry()

    assert params == original_parameters


def test_get_parameters_for_retry_normal_run(mock_context):
    """Test normal parameter loading for non-retry runs"""
    executor = ConcreteGenericPipelineExecutor()

    # Mock context for normal run
    mock_context.is_retry = False

    expected_params = {"param1": JsonParameter(value="normal_value", kind="json")}

    with patch.object(ConcreteGenericPipelineExecutor, '_context', new_callable=PropertyMock) as mock_property:
        mock_property.return_value = mock_context

        with patch.object(executor, '_get_parameters', return_value=expected_params):
            params = executor._get_parameters_for_retry()

    assert params == expected_params
