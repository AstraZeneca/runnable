"""Test structure validation when retrying pipelines."""
import os
import pytest

from extensions.pipeline_executor import GenericPipelineExecutor
from runnable import context, defaults, exceptions
from runnable.datastore import RunLog


class TestStructureValidationExecutor(GenericPipelineExecutor):
    """Test implementation of GenericPipelineExecutor for structure validation tests."""

    def execute_node(self, node, map_variable=None):
        """Implementation of the abstract method to execute nodes"""
        self._execute_node(node, map_variable)

    def add_code_identities(self, node, step_log):
        """Implementation of add_code_identities"""
        pass

    def add_task_log_to_catalog(self, name, map_variable=None):
        """Implementation of add_task_log_to_catalog"""
        pass


@pytest.fixture
def test_executor():
    """Return an instance of TestStructureValidationExecutor for testing"""
    return TestStructureValidationExecutor(service_name="test_structure_validation_executor")


@pytest.fixture
def mock_context(mocker):
    """Create a mock context for testing"""
    mock_ctx = mocker.MagicMock(spec=context.PipelineContext)
    mock_ctx.run_id = "retry-original-run-123-a1b2"
    mock_ctx.tag = "test-tag"
    mock_ctx.dag_hash = "current-structure-hash"
    mock_ctx.parameters_file = None

    # Setup run log store mock
    mock_run_log_store = mocker.MagicMock()
    mock_ctx.run_log_store = mock_run_log_store

    # Setup catalog mock
    mock_catalog = mocker.MagicMock()
    mock_ctx.catalog = mock_catalog

    mocker.patch.object(context, "run_context", mock_ctx)
    return mock_ctx


def test_structure_validation_passes_when_structures_match(test_executor, mock_context, mocker):
    """Test that structure validation passes when current and original structures match."""
    # Set up retry mode
    original_run_id = "original-run-123"
    os.environ[defaults.ENV_RETRY_RUN_ID] = original_run_id

    try:
        # Create mock for original run log with matching hash
        original_run_log_mock = mocker.MagicMock()
        original_run_log_mock.dag_hash = "current-structure-hash"

        # Mock get_run_log_by_id calls
        def mock_get_run_log_by_id(run_id, full=False):
            if run_id == original_run_id:
                return original_run_log_mock  # For original run log lookup
            else:
                raise exceptions.RunLogNotFoundError("Not found")  # For new retry run log

        mock_context.run_log_store.get_run_log_by_id.side_effect = mock_get_run_log_by_id

        # Mock _get_parameters to return some parameters
        mock_params = {}
        mocker.patch.object(test_executor, "_get_parameters", return_value=mock_params)

        # Mock context.model_dump to return a config dict
        mock_config = {"config_key": "config_value"}
        mock_context.model_dump.return_value = mock_config

        # This should NOT raise an exception because structures match
        test_executor._set_up_run_log()

        # Verify that create_run_log was called with retry metadata
        mock_context.run_log_store.create_run_log.assert_called_once_with(
            run_id=mock_context.run_id,
            tag=mock_context.tag,
            status=defaults.PROCESSING,
            dag_hash=mock_context.dag_hash,
            original_run_id=original_run_id,  # This should be added for retry
        )

    finally:
        os.environ.pop(defaults.ENV_RETRY_RUN_ID, None)


def test_structure_validation_fails_when_structures_differ(test_executor, mock_context, mocker):
    """Test that structure validation fails when current and original structures differ."""
    # Set up retry mode
    original_run_id = "original-run-123"
    os.environ[defaults.ENV_RETRY_RUN_ID] = original_run_id

    try:
        # Create mock for original run log with DIFFERENT hash
        original_run_log_mock = mocker.MagicMock()
        original_run_log_mock.dag_hash = "different-structure-hash"

        # Mock get_run_log_by_id calls
        def mock_get_run_log_by_id(run_id, full=False):
            if run_id == original_run_id:
                return original_run_log_mock  # For original run log lookup - DIFFERENT hash
            else:
                raise exceptions.RunLogNotFoundError("Not found")  # For new retry run log

        mock_context.run_log_store.get_run_log_by_id.side_effect = mock_get_run_log_by_id

        # Mock _get_parameters to return some parameters
        mock_params = {}
        mocker.patch.object(test_executor, "_get_parameters", return_value=mock_params)

        # This SHOULD raise an exception because structures differ
        with pytest.raises(exceptions.StructureChangedError, match="Pipeline structure has changed"):
            test_executor._set_up_run_log()

    finally:
        os.environ.pop(defaults.ENV_RETRY_RUN_ID, None)


def test_no_structure_validation_when_not_in_retry_mode(test_executor, mock_context, mocker):
    """Test that no structure validation occurs when not in retry mode."""
    # Ensure we're NOT in retry mode
    os.environ.pop(defaults.ENV_RETRY_RUN_ID, None)

    # Mock that run log doesn't exist (normal case)
    mock_context.run_log_store.get_run_log_by_id.side_effect = exceptions.RunLogNotFoundError("Not found")

    # Mock _get_parameters to return some parameters
    mock_params = {}
    mocker.patch.object(test_executor, "_get_parameters", return_value=mock_params)

    # Mock context.model_dump to return a config dict
    mock_config = {"config_key": "config_value"}
    mock_context.model_dump.return_value = mock_config

    # This should NOT raise an exception and should NOT perform structure validation
    test_executor._set_up_run_log()

    # Verify that create_run_log was called with original_run_id=None
    mock_context.run_log_store.create_run_log.assert_called_once_with(
        run_id=mock_context.run_id,
        tag=mock_context.tag,
        status=defaults.PROCESSING,
        dag_hash=mock_context.dag_hash,
        original_run_id=None,  # Should be None when not in retry mode
    )
