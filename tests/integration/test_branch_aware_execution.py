import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock

from runnable import context, defaults
from runnable.tasks import PythonTaskType
from runnable.datastore import JsonParameter, StepAttempt
from extensions.run_log_store.partitioned_fs import FileSystemPartitionedRunLogStore


def simple_task_function(input_param):
    """Simple function for testing."""
    return f"processed_{input_param}"


def test_branch_aware_task_execution_with_partitioned_storage():
    """Test complete workflow: tasks get scoped parameters from partitioned storage."""

    with tempfile.TemporaryDirectory() as temp_dir:
        # Setup partitioned run log store
        run_log_store = FileSystemPartitionedRunLogStore()
        run_log_store.log_folder = temp_dir

        # Create task with branch context
        task = PythonTaskType(
            command="tests.integration.test_branch_aware_execution.simple_task_function",
            internal_branch_name="map_node.iteration_1"
        )

        # Setup run context
        run_id = "test_branch_aware_run"

        # Create run log
        run_log_store.create_run_log(run_id)

        # Set parameters in branch partition
        branch_params = {
            "input_param": JsonParameter(kind="json", value="test_input")
        }
        run_log_store.set_parameters(
            run_id=run_id,
            parameters=branch_params,
            internal_branch_name="map_node.iteration_1"
        )

        # Create mock context
        mock_context = Mock()
        mock_context.run_log_store = run_log_store
        mock_context.run_id = run_id
        mock_context.retry_indicator = "0"
        mock_context.secrets = {}

        with patch('runnable.context.get_run_context', return_value=mock_context):
            # Execute task - should get parameters from branch partition
            result = task.execute_command()

            # Verify task executed successfully
            assert result.status == defaults.SUCCESS

            # Verify output parameters were stored in branch partition
            output_params = run_log_store.get_parameters(
                run_id=run_id,
                internal_branch_name="map_node.iteration_1"
            )

            # Should contain both input and output parameters
            assert "input_param" in output_params
            assert output_params["input_param"].value == "test_input"


def test_branch_aware_parameter_isolation():
    """Test that different branches maintain separate parameter namespaces."""

    with tempfile.TemporaryDirectory() as temp_dir:
        # Setup partitioned run log store
        run_log_store = FileSystemPartitionedRunLogStore()
        run_log_store.log_folder = temp_dir
        run_id = "test_isolation_run"
        run_log_store.create_run_log(run_id)

        # Set parameters in different branch partitions
        branch1_params = {
            "input": JsonParameter(kind="json", value="branch1_value")
        }
        branch2_params = {
            "input": JsonParameter(kind="json", value="branch2_value")
        }

        run_log_store.set_parameters(
            run_id=run_id,
            parameters=branch1_params,
            internal_branch_name="map_node.0"
        )
        run_log_store.set_parameters(
            run_id=run_id,
            parameters=branch2_params,
            internal_branch_name="map_node.1"
        )

        # Verify parameters are isolated
        retrieved_branch1 = run_log_store.get_parameters(
            run_id=run_id,
            internal_branch_name="map_node.0"
        )
        retrieved_branch2 = run_log_store.get_parameters(
            run_id=run_id,
            internal_branch_name="map_node.1"
        )

        assert retrieved_branch1["input"].value == "branch1_value"
        assert retrieved_branch2["input"].value == "branch2_value"

        # Root partition should be separate
        root_params = run_log_store.get_parameters(run_id=run_id)
        assert "input" not in root_params


def test_task_scoped_parameter_access():
    """Test tasks access correct partition based on internal_branch_name."""

    with tempfile.TemporaryDirectory() as temp_dir:
        # Setup partitioned run log store
        run_log_store = FileSystemPartitionedRunLogStore()
        run_log_store.log_folder = temp_dir
        run_id = "test_scoped_access_run"
        run_log_store.create_run_log(run_id)

        # Create tasks with different branch contexts
        root_task = PythonTaskType(
            command="tests.integration.test_branch_aware_execution.simple_task_function",
            internal_branch_name=None  # Root context
        )

        branch_task = PythonTaskType(
            command="tests.integration.test_branch_aware_execution.simple_task_function",
            internal_branch_name="map_node.branch1"
        )

        # Set parameters in different partitions
        root_params = {
            "test_param": JsonParameter(kind="json", value="root_value")
        }
        branch_params = {
            "test_param": JsonParameter(kind="json", value="branch_value")
        }

        run_log_store.set_parameters(
            run_id=run_id,
            parameters=root_params,
            internal_branch_name=None
        )
        run_log_store.set_parameters(
            run_id=run_id,
            parameters=branch_params,
            internal_branch_name="map_node.branch1"
        )

        # Create mock context
        mock_context = Mock()
        mock_context.run_log_store = run_log_store
        mock_context.run_id = run_id

        with patch('runnable.context.get_run_context', return_value=mock_context):
            # Root task should access root partition
            root_retrieved = root_task._get_scoped_parameters()
            assert root_retrieved["test_param"].value == "root_value"

            # Branch task should access branch partition
            branch_retrieved = branch_task._get_scoped_parameters()
            assert branch_retrieved["test_param"].value == "branch_value"


def test_clean_parameter_names_with_partitioned_storage():
    """Test that parameter names are clean (not prefixed) with partitioned storage."""

    with tempfile.TemporaryDirectory() as temp_dir:
        # Setup partitioned run log store
        run_log_store = FileSystemPartitionedRunLogStore()
        run_log_store.log_folder = temp_dir
        run_id = "test_clean_names_run"
        run_log_store.create_run_log(run_id)

        # Create task with return configuration
        task = PythonTaskType(
            command="tests.integration.test_branch_aware_execution.simple_task_function",
            internal_branch_name="map_node.iteration_1",
            returns=[{"name": "result", "kind": "json"}]
        )

        # Set input parameters
        input_params = {
            "input_param": JsonParameter(kind="json", value="test_input")
        }
        run_log_store.set_parameters(
            run_id=run_id,
            parameters=input_params,
            internal_branch_name="map_node.iteration_1"
        )

        # Create mock context
        mock_context = Mock()
        mock_context.run_log_store = run_log_store
        mock_context.run_id = run_id
        mock_context.retry_indicator = "0"
        mock_context.secrets = {}

        with patch('runnable.context.get_run_context', return_value=mock_context):
            # Execute task
            result = task.execute_command()

            # Verify task executed successfully
            assert result.status == defaults.SUCCESS

            # Verify parameter names are clean (not prefixed)
            output_params = run_log_store.get_parameters(
                run_id=run_id,
                internal_branch_name="map_node.iteration_1"
            )

            # Should have clean parameter name, not "iteration_1_result"
            assert "result" in output_params
            assert "iteration_1_result" not in output_params


def test_backward_compatibility_with_existing_stores():
    """Test that new branch-aware tasks work with existing parameter access patterns."""

    # This test ensures we haven't broken existing functionality
    # Even though we've added branch awareness, tasks should still work
    # when internal_branch_name is None (root partition access)

    with tempfile.TemporaryDirectory() as temp_dir:
        # Setup partitioned run log store
        run_log_store = FileSystemPartitionedRunLogStore()
        run_log_store.log_folder = temp_dir
        run_id = "test_backward_compatibility_run"
        run_log_store.create_run_log(run_id)

        # Create task without branch context (legacy behavior)
        task = PythonTaskType(
            command="tests.integration.test_branch_aware_execution.simple_task_function"
            # internal_branch_name defaults to None
        )

        # Set parameters in root partition (legacy behavior)
        root_params = {
            "input_param": JsonParameter(kind="json", value="legacy_input")
        }
        run_log_store.set_parameters(
            run_id=run_id,
            parameters=root_params
            # internal_branch_name defaults to None
        )

        # Create mock context
        mock_context = Mock()
        mock_context.run_log_store = run_log_store
        mock_context.run_id = run_id
        mock_context.retry_indicator = "0"
        mock_context.secrets = {}

        with patch('runnable.context.get_run_context', return_value=mock_context):
            # Execute task - should work exactly like before
            result = task.execute_command()

            # Verify task executed successfully
            assert result.status == defaults.SUCCESS

            # Verify parameters are still accessible from root
            final_params = run_log_store.get_parameters(run_id=run_id)
            assert "input_param" in final_params
            assert final_params["input_param"].value == "legacy_input"
