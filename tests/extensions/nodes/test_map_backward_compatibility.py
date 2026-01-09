import pytest
from unittest.mock import Mock, patch, MagicMock
from runnable.datastore import JsonParameter, BranchLog
from runnable.graph import Graph
from runnable.tasks import TaskReturns
from extensions.nodes.task import TaskNode


@patch('runnable.context.get_run_context')
def test_map_node_backward_compatibility_with_non_partitioned_store(mock_get_context):
    """Test that map nodes work correctly with non-partitioned stores using prefixed parameters."""
    from extensions.nodes.map import MapNode
    from runnable import defaults
    from runnable.context import PipelineContext

    # Setup mock context with non-partitioned store
    mock_context = Mock(spec=PipelineContext)
    mock_run_log_store = Mock()
    mock_run_log_store.supports_parallel_writes = False  # Non-partitioned store
    mock_context.run_log_store = mock_run_log_store
    mock_context.run_id = "test_run"
    mock_get_context.return_value = mock_context

    # Mock parameters with prefixed naming (old behavior)
    all_params = {
        "items": JsonParameter(kind="json", value=[0, 1]),
        "map_node.0_result": JsonParameter(kind="json", value="value_0"),
        "map_node.0_metric": JsonParameter(kind="json", value=10),
        "map_node.1_result": JsonParameter(kind="json", value="value_1"),
        "map_node.1_metric": JsonParameter(kind="json", value=20),
    }

    def get_parameters_side_effect(run_id, internal_branch_name=None):
        # Non-partitioned stores ignore internal_branch_name
        return all_params

    mock_run_log_store.get_parameters.side_effect = get_parameters_side_effect

    # Mock branch logs (successful branches)
    def get_branch_log_side_effect(internal_branch_name, run_id):
        branch_log = Mock()
        branch_log.status = defaults.SUCCESS
        return branch_log

    mock_run_log_store.get_branch_log.side_effect = get_branch_log_side_effect

    # Mock step log
    mock_step_log = Mock()
    mock_step_log.status = defaults.PROCESSING
    mock_run_log_store.get_step_log.return_value = mock_step_log

    # Create task node with returns
    mock_task = Mock()
    mock_task.returns = [
        TaskReturns(name="result", kind="json"),
        TaskReturns(name="metric", kind="json")
    ]
    mock_task_node = Mock(spec=TaskNode)
    mock_task_node.executable = mock_task

    # Create branch graph
    mock_branch = MagicMock(spec=Graph)
    mock_branch.nodes = {"task1": mock_task_node}

    # Create map node
    map_node = MapNode(
        name="map_node",
        internal_name="map_node",
        internal_branch_name="",
        iterate_on="items",
        iterate_as="item",
        next_node="success",
        branch=mock_branch,
        reducer=None  # Use default list reducer
    )

    # Call fan_in
    map_node.fan_in(iter_variable=None)

    # Verify aggregated parameters were set correctly
    mock_run_log_store.set_parameters.assert_called()
    call_args = mock_run_log_store.set_parameters.call_args

    assert call_args[1]["run_id"] == "test_run"
    # Non-partitioned stores don't use internal_branch_name in final call
    assert "internal_branch_name" not in call_args[1] or call_args[1].get("internal_branch_name") is None

    # Check aggregated parameters
    aggregated_params = call_args[1]["parameters"]

    # Should aggregate task returns from prefixed parameters
    assert "result" in aggregated_params
    assert aggregated_params["result"].value == ["value_0", "value_1"]
    assert "metric" in aggregated_params
    assert aggregated_params["metric"].value == [10, 20]


@patch('runnable.context.get_run_context')
def test_task_backward_compatibility_with_non_partitioned_store(mock_get_context):
    """Test that tasks work correctly with non-partitioned stores using prefixed parameters."""
    from runnable.tasks import PythonTaskType
    from runnable.context import PipelineContext

    # Setup mock context with non-partitioned store
    mock_context = Mock(spec=PipelineContext)
    mock_run_log_store = Mock()
    mock_run_log_store.supports_parallel_writes = False  # Non-partitioned store
    mock_context.run_log_store = mock_run_log_store
    mock_context.run_id = "test_run"
    mock_get_context.return_value = mock_context

    # Mock parameters with both global and prefixed
    all_params = {
        "global_param": JsonParameter(kind="json", value="global_value"),
        "map_node.0_scoped_param": JsonParameter(kind="json", value="scoped_value"),
    }

    mock_run_log_store.get_parameters.return_value = all_params

    # Create task with branch context
    task = PythonTaskType(
        command="dummy.function",
        internal_branch_name="map_node.0"
    )

    # Test _get_scoped_parameters
    scoped_params = task._get_scoped_parameters()

    # Should get prefixed parameters with clean names
    assert "scoped_param" in scoped_params
    assert scoped_params["scoped_param"].value == "scoped_value"
    # Should not get global parameters in branch context
    assert "global_param" not in scoped_params

    # Test _set_scoped_parameters
    new_params = {"new_param": JsonParameter(kind="json", value="new_value")}
    task._set_scoped_parameters(new_params)

    # Should have set prefixed parameters
    mock_run_log_store.set_parameters.assert_called_with(
        parameters={"map_node.0_new_param": new_params["new_param"]},
        run_id="test_run"
    )


@patch('runnable.context.get_run_context')
def test_task_root_context_with_non_partitioned_store(mock_get_context):
    """Test that root context tasks work correctly with non-partitioned stores."""
    from runnable.tasks import PythonTaskType
    from runnable.context import PipelineContext

    # Setup mock context with non-partitioned store
    mock_context = Mock(spec=PipelineContext)
    mock_run_log_store = Mock()
    mock_run_log_store.supports_parallel_writes = False  # Non-partitioned store
    mock_context.run_log_store = mock_run_log_store
    mock_context.run_id = "test_run"
    mock_get_context.return_value = mock_context

    # Mock parameters
    all_params = {
        "global_param": JsonParameter(kind="json", value="global_value"),
        "map_node.0_scoped_param": JsonParameter(kind="json", value="scoped_value"),
    }

    mock_run_log_store.get_parameters.return_value = all_params

    # Create task with root context (no branch)
    task = PythonTaskType(
        command="dummy.function",
        internal_branch_name=None
    )

    # Test _get_scoped_parameters
    scoped_params = task._get_scoped_parameters()

    # Root context should get all parameters
    assert "global_param" in scoped_params
    assert "map_node.0_scoped_param" in scoped_params

    # Test _set_scoped_parameters
    new_params = {"new_param": JsonParameter(kind="json", value="new_value")}
    task._set_scoped_parameters(new_params)

    # Should set parameters directly without prefixing
    mock_run_log_store.set_parameters.assert_called_with(
        parameters=new_params,
        run_id="test_run"
    )
