import pytest
from unittest.mock import Mock, patch, MagicMock
from runnable.datastore import JsonParameter, BranchLog
from runnable.graph import Graph


@patch('runnable.context.get_run_context')
def test_fan_in_discovers_parameters_from_branch_partitions(mock_get_context):
    """Test fan_in discovers and aggregates parameters from branch partitions."""
    from extensions.nodes.map import MapNode
    from runnable import defaults
    from runnable.context import PipelineContext

    # Setup mock context
    mock_context = Mock(spec=PipelineContext)
    mock_run_log_store = Mock()
    mock_context.run_log_store = mock_run_log_store
    mock_context.run_id = "test_run"
    mock_get_context.return_value = mock_context

    # Mock branch parameters
    branch_1_params = {
        "result": JsonParameter(kind="json", value="value_1"),
        "metric": JsonParameter(kind="json", value=10)
    }
    branch_2_params = {
        "result": JsonParameter(kind="json", value="value_2"),
        "metric": JsonParameter(kind="json", value=20)
    }

    def get_parameters_side_effect(run_id, internal_branch_name=None):
        if internal_branch_name == "map_node.0":
            return branch_1_params
        elif internal_branch_name == "map_node.1":
            return branch_2_params
        # Return iterate_on for root context
        return {"items": JsonParameter(kind="json", value=["0", "1"])}

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

    # Create a minimal branch graph
    mock_branch = MagicMock(spec=Graph)
    mock_branch.nodes = {}

    # Create map node
    map_node = MapNode(
        name="map_node",
        internal_name="map_node",
        internal_branch_name="",  # Empty string, not None
        iterate_on="items",
        iterate_as="item",
        next_node="success",  # Required field
        branch=mock_branch,
        reducer=None  # Use default reducer: lambda *x: list(x)
    )

    # Call fan_in - context will be retrieved via patched get_run_context
    map_node.fan_in(iter_variable=None)

    # Verify parameters were fetched from branch partitions
    assert mock_run_log_store.get_parameters.call_count >= 2
    calls = [call[1] for call in mock_run_log_store.get_parameters.call_args_list]
    assert any(call.get("internal_branch_name") == "map_node.0" for call in calls)
    assert any(call.get("internal_branch_name") == "map_node.1" for call in calls)

    # Verify aggregated parameters were set to parent partition
    mock_run_log_store.set_parameters.assert_called()
    call_args = mock_run_log_store.set_parameters.call_args

    assert call_args[1]["run_id"] == "test_run"
    # The parent context should be "" (empty string) or None depending on implementation
    assert call_args[1].get("internal_branch_name") in [None, ""]

    # Check aggregated parameters
    aggregated_params = call_args[1]["parameters"]
    assert "result" in aggregated_params
    assert "metric" in aggregated_params
    assert aggregated_params["result"].value == ["value_1", "value_2"]
    assert aggregated_params["metric"].value == [10, 20]


def test_fan_out_creates_branch_partitions_only():
    """Test fan_out only creates branch logs without parameter tracking."""
    from extensions.nodes.map import MapNode
    from runnable import exceptions, defaults
    from runnable.context import PipelineContext

    # Setup mock context
    mock_context = Mock(spec=PipelineContext)
    mock_run_log_store = Mock()
    mock_context.run_log_store = mock_run_log_store
    mock_context.run_id = "test_run"

    # Mock get_parameters to return iterate_on
    mock_run_log_store.get_parameters.return_value = {
        "items": JsonParameter(kind="json", value=["0", "1"])
    }

    # Mock branch log creation - side_effect needs to be an exception instance or callable
    def get_branch_log_side_effect(internal_branch_name, run_id):
        raise exceptions.BranchLogNotFoundError(run_id=run_id, branch_name=internal_branch_name)

    mock_run_log_store.get_branch_log.side_effect = get_branch_log_side_effect

    mock_branch_log = Mock()
    mock_branch_log.status = defaults.PROCESSING
    mock_run_log_store.create_branch_log.return_value = mock_branch_log

    # Create a minimal branch graph
    mock_branch = MagicMock(spec=Graph)
    mock_branch.nodes = {}

    # Create map node
    map_node = MapNode(
        name="map_node",
        internal_name="map_node",
        internal_branch_name="",
        iterate_on="items",
        iterate_as="item",
        next_node="success",
        branch=mock_branch
    )

    # Call fan_out - context will be retrieved via get_run_context
    with patch('runnable.context.get_run_context', return_value=mock_context):
        map_node.fan_out(iter_variable=None)

    # Verify branch logs were created
    assert mock_run_log_store.create_branch_log.call_count == 2
    assert mock_run_log_store.add_branch_log.call_count == 2

    # Verify no raw_parameters were set in fan_out (discovery-based, not tracking)
    # We should NOT have set_parameters calls during fan_out anymore
    mock_run_log_store.set_parameters.assert_not_called()


@patch('runnable.context.get_run_context')
def test_fan_in_only_aggregates_produced_parameters_not_inherited(mock_get_context):
    """Test fan_in only aggregates parameters produced by loop, not inherited from parent."""
    from extensions.nodes.map import MapNode
    from runnable import defaults
    from runnable.context import PipelineContext

    # Setup mock context
    mock_context = Mock(spec=PipelineContext)
    mock_run_log_store = Mock()
    mock_context.run_log_store = mock_run_log_store
    mock_context.run_id = "test_run"
    mock_get_context.return_value = mock_context

    # Parent parameters (should not be aggregated if unchanged)
    parent_params = {
        "config": JsonParameter(kind="json", value="global_config"),
        "base_value": JsonParameter(kind="json", value=100),
        "items": JsonParameter(kind="json", value=[0, 1])
    }

    # Branch parameters (mix of inherited and produced)
    branch_1_params = {
        "config": JsonParameter(kind="json", value="global_config"),  # Same as parent - inherited
        "base_value": JsonParameter(kind="json", value=100),  # Same as parent - inherited
        "result": JsonParameter(kind="json", value="value_1")  # New - should aggregate
    }
    branch_2_params = {
        "config": JsonParameter(kind="json", value="global_config"),  # Same as parent - inherited
        "base_value": JsonParameter(kind="json", value=200),  # Different from parent - should aggregate
        "result": JsonParameter(kind="json", value="value_2")  # New - should aggregate
    }

    def get_parameters_side_effect(run_id, internal_branch_name=None):
        if internal_branch_name is None or internal_branch_name == "":  # Parent partition
            return parent_params
        elif internal_branch_name == "map_node.0":
            return branch_1_params
        elif internal_branch_name == "map_node.1":
            return branch_2_params
        return {}

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

    # Create a minimal branch graph
    mock_branch = MagicMock(spec=Graph)
    mock_branch.nodes = {}

    # Create map node
    map_node = MapNode(
        name="map_node",
        internal_name="map_node",
        internal_branch_name="",  # Root context
        iterate_on="items",
        iterate_as="item",
        next_node="success",
        branch=mock_branch,
        reducer=None  # Use default list reducer
    )

    # Call fan_in
    map_node.fan_in(iter_variable=None)

    # Verify parameters were fetched (parent + 2 branches)
    assert mock_run_log_store.get_parameters.call_count >= 3

    # Verify aggregated parameters were set to parent partition
    mock_run_log_store.set_parameters.assert_called()
    call_args = mock_run_log_store.set_parameters.call_args

    assert call_args[1]["run_id"] == "test_run"
    # Parent context should be None or empty string
    assert call_args[1].get("internal_branch_name") in [None, ""]

    # Check aggregated parameters - should NOT include unchanged inherited ones
    aggregated_params = call_args[1]["parameters"]

    # Should aggregate "result" (new parameter not in parent)
    assert "result" in aggregated_params
    assert aggregated_params["result"].value == ["value_1", "value_2"]

    # Should aggregate "base_value" (modified parameter - one branch changed it)
    assert "base_value" in aggregated_params
    assert aggregated_params["base_value"].value == [100, 200]

    # Should NOT aggregate "config" (inherited unchanged in all branches)
    assert "config" not in aggregated_params
