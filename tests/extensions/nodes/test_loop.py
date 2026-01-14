import pytest
from extensions.nodes.loop import LoopNode
from runnable.graph import Graph, create_graph


def test_loop_node_creation():
    """Test basic LoopNode creation and attributes."""
    # Use a simple mock graph with required fields
    branch = Graph(start_at="dummy", name="test_branch")

    loop = LoopNode(
        name="testloop",
        internal_name="testloop",
        next_node="success",
        branch=branch,
        max_iterations=5,
        break_on="shouldstop",
        index_as="iteration"
    )

    assert loop.name == "testloop"
    assert loop.node_type == "loop"
    assert loop.max_iterations == 5
    assert loop.break_on == "shouldstop"
    assert loop.index_as == "iteration"
    assert loop.branch == branch


def test_loop_node_branch_name_generation():
    """Test loop node generates correct branch names using placeholders."""
    from runnable.defaults import IterableParameterModel, LoopIndexModel, LOOP_PLACEHOLDER

    branch = Graph(start_at="dummy", name="test_branch")
    loop = LoopNode(
        name="testloop",
        internal_name="testloop",
        next_node="success",
        branch=branch,
        max_iterations=3,
        break_on="done",
        index_as="idx"
    )

    # Should use LOOP_PLACEHOLDER in branch name template
    base_template = f"testloop.{LOOP_PLACEHOLDER}"

    # Mock iter_variable for iteration 2
    iter_var = IterableParameterModel()
    iter_var.loop_variable = [LoopIndexModel(value=2)]

    result = loop._get_iteration_branch_name(iter_var)
    assert result == "testloop.2"


def test_get_break_condition_value():
    """Test reading break condition from parameters."""
    from runnable.datastore import JsonParameter
    from runnable.defaults import IterableParameterModel, LoopIndexModel
    from runnable.context import PipelineContext
    from unittest.mock import Mock, patch

    # Mock context and run_log_store
    mock_context = Mock(spec=PipelineContext)
    mock_run_log_store = Mock()
    mock_context.run_log_store = mock_run_log_store
    mock_context.run_id = "test-run-123"

    # Set up parameters
    parameters = {
        "shouldstop": JsonParameter(value=False, kind="json")
    }
    mock_run_log_store.get_parameters.return_value = parameters

    loop = LoopNode(
        name="test_loop",
        internal_name="test_loop",
        next_node="success",
        branch=Graph(start_at="dummy", name="test_branch"),
        max_iterations=5,
        break_on="shouldstop",
        index_as="iteration"
    )

    # Create iter_variable for iteration 1
    iter_var = IterableParameterModel()
    iter_var.loop_variable = [LoopIndexModel(value=1)]

    with patch("runnable.context.get_run_context", return_value=mock_context):
        result = loop.get_break_condition_value(iter_var)
        assert result is False
        mock_run_log_store.get_parameters.assert_called_with(
            run_id="test-run-123",
            internal_branch_name="test_loop.1"
        )


def test_create_iteration_branch_log():
    """Test creating branch logs with proper iteration naming."""
    from runnable.defaults import IterableParameterModel, LoopIndexModel
    from runnable.context import PipelineContext
    from unittest.mock import Mock, patch

    mock_context = Mock(spec=PipelineContext)
    mock_run_log_store = Mock()
    mock_context.run_log_store = mock_run_log_store
    mock_context.run_id = "test-run-456"

    # Mock branch log creation
    mock_branch_log = Mock()
    mock_run_log_store.create_branch_log.return_value = mock_branch_log
    # Make get_branch_log raise an exception to force creation path
    mock_run_log_store.get_branch_log.side_effect = Exception("Branch log not found")

    loop = LoopNode(
        name="test_loop",
        internal_name="test_loop",
        next_node="success",
        branch=Graph(start_at="dummy", name="test_branch"),
        max_iterations=3,
        break_on="done",
        index_as="idx"
    )

    # Create iter_variable for iteration 1
    iter_var = IterableParameterModel()
    iter_var.loop_variable = [LoopIndexModel(value=1)]

    with patch("runnable.context.get_run_context", return_value=mock_context):
        loop._create_iteration_branch_log(iter_var)

        # Should create branch log with resolved name
        expected_name = "test_loop.1"
        mock_run_log_store.create_branch_log.assert_called_with(expected_name)
        mock_run_log_store.add_branch_log.assert_called_with(mock_branch_log, "test-run-456")


def test_build_iteration_iter_variable():
    """Test building iteration iter_variable with proper loop index."""
    from runnable.defaults import IterableParameterModel, LoopIndexModel

    loop = LoopNode(
        name="test_loop",
        internal_name="test_loop",
        next_node="success",
        branch=Graph(start_at="dummy", name="test_branch"),
        max_iterations=5,
        break_on="done",
        index_as="idx"
    )

    # Test with None parent_iter_variable
    result = loop._build_iteration_iter_variable(None, 2)
    assert isinstance(result, IterableParameterModel)
    assert len(result.loop_variable) == 1
    assert result.loop_variable[0].value == 2

    # Test with existing parent_iter_variable
    parent_iter_var = IterableParameterModel()
    parent_iter_var.loop_variable = [LoopIndexModel(value=1)]

    result = loop._build_iteration_iter_variable(parent_iter_var, 3)
    assert isinstance(result, IterableParameterModel)
    assert len(result.loop_variable) == 2
    assert result.loop_variable[0].value == 1  # Original
    assert result.loop_variable[1].value == 3  # New iteration
