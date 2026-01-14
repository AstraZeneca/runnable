import pytest

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
        index_as="iteration",
    )

    assert loop.name == "testloop"
    assert loop.node_type == "loop"
    assert loop.max_iterations == 5
    assert loop.break_on == "shouldstop"
    assert loop.index_as == "iteration"
    assert loop.branch == branch


def test_loop_node_branch_name_generation():
    """Test loop node generates correct branch names using placeholders."""
    from runnable.defaults import (
        LOOP_PLACEHOLDER,
        IterableParameterModel,
        LoopIndexModel,
    )

    branch = Graph(start_at="dummy", name="test_branch")
    loop = LoopNode(
        name="testloop",
        internal_name="testloop",
        next_node="success",
        branch=branch,
        max_iterations=3,
        break_on="done",
        index_as="idx",
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
    from unittest.mock import Mock, patch

    from runnable.context import PipelineContext
    from runnable.datastore import JsonParameter
    from runnable.defaults import IterableParameterModel, LoopIndexModel

    # Mock context and run_log_store
    mock_context = Mock(spec=PipelineContext)
    mock_run_log_store = Mock()
    mock_context.run_log_store = mock_run_log_store
    mock_context.run_id = "test-run-123"

    # Set up parameters
    parameters = {"shouldstop": JsonParameter(value=False, kind="json")}
    mock_run_log_store.get_parameters.return_value = parameters

    loop = LoopNode(
        name="test_loop",
        internal_name="test_loop",
        next_node="success",
        branch=Graph(start_at="dummy", name="test_branch"),
        max_iterations=5,
        break_on="shouldstop",
        index_as="iteration",
    )

    # Create iter_variable for iteration 1
    iter_var = IterableParameterModel()
    iter_var.loop_variable = [LoopIndexModel(value=1)]

    with patch("runnable.context.get_run_context", return_value=mock_context):
        result = loop.get_break_condition_value(iter_var)
        assert result is False
        mock_run_log_store.get_parameters.assert_called_with(
            run_id="test-run-123", internal_branch_name="test_loop.1"
        )


def test_create_iteration_branch_log():
    """Test creating branch logs with proper iteration naming."""
    from unittest.mock import Mock, patch

    from runnable.context import PipelineContext
    from runnable.defaults import IterableParameterModel, LoopIndexModel

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
        index_as="idx",
    )

    # Create iter_variable for iteration 1
    iter_var = IterableParameterModel()
    iter_var.loop_variable = [LoopIndexModel(value=1)]

    with patch("runnable.context.get_run_context", return_value=mock_context):
        loop._create_iteration_branch_log(iter_var)

        # Should create branch log with resolved name
        expected_name = "test_loop.1"
        mock_run_log_store.create_branch_log.assert_called_with(expected_name)
        mock_run_log_store.add_branch_log.assert_called_with(
            mock_branch_log, "test-run-456"
        )


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
        index_as="idx",
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


def test_fan_out_initial_iteration():
    """Test fan_out creates branch log and copies parent parameters."""
    from unittest.mock import Mock, patch

    from runnable.context import PipelineContext

    mock_run_log_store = Mock()
    mock_context = Mock(spec=PipelineContext)
    mock_context.run_log_store = mock_run_log_store
    mock_context.run_id = "test-run-123"

    # Mock parent parameters
    parent_params = {"param1": Mock()}
    mock_run_log_store.get_parameters.return_value = parent_params

    loop = LoopNode(
        name="test_loop",
        internal_name="test_loop",
        next_node="success",
        branch=Graph(start_at="dummy", name="test_branch"),
        max_iterations=5,
        break_on="shouldstop",
        index_as="iteration",
    )
    loop.internal_branch_name = "root"

    # Create iter_variable for iteration 0
    from runnable.defaults import IterableParameterModel, LoopIndexModel

    iter_var = IterableParameterModel()
    iter_var.loop_variable = [LoopIndexModel(value=0)]

    with (
        patch("runnable.context.get_run_context", return_value=mock_context),
        patch.object(loop, "_create_iteration_branch_log") as mock_create_branch,
    ):
        loop.fan_out(iter_var)

    # Should create branch log
    mock_create_branch.assert_called_once_with(iter_var)

    # Should copy parent parameters to iteration branch
    mock_run_log_store.set_parameters.assert_called_once_with(
        parameters=parent_params,
        run_id="test-run-123",
        internal_branch_name="test_loop.0",
    )


def test_fan_out_subsequent_iteration():
    """Test fan_out copies from previous iteration."""
    from unittest.mock import Mock, patch

    from runnable.context import PipelineContext
    from runnable.defaults import IterableParameterModel, LoopIndexModel

    mock_run_log_store = Mock()
    mock_context = Mock(spec=PipelineContext)
    mock_context.run_log_store = mock_run_log_store
    mock_context.run_id = "test-run-123"

    # Mock previous iteration parameters
    prev_params = {"param1": Mock(), "result": Mock()}
    mock_run_log_store.get_parameters.return_value = prev_params

    loop = LoopNode(
        name="test_loop",
        internal_name="test_loop",
        next_node="success",
        branch=Graph(start_at="dummy", name="test_branch"),
        max_iterations=5,
        break_on="shouldstop",
        index_as="iteration",
    )

    # Create iter_variable for iteration 2
    iter_var = IterableParameterModel()
    iter_var.loop_variable = [LoopIndexModel(value=2)]

    with (
        patch("runnable.context.get_run_context", return_value=mock_context),
        patch.object(loop, "_create_iteration_branch_log") as mock_create_branch,
    ):
        loop.fan_out(iter_var)

    # Should get parameters from iteration 1
    prev_iter_var = IterableParameterModel()
    prev_iter_var.loop_variable = [LoopIndexModel(value=1)]
    expected_prev_name = "test_loop.1"

    mock_run_log_store.get_parameters.assert_called_with(
        run_id="test-run-123", internal_branch_name=expected_prev_name
    )

    # Should copy to iteration 2 branch
    mock_run_log_store.set_parameters.assert_called_once_with(
        parameters=prev_params,
        run_id="test-run-123",
        internal_branch_name="test_loop.2",
    )


def test_fan_in_should_continue():
    """Test fan_in returns False when break condition not met."""
    from unittest.mock import Mock, patch

    from runnable.context import PipelineContext
    from runnable.datastore import JsonParameter
    from runnable.defaults import IterableParameterModel, LoopIndexModel

    mock_context = Mock(spec=PipelineContext)
    mock_run_log_store = Mock()
    mock_context.run_log_store = mock_run_log_store
    mock_context.run_id = "test-run-123"

    # Break condition is False, should continue
    parameters = {"shouldstop": JsonParameter(value=False, kind="json")}
    mock_run_log_store.get_parameters.return_value = parameters

    # Mock successful branch log
    mock_branch_log = Mock()
    mock_branch_log.status = "SUCCESS"
    mock_run_log_store.get_branch_log.return_value = mock_branch_log

    loop = LoopNode(
        name="test_loop",
        internal_name="test_loop",
        next_node="success",
        branch=Graph(start_at="dummy", name="test_branch"),
        max_iterations=5,
        break_on="shouldstop",
        index_as="iteration",
    )

    # Iteration 1 (0-indexed), not at max yet
    iter_var = IterableParameterModel()
    iter_var.loop_variable = [LoopIndexModel(value=1)]

    with patch("runnable.context.get_run_context", return_value=mock_context):
        loop.fan_in(iter_var)

    assert loop._should_exit is False  # Should continue looping


def test_fan_in_should_exit_break_condition():
    """Test fan_in returns True when break condition is met."""
    from unittest.mock import Mock, patch

    from runnable.context import PipelineContext
    from runnable.datastore import JsonParameter
    from runnable.defaults import IterableParameterModel, LoopIndexModel

    mock_context = Mock(spec=PipelineContext)
    mock_run_log_store = Mock()
    mock_context.run_log_store = mock_run_log_store
    mock_context.run_id = "test-run-123"

    # Break condition is True, should exit
    parameters = {"shouldstop": JsonParameter(value=True, kind="json")}
    mock_run_log_store.get_parameters.return_value = parameters

    # Mock successful branch log
    mock_branch_log = Mock()
    mock_branch_log.status = "SUCCESS"
    mock_run_log_store.get_branch_log.return_value = mock_branch_log

    loop = LoopNode(
        name="test_loop",
        internal_name="test_loop",
        next_node="success",
        branch=Graph(start_at="dummy", name="test_branch"),
        max_iterations=5,
        break_on="shouldstop",
        index_as="iteration",
    )

    iter_var = IterableParameterModel()
    iter_var.loop_variable = [LoopIndexModel(value=2)]

    with (
        patch("runnable.context.get_run_context", return_value=mock_context),
        patch.object(loop, "_rollback_parameters_to_parent") as mock_rollback,
        patch.object(loop, "_set_final_step_status") as mock_set_status,
    ):
        loop.fan_in(iter_var)

        assert loop._should_exit is True
        mock_rollback.assert_called_once_with(iter_var)
        mock_set_status.assert_called_once_with(iter_var)


def test_fan_in_should_exit_max_iterations():
    """Test fan_in returns True when max iterations reached."""
    from unittest.mock import Mock, patch

    from runnable.context import PipelineContext
    from runnable.datastore import JsonParameter
    from runnable.defaults import IterableParameterModel, LoopIndexModel

    mock_context = Mock(spec=PipelineContext)
    mock_run_log_store = Mock()
    mock_context.run_log_store = mock_run_log_store
    mock_context.run_id = "test-run-123"

    # Break condition is False but max iterations reached
    parameters = {"shouldstop": JsonParameter(value=False, kind="json")}
    mock_run_log_store.get_parameters.return_value = parameters

    # Mock successful branch log
    mock_branch_log = Mock()
    mock_branch_log.status = "SUCCESS"
    mock_run_log_store.get_branch_log.return_value = mock_branch_log

    loop = LoopNode(
        name="test_loop",
        internal_name="test_loop",
        next_node="success",
        branch=Graph(start_at="dummy", name="test_branch"),
        max_iterations=3,  # 0, 1, 2 (3 iterations)
        break_on="shouldstop",
        index_as="iteration",
    )

    # Iteration 2 (0-indexed) = 3rd iteration = max reached
    iter_var = IterableParameterModel()
    iter_var.loop_variable = [LoopIndexModel(value=2)]

    with (
        patch("runnable.context.get_run_context", return_value=mock_context),
        patch.object(loop, "_rollback_parameters_to_parent") as mock_rollback,
        patch.object(loop, "_set_final_step_status") as mock_set_status,
    ):
        loop.fan_in(iter_var)

        assert loop._should_exit is True  # Should exit due to max iterations
        mock_rollback.assert_called_once_with(iter_var)
        mock_set_status.assert_called_once_with(iter_var)


# Tests for execute_as_graph method
def test_execute_as_graph_exits_on_break_condition():
    """Test that execute_as_graph exits when break condition is met."""
    import os
    from unittest.mock import Mock, patch

    from extensions.nodes.loop import LoopNode
    from runnable.context import PipelineContext
    from runnable.datastore import JsonParameter
    from runnable.defaults import IterableParameterModel, LoopIndexModel
    from runnable.graph import Graph

    mock_context = Mock(spec=PipelineContext)
    mock_run_log_store = Mock()
    mock_pipeline_executor = Mock()
    mock_context.run_log_store = mock_run_log_store
    mock_context.pipeline_executor = mock_pipeline_executor
    mock_context.run_id = "test-run-123"

    # Setup parameters - break condition becomes True after iteration 1
    parameters_iter_0 = {"shouldstop": JsonParameter(value=False, kind="json")}
    parameters_iter_1 = {"shouldstop": JsonParameter(value=True, kind="json")}
    mock_run_log_store.get_parameters.side_effect = [
        parameters_iter_0,  # For iteration 0 fan_in
        parameters_iter_1,  # For iteration 1 fan_in
    ]

    loop = LoopNode(
        name="test_loop",
        internal_name="test_loop",
        next_node="success",
        branch=Graph(start_at="dummy", name="test_branch"),
        max_iterations=5,
        break_on="shouldstop",
        index_as="iteration",
    )

    def mock_fan_in_impl(iter_variable=None):
        # First call: continue (don't exit)
        # Second call: exit
        if mock_fan_in.call_count == 1:
            loop._should_exit = False
        else:
            loop._should_exit = True

    with (
        patch("runnable.context.get_run_context", return_value=mock_context),
        patch.object(LoopNode, "fan_out") as mock_fan_out,
        patch.object(LoopNode, "fan_in", side_effect=mock_fan_in_impl) as mock_fan_in,
        patch.object(LoopNode, "_rollback_parameters_to_parent"),
        patch.object(LoopNode, "_set_final_step_status"),
    ):
        # Store original environ value
        original_env = os.environ.get("iteration")

        try:
            loop.execute_as_graph()

            # Should have called fan_out twice (iteration 0 and 1)
            assert mock_fan_out.call_count == 2

            # Should have called pipeline_executor.execute_graph twice
            assert mock_pipeline_executor.execute_graph.call_count == 2

            # Should have called fan_in twice
            assert mock_fan_in.call_count == 2

            # Environment variable should be set to final iteration (1)
            assert os.environ["iteration"] == "1"

        finally:
            # Clean up environment
            if original_env is not None:
                os.environ["iteration"] = original_env
            else:
                os.environ.pop("iteration", None)


def test_execute_as_graph_exits_on_max_iterations():
    """Test that execute_as_graph exits when max_iterations is reached."""
    import os
    from unittest.mock import Mock, patch

    from extensions.nodes.loop import LoopNode
    from runnable.context import PipelineContext
    from runnable.datastore import JsonParameter
    from runnable.defaults import IterableParameterModel, LoopIndexModel
    from runnable.graph import Graph

    mock_context = Mock(spec=PipelineContext)
    mock_run_log_store = Mock()
    mock_pipeline_executor = Mock()
    mock_context.run_log_store = mock_run_log_store
    mock_context.pipeline_executor = mock_pipeline_executor
    mock_context.run_id = "test-run-123"

    # Setup parameters - break condition never becomes True
    parameters = {"shouldstop": JsonParameter(value=False, kind="json")}
    mock_run_log_store.get_parameters.return_value = parameters

    loop = LoopNode(
        name="test_loop",
        internal_name="test_loop",
        next_node="success",
        branch=Graph(start_at="dummy", name="test_branch"),
        max_iterations=3,  # Should exit after iterations 0, 1, 2
        break_on="shouldstop",
        index_as="iteration",
    )

    with (
        patch("runnable.context.get_run_context", return_value=mock_context),
        patch.object(LoopNode, "fan_out") as mock_fan_out,
        patch.object(
            LoopNode, "fan_in", side_effect=[False, False, True]
        ) as mock_fan_in,
        patch.object(LoopNode, "_rollback_parameters_to_parent"),
        patch.object(LoopNode, "_set_final_step_status"),
    ):
        # Store original environ value
        original_env = os.environ.get("iteration")

        try:
            loop.execute_as_graph()

            # Should have called fan_out three times (iterations 0, 1, 2)
            assert mock_fan_out.call_count == 3

            # Should have called pipeline_executor.execute_graph three times
            assert mock_pipeline_executor.execute_graph.call_count == 3

            # Should have called fan_in three times
            assert mock_fan_in.call_count == 3

            # Environment variable should be set to final iteration (2)
            assert os.environ["iteration"] == "2"

        finally:
            # Clean up environment
            if original_env is not None:
                os.environ["iteration"] = original_env
            else:
                os.environ.pop("iteration", None)


def test_execute_as_graph_builds_correct_iter_variables():
    """Test that execute_as_graph builds correct iter_variables for each iteration."""
    import os
    from unittest.mock import Mock, patch

    from extensions.nodes.loop import LoopNode
    from runnable.context import PipelineContext
    from runnable.datastore import JsonParameter
    from runnable.defaults import IterableParameterModel, LoopIndexModel
    from runnable.graph import Graph

    mock_context = Mock(spec=PipelineContext)
    mock_run_log_store = Mock()
    mock_pipeline_executor = Mock()
    mock_context.run_log_store = mock_run_log_store
    mock_context.pipeline_executor = mock_pipeline_executor
    mock_context.run_id = "test-run-123"

    # Setup parameters - break after iteration 0
    parameters = {"shouldstop": JsonParameter(value=True, kind="json")}
    mock_run_log_store.get_parameters.return_value = parameters

    loop = LoopNode(
        name="test_loop",
        internal_name="test_loop",
        next_node="success",
        branch=Graph(start_at="dummy", name="test_branch"),
        max_iterations=5,
        break_on="shouldstop",
        index_as="iteration",
    )

    # Parent iter_variable to test nesting
    parent_iter_var = IterableParameterModel()
    parent_iter_var.loop_variable = [LoopIndexModel(value=99)]  # Simulate outer loop

    captured_iter_variables = []

    def capture_execute_graph(branch, iter_variable):
        captured_iter_variables.append(iter_variable)

    mock_pipeline_executor.execute_graph.side_effect = capture_execute_graph

    def mock_fan_in_exit_immediately(iter_variable=None):
        # Always exit on first call
        loop._should_exit = True

    with (
        patch("runnable.context.get_run_context", return_value=mock_context),
        patch.object(LoopNode, "fan_out"),
        patch.object(LoopNode, "fan_in", side_effect=mock_fan_in_exit_immediately),
        patch.object(LoopNode, "_rollback_parameters_to_parent"),
        patch.object(LoopNode, "_set_final_step_status"),
    ):
        # Store original environ value
        original_env = os.environ.get("iteration")

        try:
            loop.execute_as_graph(iter_variable=parent_iter_var)

            # Should have executed once
            assert len(captured_iter_variables) == 1

            # Check the iter_variable for iteration 0
            iter_var = captured_iter_variables[0]
            assert iter_var.loop_variable is not None
            assert len(iter_var.loop_variable) == 2  # Parent + current iteration
            assert iter_var.loop_variable[0].value == 99  # Parent iteration
            assert iter_var.loop_variable[1].value == 0  # Current iteration

        finally:
            # Clean up environment
            if original_env is not None:
                os.environ["iteration"] = original_env
            else:
                os.environ.pop("iteration", None)


def test_execute_as_graph_handles_safety_limit():
    """Test that execute_as_graph has a safety check for max_iterations."""
    import os
    from unittest.mock import Mock, patch

    from extensions.nodes.loop import LoopNode
    from runnable.context import PipelineContext
    from runnable.datastore import JsonParameter
    from runnable.defaults import IterableParameterModel, LoopIndexModel
    from runnable.graph import Graph

    mock_context = Mock(spec=PipelineContext)
    mock_run_log_store = Mock()
    mock_pipeline_executor = Mock()
    mock_context.run_log_store = mock_run_log_store
    mock_context.pipeline_executor = mock_pipeline_executor
    mock_context.run_id = "test-run-123"

    # Setup parameters - break condition never becomes True
    parameters = {"shouldstop": JsonParameter(value=False, kind="json")}
    mock_run_log_store.get_parameters.return_value = parameters

    loop = LoopNode(
        name="test_loop",
        internal_name="test_loop",
        next_node="success",
        branch=Graph(start_at="dummy", name="test_branch"),
        max_iterations=2,
        break_on="shouldstop",
        index_as="iteration",
    )

    # Mock fan_in to always return False (never exit)
    # This tests the safety limit in execute_as_graph
    with (
        patch("runnable.context.get_run_context", return_value=mock_context),
        patch.object(LoopNode, "fan_out") as mock_fan_out,
        patch.object(LoopNode, "fan_in", return_value=False) as mock_fan_in,
    ):
        # Store original environ value
        original_env = os.environ.get("iteration")

        try:
            loop.execute_as_graph()

            # Should have called fan_out twice (iterations 0, 1)
            assert mock_fan_out.call_count == 2

            # Should have called pipeline_executor.execute_graph twice
            assert mock_pipeline_executor.execute_graph.call_count == 2

            # Should have called fan_in twice
            assert mock_fan_in.call_count == 2

        finally:
            # Clean up environment
            if original_env is not None:
                os.environ["iteration"] = original_env
            else:
                os.environ.pop("iteration", None)


def test_fan_in_exits_on_branch_failure():
    """Test that fan_in exits immediately when branch execution fails."""
    from unittest.mock import Mock, patch

    from extensions.nodes.loop import LoopNode
    from runnable import defaults
    from runnable.context import PipelineContext
    from runnable.defaults import IterableParameterModel, LoopIndexModel
    from runnable.graph import Graph

    mock_context = Mock(spec=PipelineContext)
    mock_run_log_store = Mock()
    mock_context.run_log_store = mock_run_log_store
    mock_context.run_id = "test-run-123"

    # Mock branch log with FAIL status
    mock_branch_log = Mock()
    mock_branch_log.status = defaults.FAIL
    mock_run_log_store.get_branch_log.return_value = mock_branch_log

    loop = LoopNode(
        name="test_loop",
        internal_name="test_loop",
        next_node="success",
        branch=Graph(start_at="dummy", name="test_branch"),
        max_iterations=5,
        break_on="shouldstop",
        index_as="iteration",
    )

    # Iteration 0 with failed branch
    iter_var = IterableParameterModel()
    iter_var.loop_variable = [LoopIndexModel(value=0)]

    with (
        patch("runnable.context.get_run_context", return_value=mock_context),
        patch.object(loop, "_rollback_parameters_to_parent") as mock_rollback,
        patch.object(loop, "_set_step_status_to_fail") as mock_set_fail,
    ):
        loop.fan_in(iter_var)

        # Should exit due to branch failure
        assert loop._should_exit is True
        mock_rollback.assert_called_once_with(iter_var)
        mock_set_fail.assert_called_once_with(iter_var)


def test_fan_in_continues_on_branch_success():
    """Test that fan_in continues checking conditions when branch succeeds."""
    from unittest.mock import Mock, patch

    from extensions.nodes.loop import LoopNode
    from runnable import defaults
    from runnable.context import PipelineContext
    from runnable.datastore import JsonParameter
    from runnable.defaults import IterableParameterModel, LoopIndexModel
    from runnable.graph import Graph

    mock_context = Mock(spec=PipelineContext)
    mock_run_log_store = Mock()
    mock_context.run_log_store = mock_run_log_store
    mock_context.run_id = "test-run-123"

    # Mock branch log with SUCCESS status
    mock_branch_log = Mock()
    mock_branch_log.status = defaults.SUCCESS
    mock_run_log_store.get_branch_log.return_value = mock_branch_log

    # Mock parameters - break condition is False
    parameters = {"shouldstop": JsonParameter(value=False, kind="json")}
    mock_run_log_store.get_parameters.return_value = parameters

    loop = LoopNode(
        name="test_loop",
        internal_name="test_loop",
        next_node="success",
        branch=Graph(start_at="dummy", name="test_branch"),
        max_iterations=5,
        break_on="shouldstop",
        index_as="iteration",
    )

    # Iteration 0 with successful branch and no break condition
    iter_var = IterableParameterModel()
    iter_var.loop_variable = [LoopIndexModel(value=0)]

    with (
        patch("runnable.context.get_run_context", return_value=mock_context),
        patch.object(loop, "_rollback_parameters_to_parent"),
        patch.object(loop, "_set_step_status_to_fail"),
    ):
        loop.fan_in(iter_var)

        # Should continue (not exit) since branch succeeded and break condition is False
        assert loop._should_exit is False


from extensions.nodes.loop import LoopNode
