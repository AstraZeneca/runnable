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
