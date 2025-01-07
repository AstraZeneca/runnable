import pytest

from runnable import (  # pylint: disable=import-error  # pylint: disable=import-error
    defaults,
    exceptions,
    nodes,
)


@pytest.fixture(autouse=True)
def instantiable_base_class(monkeypatch):
    monkeypatch.setattr(nodes.BaseNode, "__abstractmethods__", set())
    yield


@pytest.fixture()
def instantiable_traversal_node(monkeypatch):
    monkeypatch.setattr(nodes.TraversalNode, "__abstractmethods__", set())
    yield


@pytest.fixture()
def instantiable_executable_node(monkeypatch):
    monkeypatch.setattr(nodes.ExecutableNode, "__abstractmethods__", set())
    yield


@pytest.fixture()
def instantiable_composite_node(monkeypatch):
    monkeypatch.setattr(nodes.CompositeNode, "__abstractmethods__", set())
    yield


@pytest.fixture()
def instantiable_terminal_node(monkeypatch):
    monkeypatch.setattr(nodes.TerminalNode, "__abstractmethods__", set())
    yield


def test_base_run_log_store_context_property(
    mocker, monkeypatch, instantiable_base_class
):
    mock_run_context = mocker.Mock()

    monkeypatch.setattr(nodes.context, "run_context", mock_run_context)

    assert (
        nodes.BaseNode(node_type="dummy", name="test", internal_name="")._context
        == mock_run_context
    )


def test_validate_name_for_dot(instantiable_base_class):
    with pytest.raises(ValueError):
        nodes.BaseNode(name="test.", internal_name="test", node_type="dummy")


def test_validate_name_for_percent(instantiable_base_class):
    with pytest.raises(ValueError):
        nodes.BaseNode(name="test%", internal_name="test", node_type="dummy")


def test_base_node__command_friendly_name_replaces_whitespace_with_character():
    node = nodes.BaseNode(name="test", internal_name="test", node_type="dummy")

    assert node._command_friendly_name() == "test"

    node.internal_name = "test "
    assert node._command_friendly_name() == "test" + defaults.COMMAND_FRIENDLY_CHARACTER


def test_base_node__get_internal_name_from_command_name_replaces_character_with_whitespace():
    assert nodes.BaseNode._get_internal_name_from_command_name("test") == "test"

    assert nodes.BaseNode._get_internal_name_from_command_name("test%") == "test "


def test_base_node__get_step_log_name_returns_internal_name_if_no_map_variable():
    node = nodes.BaseNode(name="test", internal_name="test", node_type="dummy")

    assert node._get_step_log_name() == "test"


def test_base_node__get_step_log_name_returns_map_modified_internal_name_if_map_variable():
    node = nodes.BaseNode(
        name="test", internal_name="test." + defaults.MAP_PLACEHOLDER, node_type="dummy"
    )

    assert node._get_step_log_name(map_variable={"map_key": "a"}) == "test.a"


def test_base_node__get_step_log_name_returns_map_modified_internal_name_if_map_variable_multiple():
    node = nodes.BaseNode(
        name="test",
        internal_name="test."
        + defaults.MAP_PLACEHOLDER
        + ".step."
        + defaults.MAP_PLACEHOLDER,
        node_type="dummy",
    )

    assert (
        node._get_step_log_name(map_variable={"map_key": "a", "map_key1": "b"})
        == "test.a.step.b"
    )


def test_base_node__get_branch_log_name_returns_null_if_not_set():
    node = nodes.BaseNode(name="test", internal_name="test", node_type="dummy")

    assert node._get_branch_log_name() is ""


def test_base_node__get_branch_log_name_returns_internal_name_if_set():
    node = nodes.BaseNode(
        name="test",
        internal_name="test",
        internal_branch_name="test_internal",
        node_type="dummy",
    )

    assert node._get_branch_log_name() == "test_internal"


def test_base_node__get_branch_log_name_returns_map_modified_internal_name_if_map_variable():
    node = nodes.BaseNode(
        name="test",
        internal_name="test_",
        internal_branch_name="test." + defaults.MAP_PLACEHOLDER,
        node_type="dummy",
    )

    assert node._get_branch_log_name(map_variable={"map_key": "a"}) == "test.a"


def test_base_node__get_branch_log_name_returns_map_modified_internal_name_if_map_variable_multiple():
    node = nodes.BaseNode(
        name="test",
        internal_name="test_",
        internal_branch_name="test."
        + defaults.MAP_PLACEHOLDER
        + ".step."
        + defaults.MAP_PLACEHOLDER,
        node_type="dummy",
    )

    assert (
        node._get_branch_log_name(map_variable={"map_key": "a", "map_key1": "b"})
        == "test.a.step.b"
    )


def test_traversal_node_get_on_failure_node_returns_from_config(
    instantiable_traversal_node,
):
    traversal_class = nodes.TraversalNode(
        name="test",
        internal_name="test",
        node_type="test",
        next_node="next",
        on_failure="on_failure",
    )

    assert traversal_class._get_on_failure_node() == "on_failure"


def test_traversal_node_get_next_node_returns_from_config(instantiable_traversal_node):
    traversal_class = nodes.TraversalNode(
        name="test",
        internal_name="test",
        node_type="test",
        next_node="next",
        on_failure="on_failure",
    )

    assert traversal_class._get_next_node() == "next"


def test_traversal_node_is_terminal_node_is_false(instantiable_traversal_node):
    traversal_class = nodes.TraversalNode(
        name="test",
        internal_name="test",
        node_type="test",
        next_node="next",
        on_failure="on_failure",
    )

    assert traversal_class._is_terminal_node() is False


def test_traversal_node_get_executor_config_defaults_to_empty_dict(
    instantiable_traversal_node,
):
    traversal_class = nodes.TraversalNode(
        name="test",
        internal_name="test",
        node_type="test",
        next_node="next",
        on_failure="on_failure",
    )

    assert traversal_class._get_executor_config("I do not exist") == ""


def test_traversal_node_get_executor_returns_configured_config(
    instantiable_traversal_node,
):
    traversal_class = nodes.TraversalNode(
        name="test",
        internal_name="test",
        node_type="test",
        next_node="next",
        on_failure="on_failure",
        overrides={"test": "key"},
    )

    assert traversal_class._get_executor_config("test") == "key"


def test_executable_node_get_catalog_detaults_to_empty(instantiable_executable_node):
    traversal_class = nodes.ExecutableNode(
        name="test",
        internal_name="test",
        node_type="test",
        next_node="next",
        on_failure="on_failure",
    )

    assert traversal_class._get_catalog_settings() == {}


def test_executable_node_get_max_attempts_from_config(instantiable_executable_node):
    traversal_class = nodes.ExecutableNode(
        name="test",
        internal_name="test",
        node_type="test",
        next_node="next",
        on_failure="on_failure",
        max_attempts=10,
    )

    assert traversal_class._get_max_attempts() == 10


def test_executable_node_get_catalog_detaults_to_1(instantiable_executable_node):
    traversal_class = nodes.ExecutableNode(
        name="test",
        internal_name="test",
        node_type="test",
        next_node="next",
        on_failure="on_failure",
    )

    assert traversal_class._get_max_attempts() == 1


def test_executable_node_get_branch_by_name_raises_exception(
    instantiable_executable_node,
):
    traversal_class = nodes.ExecutableNode(
        name="test",
        internal_name="test",
        node_type="test",
        next_node="next",
    )

    with pytest.raises(exceptions.NodeMethodCallError) as execinfo:
        traversal_class._get_branch_by_name("test")

    assert "This is an executable node and" in str(execinfo.value.message)


def test_executable_node_execute_as_graph_raises_exception(
    instantiable_executable_node,
):
    traversal_class = nodes.ExecutableNode(
        name="test",
        internal_name="test",
        node_type="test",
        next_node="next",
    )

    with pytest.raises(exceptions.NodeMethodCallError) as execinfo:
        traversal_class.execute_as_graph()
    assert "This is an executable node and" in str(execinfo.value.message)


def test_executable_node_fan_in_raises_exception(instantiable_executable_node):
    traversal_class = nodes.ExecutableNode(
        name="test",
        internal_name="test",
        node_type="test",
        next_node="next",
    )

    with pytest.raises(exceptions.NodeMethodCallError) as execinfo:
        traversal_class.fan_in()
    assert "This is an executable node and" in str(execinfo.value.message)


def test_executable_node_fan_out_raises_exception(instantiable_executable_node):
    traversal_class = nodes.ExecutableNode(
        name="test",
        internal_name="test",
        node_type="test",
        next_node="next",
    )

    with pytest.raises(exceptions.NodeMethodCallError) as execinfo:
        traversal_class.fan_out()

    assert "This is an executable node and" in str(execinfo.value.message)


def test_composite_node_get_catalog_settings_raises_exception(
    instantiable_composite_node,
):
    traversal_class = nodes.CompositeNode(
        name="test",
        internal_name="test",
        node_type="test",
        next_node="next",
    )

    with pytest.raises(exceptions.NodeMethodCallError) as execinfo:
        traversal_class._get_catalog_settings()

    assert "This is a composite node and" in str(execinfo.value.message)


def test_composite_node_get_max_attempts_raises_exception(instantiable_composite_node):
    traversal_class = nodes.CompositeNode(
        name="test",
        internal_name="test",
        node_type="test",
        next_node="next",
    )

    with pytest.raises(Exception, match="This is a composite node and"):
        traversal_class._get_max_attempts()


def test_composite_node_execute_raises_exception(instantiable_composite_node):
    traversal_class = nodes.CompositeNode(
        name="test",
        internal_name="test",
        node_type="test",
        next_node="next",
    )

    with pytest.raises(exceptions.NodeMethodCallError) as execinfo:
        traversal_class.execute()
    assert "This is a composite node and" in str(execinfo.value.message)


def test_terminal_node_get_on_failure_node_raises_exception(instantiable_terminal_node):
    node = nodes.TerminalNode(name="test", internal_name="test", node_type="dummy")

    assert node._get_on_failure_node() == ""


def test_terminal_node__get_next_node_raises_exception(instantiable_terminal_node):
    node = nodes.TerminalNode(name="test", internal_name="test", node_type="dummy")

    with pytest.raises(exceptions.TerminalNodeError):
        node._get_next_node()


def test_terminal_node__get_catalog_settings_raises_exception(
    instantiable_terminal_node,
):
    node = nodes.TerminalNode(name="test", internal_name="test", node_type="dummy")

    with pytest.raises(exceptions.TerminalNodeError):
        node._get_catalog_settings()


def test_terminal_node__get_branch_by_name_raises_exception(instantiable_terminal_node):
    node = nodes.TerminalNode(name="test", internal_name="test", node_type="dummy")

    with pytest.raises(exceptions.TerminalNodeError):
        node._get_branch_by_name("does not matter")


def test_terminal_node__get_executor_config_raises_exception(
    instantiable_terminal_node,
):
    node = nodes.TerminalNode(name="test", internal_name="test", node_type="dummy")

    with pytest.raises(exceptions.TerminalNodeError):
        node._get_executor_config("does not matter")


def test_terminal_node_execute_as_graph_raises_exception(instantiable_terminal_node):
    node = nodes.TerminalNode(name="test", internal_name="test", node_type="dummy")

    with pytest.raises(exceptions.TerminalNodeError):
        node.execute_as_graph()


def test_terminal_node_fan_out_raises_exception(instantiable_terminal_node):
    node = nodes.TerminalNode(name="test", internal_name="test", node_type="dummy")

    with pytest.raises(exceptions.TerminalNodeError):
        node.fan_out()


def test_terminal_node_fan_in_raises_exception(instantiable_terminal_node):
    node = nodes.TerminalNode(name="test", internal_name="test", node_type="dummy")

    with pytest.raises(exceptions.TerminalNodeError):
        node.fan_in()


def test_terminal_node_max_attempts_returns_1(instantiable_terminal_node):
    node = nodes.TerminalNode(name="test", internal_name="test", node_type="dummy")

    assert node._get_max_attempts() == 1


def test_terminal_node_is_terminal_node_returns_true(instantiable_terminal_node):
    node = nodes.TerminalNode(name="test", internal_name="test", node_type="dummy")

    assert node._is_terminal_node()


def test_terminal_node_parse_from_config_sends_the_config_for_instantiation(
    instantiable_terminal_node,
):
    config = {
        "node_type": "dummy",
        "name": "test",
        "internal_name": "test",
    }

    node = nodes.TerminalNode.parse_from_config(config)
    assert node.node_type == "dummy"
    assert node.name == "test"
    assert node.internal_name == "test"
