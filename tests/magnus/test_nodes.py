import pytest

from magnus import (
    defaults,  # pylint: disable=import-error
    nodes,  # pylint: disable=import-error
)


def test_base_node__command_friendly_name_replaces_whitespace_with_character():
    node = nodes.BaseNode(name="test", internal_name="test", config={})

    assert node._command_friendly_name() == "test"

    node.internal_name = "test "
    assert node._command_friendly_name() == "test" + defaults.COMMAND_FRIENDLY_CHARACTER


def test_base_node__get_internal_name_from_command_name_replaces_character_with_whitespace():
    assert nodes.BaseNode._get_internal_name_from_command_name("test") == "test"

    assert nodes.BaseNode._get_internal_name_from_command_name("test%") == "test "


def test_base_node__get_step_log_name_returns_internal_name_if_no_map_variable():
    node = nodes.BaseNode(name="test", internal_name="test", config={})

    assert node._get_step_log_name() == "test"


def test_base_node__get_step_log_name_returns_map_modified_internal_name_if_map_variable():
    node = nodes.BaseNode(name="test", internal_name="test." + defaults.MAP_PLACEHOLDER, config={})

    assert node._get_step_log_name(map_variable={"map_key": "a"}) == "test.a"


def test_base_node__get_step_log_name_returns_map_modified_internal_name_if_map_variable_multiple():
    node = nodes.BaseNode(
        name="test", internal_name="test." + defaults.MAP_PLACEHOLDER + ".step." + defaults.MAP_PLACEHOLDER, config={}
    )

    assert node._get_step_log_name(map_variable={"map_key": "a", "map_key1": "b"}) == "test.a.step.b"


def test_base_node__get_branch_log_name_returns_null_if_not_set():
    node = nodes.BaseNode(name="test", internal_name="test", config={})

    assert node._get_branch_log_name() is None


def test_base_node__get_branch_log_name_returns_internal_name_if_set():
    node = nodes.BaseNode(name="test", internal_name="test", config={}, internal_branch_name="test_internal")

    assert node._get_branch_log_name() == "test_internal"


def test_base_node__get_branch_log_name_returns_map_modified_internal_name_if_map_variable():
    node = nodes.BaseNode(
        name="test", internal_name="test_", config={}, internal_branch_name="test." + defaults.MAP_PLACEHOLDER
    )

    assert node._get_branch_log_name(map_variable={"map_key": "a"}) == "test.a"


def test_base_node__get_branch_log_name_returns_map_modified_internal_name_if_map_variable_multiple():
    node = nodes.BaseNode(
        name="test",
        internal_name="test_",
        config={},
        internal_branch_name="test." + defaults.MAP_PLACEHOLDER + ".step." + defaults.MAP_PLACEHOLDER,
    )

    assert node._get_branch_log_name(map_variable={"map_key": "a", "map_key1": "b"}) == "test.a.step.b"


def test_base_node__get_branch_by_name_raises_exception():
    node = nodes.BaseNode(name="test", internal_name="test", config={})

    with pytest.raises(Exception):
        node._get_branch_by_name("fail")


def test_base_node_execute_raises_not_implemented_error():
    node = nodes.BaseNode(name="test", internal_name="test", config={})

    with pytest.raises(NotImplementedError):
        node.execute(executor="test")


def test_base_node_execute_as_graph_raises_not_implemented_error():
    node = nodes.BaseNode(name="test", internal_name="test", config={})

    with pytest.raises(NotImplementedError):
        node.execute_as_graph(executor="test")


def test_task_node_mocks_if_mock_is_true(mocker, monkeypatch):
    mock_attempt_log = mocker.MagicMock()

    mock_executor = mocker.MagicMock()
    mock_executor.run_log_store.create_attempt_log = mocker.MagicMock(return_value=mock_attempt_log)

    configuration = {"command": "test", "next": "next_node"}
    task_node = nodes.TaskNode(name="test", internal_name="test", config=configuration)

    task_node.execute(executor=mock_executor, mock=True)

    assert mock_attempt_log.status == defaults.SUCCESS


def test_task_node_sets_attempt_log_fail_in_exception_of_execution(mocker, monkeypatch):
    mock_attempt_log = mocker.MagicMock()

    mock_executor = mocker.MagicMock()
    mock_executor.run_log_store.create_attempt_log = mocker.MagicMock(return_value=mock_attempt_log)

    configuration = {"command": "test", "next": "next_node"}
    task_node = nodes.TaskNode(name="test", internal_name="test", config=configuration)

    mock_execution_type = mocker.MagicMock()
    task_node.execution_type = mocker.MagicMock(return_value=mock_execution_type)
    mock_execution_type.execute_command = mocker.MagicMock(side_effect=Exception())
    task_node.execute(executor=mock_executor)

    assert mock_attempt_log.status == defaults.FAIL


def test_task_node_sets_attempt_log_success_in_no_exception_of_execution(mocker, monkeypatch):
    mock_attempt_log = mocker.MagicMock()

    mock_executor = mocker.MagicMock()
    mock_executor.run_log_store.create_attempt_log = mocker.MagicMock(return_value=mock_attempt_log)

    configuration = {"command": "test", "next": "next_node"}
    task_node = nodes.TaskNode(name="test", internal_name="test", config=configuration)

    task_node.executable = mocker.MagicMock()

    task_node.execute(executor=mock_executor)

    assert mock_attempt_log.status == defaults.SUCCESS


def test_task_node_execute_as_graph_raises_exception():
    configuration = {"command": "test", "next": "next_node"}
    task_node = nodes.TaskNode(name="test", internal_name="test", config=configuration)

    with pytest.raises(Exception):
        task_node.execute_as_graph(None)


def test_fail_node_sets_branch_log_fail(mocker, monkeypatch):
    mock_attempt_log = mocker.MagicMock()
    mock_branch_log = mocker.MagicMock()

    mock_executor = mocker.MagicMock()
    mock_executor.run_log_store.create_attempt_log = mocker.MagicMock(return_value=mock_attempt_log)
    mock_executor.run_log_store.get_branch_log = mocker.MagicMock(return_value=mock_branch_log)

    node = nodes.FailNode(name="test", internal_name="test", config={})

    node.execute(executor=mock_executor)

    assert mock_attempt_log.status == defaults.SUCCESS
    assert mock_branch_log.status == defaults.FAIL


def test_fail_node_sets_attempt_log_success_even_in_exception(mocker, monkeypatch):
    mock_attempt_log = mocker.MagicMock()

    mock_executor = mocker.MagicMock()
    mock_executor.run_log_store.create_attempt_log = mocker.MagicMock(return_value=mock_attempt_log)
    mock_executor.run_log_store.get_branch_log = mocker.MagicMock(side_effect=Exception())

    node = nodes.FailNode(name="test", internal_name="test", config={})

    node.execute(executor=mock_executor)

    assert mock_attempt_log.status == defaults.SUCCESS


def test_fail_node_execute_as_graph_raises_exception():
    fail_node = nodes.FailNode(name="test", internal_name="test", config={})

    with pytest.raises(Exception):
        fail_node.execute_as_graph(None)


def test_success_node_sets_branch_log_success(mocker, monkeypatch):
    mock_attempt_log = mocker.MagicMock()
    mock_branch_log = mocker.MagicMock()

    mock_executor = mocker.MagicMock()
    mock_executor.run_log_store.create_attempt_log = mocker.MagicMock(return_value=mock_attempt_log)
    mock_executor.run_log_store.get_branch_log = mocker.MagicMock(return_value=mock_branch_log)

    node = nodes.SuccessNode(name="test", internal_name="test", config={})

    node.execute(executor=mock_executor)

    assert mock_attempt_log.status == defaults.SUCCESS
    assert mock_branch_log.status == defaults.SUCCESS


def test_success_node_sets_attempt_log_success_even_in_exception(mocker, monkeypatch):
    mock_attempt_log = mocker.MagicMock()

    mock_executor = mocker.MagicMock()
    mock_executor.run_log_store.create_attempt_log = mocker.MagicMock(return_value=mock_attempt_log)
    mock_executor.run_log_store.get_branch_log = mocker.MagicMock(side_effect=Exception())

    node = nodes.SuccessNode(name="test", internal_name="test", config={})

    node.execute(executor=mock_executor)

    assert mock_attempt_log.status == defaults.SUCCESS


def test_success_node_execute_as_graph_raises_exception():
    success_node = nodes.SuccessNode(name="test", internal_name="test", config={})

    with pytest.raises(Exception):
        success_node.execute_as_graph(None)


def test_parallel_node_raises_exception_for_empty_branches():
    with pytest.raises(Exception):
        nodes.ParallelNode(name="test", internal_name="test", config={"branches": {}}, execution_type="python")


def test_parallel_node_get_sub_graphs_creates_graphs(mocker, monkeypatch):
    mock_create_graph = mocker.MagicMock(return_value="agraphobject")

    monkeypatch.setattr(nodes, "create_graph", mock_create_graph)

    parallel_config = {"branches": {"a": {}, "b": {}}, "next": "next_node"}
    node = nodes.ParallelNode(name="test", internal_name="test", config=parallel_config)
    assert mock_create_graph.call_count == 2
    assert len(node.branches.items()) == 2


def test_parallel_node__get_branch_by_name_raises_exception_if_branch_not_found(mocker, monkeypatch):
    monkeypatch.setattr(nodes.ParallelNode, "get_sub_graphs", mocker.MagicMock())

    parallel_config = {"branches": {"a": {}, "b": {}}, "next": "next_node"}

    node = nodes.ParallelNode(name="test", internal_name="test", config=parallel_config)

    with pytest.raises(Exception):
        node._get_branch_by_name("a1")


def test_parallel_node__get_branch_by_name_returns_branch_if_found(mocker, monkeypatch):
    monkeypatch.setattr(nodes.ParallelNode, "get_sub_graphs", mocker.MagicMock())

    parallel_config = {"branches": {"a": {}, "b": {}}, "next": "next_node"}

    node = nodes.ParallelNode(name="test", internal_name="test", config=parallel_config)
    node.branches = {"a": "somegraph"}

    assert node._get_branch_by_name("a") == "somegraph"


def test_parallel_node_execute_raises_exception(mocker, monkeypatch):
    monkeypatch.setattr(nodes.ParallelNode, "get_sub_graphs", mocker.MagicMock())

    parallel_config = {"branches": {"a": {}, "b": {}}, "next": "next_node"}

    node = nodes.ParallelNode(name="test", internal_name="test", config=parallel_config)

    with pytest.raises(Exception):
        node.execute(executor="test")


def test_nodes_map_node_raises_exception_if_config_not_have_iterate_on():
    map_config = {"branch": {}, "next": "next_node", "iterate_as": "test"}
    with pytest.raises(Exception):
        nodes.MapNode(name="test", internal_name="test", config=map_config)


def test_nodes_map_node_raises_exception_if_config_not_have_iterate_as():
    map_config = {"branch": {}, "next": "next_node", "iterate_on": "test"}
    with pytest.raises(Exception):
        nodes.MapNode(name="test", internal_name="test", config=map_config)


def test_nodes_map_node_names_the_branch_as_defaults_place_holder(monkeypatch, mocker):
    monkeypatch.setattr(nodes.MapNode, "get_sub_graph", mocker.MagicMock())

    map_config = {"branch": {}, "next": "next_node", "iterate_on": "test", "iterate_as": "test"}

    node = nodes.MapNode(name="test", internal_name="test", config=map_config)

    assert node.branch_placeholder_name == defaults.MAP_PLACEHOLDER


def test_nodes_map_get_sub_graph_calls_create_graph_with_correct_naming(mocker, monkeypatch):
    mock_create_graph = mocker.MagicMock()
    monkeypatch.setattr(nodes, "create_graph", mock_create_graph)

    map_config = {"branch": {}, "next": "next_node", "iterate_on": "test", "iterate_as": "test"}

    _ = nodes.MapNode(name="test", internal_name="test", config=map_config)

    mock_create_graph.assert_called_once_with({}, internal_branch_name="test." + defaults.MAP_PLACEHOLDER)


def test_nodes_map__get_branch_by_name_returns_a_sub_graph(mocker, monkeypatch):
    mock_create_graph = mocker.MagicMock(return_value="a")
    monkeypatch.setattr(nodes, "create_graph", mock_create_graph)

    map_config = {"branch": {}, "next": "next_node", "iterate_on": "test", "iterate_as": "test"}
    node = nodes.MapNode(name="test", internal_name="test", config=map_config)

    assert node._get_branch_by_name("anyname") == "a"


def test_nodes_map_node_execute_raises_exception(mocker, monkeypatch):
    monkeypatch.setattr(nodes.MapNode, "get_sub_graph", mocker.MagicMock())

    map_config = {"branch": {}, "next": "next_node", "iterate_on": "test", "iterate_as": "test"}

    node = nodes.MapNode(name="test", internal_name="test", config=map_config)

    with pytest.raises(Exception):
        node.execute("dummy")


def test_nodes_dag_node_raises_exception_if_dag_definition_is_not_present():
    dag_config = {"next": "test"}
    with pytest.raises(Exception):
        nodes.DagNode(name="test", internal_name="test", config=dag_config)


def test_node_dag_node_get_sub_graph_raises_exception_if_dag_block_not_present(mocker, monkeypatch):
    mock_load_yaml = mocker.MagicMock(return_value={})
    monkeypatch.setattr(nodes.utils, "load_yaml", mock_load_yaml)

    dag_config = {"next": "test", "dag_definition": "test"}

    with pytest.raises(Exception):
        nodes.DagNode(name="test", internal_name="test", config=dag_config)


def test_nodes_dag_node_get_sub_graph_calls_create_graph_with_correct_parameters(mocker, monkeypatch):
    mock_load_yaml = mocker.MagicMock(return_value={"dag": "a"})
    mock_create_graph = mocker.MagicMock(return_value="branch")

    monkeypatch.setattr(nodes.utils, "load_yaml", mock_load_yaml)
    monkeypatch.setattr(nodes, "create_graph", mock_create_graph)

    dag_config = {"next": "test", "dag_definition": "test"}

    _ = nodes.DagNode(name="test", internal_name="test", config=dag_config)

    mock_create_graph.assert_called_once_with("a", internal_branch_name="test." + defaults.DAG_BRANCH_NAME)


def test_nodes_dag_node__get_branch_by_name_raises_exception_if_branch_name_is_invalid(mocker, monkeypatch):
    monkeypatch.setattr(nodes.DagNode, "get_sub_graph", mocker.MagicMock(return_value="branch"))

    dag_config = {"next": "test", "dag_definition": "test"}
    node = nodes.DagNode(name="test", internal_name="test", config=dag_config)

    with pytest.raises(Exception):
        node._get_branch_by_name("test")


def test_nodes_dag_node_get_branch_by_name_returns_if_branch_name_is_valid(mocker, monkeypatch):
    monkeypatch.setattr(nodes.DagNode, "get_sub_graph", mocker.MagicMock(return_value="branch"))

    dag_config = {"next": "test", "dag_definition": "test"}

    node = nodes.DagNode(name="test", internal_name="test", config=dag_config)

    assert node._get_branch_by_name("test." + defaults.DAG_BRANCH_NAME) == "branch"


def test_nodes_dag_node_execute_raises_exception(mocker, monkeypatch):
    monkeypatch.setattr(nodes.DagNode, "get_sub_graph", mocker.MagicMock(return_value="branch"))

    dag_config = {"next": "test", "dag_definition": "test"}

    node = nodes.DagNode(name="test", internal_name="test", config=dag_config)

    with pytest.raises(Exception):
        node.execute("dummy")


def test_nodes_as_is_node_accepts_what_is_given():
    node = nodes.AsIsNode(
        name="test", internal_name="test", config={"command_config": {"render_string": "test"}, "next": "test"}
    )

    assert node.config.command_config == {"render_string": "test"}


def test_as_is_node_execute_as_graph_raises_exception():
    as_is_node = nodes.AsIsNode(name="test", internal_name="test", config={"command": "nocommand", "next": "test"})

    with pytest.raises(Exception):
        as_is_node.execute_as_graph(None)


def test_as_is_node_sets_attempt_log_success(mocker, monkeypatch):
    mock_attempt_log = mocker.MagicMock()

    mock_executor = mocker.MagicMock()
    mock_executor.run_log_store.create_attempt_log = mocker.MagicMock(return_value=mock_attempt_log)

    node = nodes.AsIsNode(name="test", internal_name="test", config={"next": "test"})

    node.execute(executor=mock_executor)

    assert mock_attempt_log.status == defaults.SUCCESS
