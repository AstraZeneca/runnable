import pytest

from extensions.nodes import map, parallel, stub, task
from runnable import defaults
from runnable.graph import Graph
from runnable.tasks import BaseTaskType


@pytest.fixture(autouse=True)
def instantiable_base_class(monkeypatch):
    monkeypatch.setattr(BaseTaskType, "__abstractmethods__", set())
    yield


def test_task_node_parse_from_config_seperates_task_from_node_confifg(
    mocker, monkeypatch
):
    base_task = BaseTaskType(task_type="dummy")
    mock_create_task = mocker.MagicMock(return_value=base_task)

    command_config = {"to_be_sent_to_task": "yes"}
    node_config = {
        "name": "test",
        "node_type": "task",
        "internal_name": "test",
        "next_node": "next_node",
    }
    monkeypatch.setattr(task, "create_task", mock_create_task)
    task_node = task.TaskNode.parse_from_config({**node_config, **command_config})

    mock_create_task.assert_called_once_with(command_config)
    assert task_node.executable == base_task


def test_task_node_mocks_if_mock_is_true(mocker, monkeypatch):
    mock_attempt_log = mocker.MagicMock()
    mock_context = mocker.MagicMock()

    monkeypatch.setattr(task.TaskNode, "_context", mock_context)
    mock_context.run_log_store.create_attempt_log = mocker.MagicMock(
        return_value=mock_attempt_log
    )

    base_task = BaseTaskType(task_type="dummy")
    task_node = task.TaskNode(
        name="test", internal_name="test", next_node="next_node", executable=base_task
    )

    attempt_log = task_node.execute(mock=True)

    assert attempt_log.status == defaults.SUCCESS


def test_parallel_node_parse_from_config_creates_sub_graph(mocker, monkeypatch):
    graph = Graph(start_at="first", name="first_branch")

    mock_create_graph = mocker.MagicMock(return_value=graph)
    config = {
        "branches": {"first": {"name": "first"}, "second": {"name": "second"}},
        "next_node": "next_node",
        "name": "test",
        "internal_name": "parent",
    }
    monkeypatch.setattr(parallel, "create_graph", mock_create_graph)

    parallel_node = parallel.ParallelNode.parse_from_config(config=config)
    assert mock_create_graph.call_count == 2
    assert len(parallel_node.branches.items()) == 2

    for name, branch in parallel_node.branches.items():
        assert name == "parent.first" or name == "parent.second"
        assert branch == graph


def test_parallel_node_parse_from_config_raises_exception_if_no_branches(
    mocker, monkeypatch
):
    config = {
        "branches": {},
        "next_node": "next_node",
        "name": "test",
        "internal_name": "parent",
    }
    with pytest.raises(Exception, match="A parallel node should have branches"):
        _ = parallel.ParallelNode.parse_from_config(config=config)


def test_map_node_parse_from_config_raises_exception_if_no_branch(mocker, monkeypatch):
    config = {}
    with pytest.raises(Exception, match="A map node should have a branch"):
        _ = map.MapNode.parse_from_config(config=config)


def test_map_node_parse_from_config_calls_create_graph(mocker, monkeypatch):
    graph = Graph(start_at="first", name="first_branch")

    mock_create_graph = mocker.MagicMock(return_value=graph)
    config = {
        "branch": {"name": "test"},
        "next_node": "next_node",
        "name": "test",
        "internal_name": "parent",
        "iterate_on": "me",
        "iterate_as": "you",
    }
    monkeypatch.setattr(map, "create_graph", mock_create_graph)
    map_node = map.MapNode.parse_from_config(config=config)

    assert mock_create_graph.call_count == 1
    mock_create_graph.assert_called_once_with(
        {"name": "test"}, internal_branch_name=f"parent.{defaults.MAP_PLACEHOLDER}"
    )
    assert map_node.iterate_as == "you"
    assert map_node.iterate_on == "me"


def test_map_node_get_branch_by_name_returns_branch(mocker, monkeypatch):
    graph = Graph(start_at="first", name="first_branch")

    mock_create_graph = mocker.MagicMock(return_value=graph)
    config = {
        "branch": {"name": "test"},
        "next_node": "next_node",
        "name": "test",
        "internal_name": "parent",
        "iterate_on": "me",
        "iterate_as": "you",
    }
    monkeypatch.setattr(map, "create_graph", mock_create_graph)
    map_node = map.MapNode.parse_from_config(config=config)

    assert map_node._get_branch_by_name("test") == graph


def test__as_is_node_takes_anything_as_input(mocker, monkeypatch):
    config = {
        "name": "test",
        "internal_name": "test",
        "next_node": "next_node",
        "whose": "me",
        "wheres": "them",
    }

    _ = stub.StubNode.parse_from_config(config=config)
