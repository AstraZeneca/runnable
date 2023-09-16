import pytest

from magnus import defaults
from magnus.extensions import nodes as nodes

from magnus.tasks import BaseTaskType


@pytest.fixture(autouse=True)
def instantiable_base_class(monkeypatch):
    monkeypatch.setattr(BaseTaskType, "__abstractmethods__", set())
    yield


def test_task_node_mocks_if_mock_is_true(mocker, monkeypatch):
    mock_attempt_log = mocker.MagicMock()
    mock_context = mocker.MagicMock()

    monkeypatch.setattr(nodes.TaskNode, "_context", mock_context)
    mock_context.run_log_store.create_attempt_log = mocker.MagicMock(return_value=mock_attempt_log)

    base_task = BaseTaskType(node_name="test")
    task_node = nodes.TaskNode(name="test", internal_name="test", next_node="next_node", executable=base_task)

    attempt_log = task_node.execute(mock=True)

    assert attempt_log.status == defaults.SUCCESS


def test_task_node_sets_attempt_log_fail_in_exception_of_execution(mocker, monkeypatch):
    mock_attempt_log = mocker.MagicMock()
    mock_context = mocker.MagicMock()

    monkeypatch.setattr(nodes.TaskNode, "_context", mock_context)
    mock_context.run_log_store.create_attempt_log = mocker.MagicMock(return_value=mock_attempt_log)

    monkeypatch.setattr(BaseTaskType, "execute_command", mocker.MagicMock(side_effect=Exception()))
    base_task = BaseTaskType(node_name="test")

    task_node = nodes.TaskNode(name="test", internal_name="test", next_node="next_node", executable=base_task)

    task_node.execute()

    assert mock_attempt_log.status == defaults.FAIL


def test_task_node_sets_attempt_log_success_in_no_exception_of_execution(mocker, monkeypatch):
    mock_attempt_log = mocker.MagicMock()
    mock_context = mocker.MagicMock()

    monkeypatch.setattr(nodes.TaskNode, "_context", mock_context)
    mock_context.run_log_store.create_attempt_log = mocker.MagicMock(return_value=mock_attempt_log)

    monkeypatch.setattr(BaseTaskType, "execute_command", mocker.MagicMock())
    base_task = BaseTaskType(node_name="test")
    task_node = nodes.TaskNode(name="test", internal_name="test", next_node="next_node", executable=base_task)

    task_node.execute()

    assert mock_attempt_log.status == defaults.SUCCESS


def test_fail_node_sets_branch_log_fail(mocker, monkeypatch):
    mock_attempt_log = mocker.MagicMock()
    mock_branch_log = mocker.MagicMock()
    mock_context = mocker.MagicMock()

    monkeypatch.setattr(nodes.FailNode, "_context", mock_context)

    mock_context.run_log_store.create_attempt_log = mocker.MagicMock(return_value=mock_attempt_log)
    mock_context.run_log_store.get_branch_log = mocker.MagicMock(return_value=mock_branch_log)

    node = nodes.FailNode(name="test", internal_name="test")

    node.execute()

    assert mock_attempt_log.status == defaults.SUCCESS
    assert mock_branch_log.status == defaults.FAIL


def test_fail_node_sets_attempt_log_success_even_in_exception(mocker, monkeypatch):
    mock_attempt_log = mocker.MagicMock()
    mock_context = mocker.MagicMock()

    monkeypatch.setattr(nodes.FailNode, "_context", mock_context)

    mock_context.run_log_store.create_attempt_log = mocker.MagicMock(return_value=mock_attempt_log)
    mock_context.run_log_store.get_branch_log = mocker.MagicMock(side_effect=Exception())

    node = nodes.FailNode(name="test", internal_name="test")

    node.execute()

    assert mock_attempt_log.status == defaults.SUCCESS


def test_success_node_sets_branch_log_success(mocker, monkeypatch):
    mock_attempt_log = mocker.MagicMock()
    mock_branch_log = mocker.MagicMock()
    mock_context = mocker.MagicMock()

    monkeypatch.setattr(nodes.SuccessNode, "_context", mock_context)

    mock_context.run_log_store.create_attempt_log = mocker.MagicMock(return_value=mock_attempt_log)
    mock_context.run_log_store.get_branch_log = mocker.MagicMock(return_value=mock_branch_log)

    node = nodes.SuccessNode(name="test", internal_name="test")

    node.execute()

    assert mock_attempt_log.status == defaults.SUCCESS
    assert mock_branch_log.status == defaults.SUCCESS


def test_success_node_sets_attempt_log_success_even_in_exception(mocker, monkeypatch):
    mock_attempt_log = mocker.MagicMock()
    mock_branch_log = mocker.MagicMock()
    mock_context = mocker.MagicMock()

    monkeypatch.setattr(nodes.SuccessNode, "_context", mock_context)

    mock_context.run_log_store.create_attempt_log = mocker.MagicMock(return_value=mock_attempt_log)
    mock_context.run_log_store.get_branch_log = mocker.MagicMock(side_effect=Exception())

    node = nodes.SuccessNode(name="test", internal_name="test")

    node.execute()

    assert mock_attempt_log.status == defaults.SUCCESS


# def test_parallel_node_get_sub_graphs_creates_graphs(mocker, monkeypatch):
#     mock_create_graph = mocker.MagicMock(return_value="agraphobject")

#     monkeypatch.setattr(nodes, "create_graph", mock_create_graph)

#     parallel_config = {"branches": {"a": {}, "b": {}}, "next": "next_node"}
#     node = nodes.ParallelNode(name="test", internal_name="test", config=parallel_config)
#     assert mock_create_graph.call_count == 2
#     assert len(node.branches.items()) == 2


def test_nodes_map_node_names_the_branch_as_defaults_place_holder(monkeypatch, mocker):
    monkeypatch.setattr(nodes.MapNode, "get_sub_graph", mocker.MagicMock())

    map_config = {"branch": {}, "next": "next_node", "iterate_on": "test", "iterate_as": "test"}

    node = nodes.MapNode(name="test", internal_name="test", config=map_config)

    assert node.branch_placeholder_name == defaults.MAP_PLACEHOLDER


def test_as_is_node_sets_attempt_log_success(mocker, monkeypatch):
    mock_attempt_log = mocker.MagicMock()

    mock_executor = mocker.MagicMock()
    mock_executor.run_log_store.create_attempt_log = mocker.MagicMock(return_value=mock_attempt_log)

    node = nodes.AsIsNode(name="test", internal_name="test", config={"next": "test"})

    node.execute(executor=mock_executor)

    assert mock_attempt_log.status == defaults.SUCCESS
