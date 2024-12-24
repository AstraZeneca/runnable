import pytest

from extensions.nodes.nodes import FailNode, StubNode, SuccessNode
from runnable import (
    exceptions,  # pylint: disable=import-error
    graph,  # pylint: disable=import-error
)


def get_new_graph(start_at="this", internal_branch_name="i_name"):
    return graph.Graph(start_at=start_at, internal_branch_name=internal_branch_name)


@pytest.fixture
def new_graph():
    return get_new_graph()


class Node:
    def __init__(self, name="a", internal_name="a.b", node_type="task"):
        self.name = name
        self.internal_name = internal_name
        self.node_type = node_type


@pytest.fixture
def dummy_node(name="a", internal_name="a.b", node_type="task"):
    return Node(name, internal_name, node_type)


def test_init():
    new_graph = graph.Graph(start_at="this", internal_branch_name="i_name")
    assert new_graph.start_at == "this"
    assert new_graph.internal_branch_name == "i_name"


def test_init_default():
    new_graph = graph.Graph(start_at="this")
    assert new_graph.start_at == "this"
    assert new_graph.internal_branch_name == ""
    assert len(new_graph.nodes) == 0


def test_get_node_by_name_raises_exception_if_no_match(new_graph):
    with pytest.raises(exceptions.NodeNotFoundError):
        new_graph.get_node_by_name("a")


def test_get_node_by_name_returns_node_if_match(new_graph, dummy_node):
    new_graph.nodes["a"] = dummy_node
    assert dummy_node == new_graph.get_node_by_name("a")


def test_get_node_by_internal_name_raises_exception_if_no_match(new_graph):
    with pytest.raises(exceptions.NodeNotFoundError):
        new_graph.get_node_by_internal_name("a")


def test_get_node_by_internal_name_returns_node_if_match(new_graph, dummy_node):
    new_graph.nodes["a.b"] = dummy_node
    assert dummy_node == new_graph.get_node_by_internal_name("a.b")


def test_add_node_adds_to_nodes(new_graph, dummy_node):
    new_graph.add_node(dummy_node)
    assert len(new_graph.nodes) == 1
    assert new_graph.nodes["a"] == dummy_node


def test_get_success_node_fails_if_none_present(new_graph):
    with pytest.raises(Exception):
        new_graph.get_success_node()


def test_get_success_node_returns_success_node_if_present(new_graph):
    new_node = Node(node_type="success", name="a")
    new_graph.nodes["a"] = new_node

    assert new_graph.get_success_node() == new_node


def test_get_fail_node_fails_if_none_present(new_graph):
    with pytest.raises(Exception):
        new_graph.get_fail_node()


def test_get_fail_node_returns_success_node_if_present(new_graph):
    new_node = Node(node_type="fail", name="a")
    new_graph.nodes["a"] = new_node

    assert new_graph.get_fail_node() == new_node


def test_is_start_node_present_returns_false_if_node_absent(monkeypatch, mocker):
    monkeypatch.setattr(
        graph.Graph,
        "get_node_by_name",
        mocker.MagicMock(side_effect=exceptions.NodeNotFoundError("a")),
    )
    new_graph = get_new_graph(start_at="a")
    assert new_graph.is_start_node_present() is False


def test_is_start_node_present_returns_true_if_node_present(mocker, monkeypatch):
    monkeypatch.setattr(graph.Graph, "get_node_by_name", mocker.MagicMock())
    new_graph = get_new_graph(start_at="a")
    assert new_graph.is_start_node_present() is True


def test_success_node_validation_returns_false_if_neq_1(new_graph):
    assert new_graph.success_node_validation() is False


def test_success_node_validation_returns_false_if_gr_1(new_graph):
    node = Node(node_type="success", name="a")
    new_graph.nodes["a"] = node
    new_graph.nodes["b"] = node
    assert new_graph.success_node_validation() is False


def test_success_node_validation_returns_true_if_eq_1(new_graph):
    node = Node(node_type="success", name="a")
    new_graph.nodes["a"] = node
    assert new_graph.success_node_validation() is True


def test_fail_node_validation_returns_false_if_neq_1(new_graph):
    assert new_graph.fail_node_validation() is False


def test_fail_node_validation_returns_false_if_gr_1(new_graph):
    node = Node(node_type="fail", name="a")
    new_graph.nodes["a"] = node
    new_graph.nodes["b"] = node
    assert new_graph.fail_node_validation() is False


def test_fail_node_validation_returns_true_if_eq_1(new_graph):
    node = Node(node_type="fail", name="a")
    new_graph.nodes["a"] = node
    assert new_graph.fail_node_validation() is True


def test_check_graph_does_not_raise_exception_if_all_pass(monkeypatch, mocker):
    try:
        monkeypatch.setattr(
            graph.Graph, "missing_neighbors", mocker.MagicMock(return_value=[])
        )
        monkeypatch.setattr(graph.Graph, "is_dag", mocker.MagicMock(return_value=True))
        monkeypatch.setattr(
            graph.Graph, "is_start_node_present", mocker.MagicMock(return_value=True)
        )
        monkeypatch.setattr(
            graph.Graph, "success_node_validation", mocker.MagicMock(return_value=True)
        )
        monkeypatch.setattr(
            graph.Graph, "fail_node_validation", mocker.MagicMock(return_value=True)
        )
        new_graph = get_new_graph()
        new_graph.check_graph()
    except BaseException:
        assert False


def test_validate_raises_exception_if_is_dag_fails(mocker, monkeypatch):
    with pytest.raises(Exception):
        monkeypatch.setattr(
            graph.Graph, "missing_neighbors", mocker.MagicMock(return_value=[])
        )
        monkeypatch.setattr(graph.Graph, "is_dag", mocker.MagicMock(return_value=False))
        monkeypatch.setattr(
            graph.Graph, "is_start_node_present", mocker.MagicMock(return_value=True)
        )
        monkeypatch.setattr(
            graph.Graph, "success_node_validation", mocker.MagicMock(return_value=True)
        )
        monkeypatch.setattr(
            graph.Graph, "fail_node_validation", mocker.MagicMock(return_value=True)
        )
        new_graph = get_new_graph()
        new_graph.check_graph()


def test_validate_raises_exception_if_is_start_node_present_fails(mocker, monkeypatch):
    with pytest.raises(Exception):
        monkeypatch.setattr(
            graph.Graph, "missing_neighbors", mocker.MagicMock(return_value=[])
        )
        monkeypatch.setattr(graph.Graph, "is_dag", mocker.MagicMock(return_value=True))
        monkeypatch.setattr(
            graph.Graph, "is_start_node_present", mocker.MagicMock(return_value=False)
        )
        monkeypatch.setattr(
            graph.Graph, "success_node_validation", mocker.MagicMock(return_value=True)
        )
        monkeypatch.setattr(
            graph.Graph, "fail_node_validation", mocker.MagicMock(return_value=True)
        )
        new_graph = get_new_graph()
        new_graph.check_graph()


def test_validate_raises_exception_if_success_node_validation_fails(
    mocker, monkeypatch
):
    with pytest.raises(Exception):
        monkeypatch.setattr(
            graph.Graph, "missing_neighbors", mocker.MagicMock(return_value=[])
        )
        monkeypatch.setattr(graph.Graph, "is_dag", mocker.MagicMock(return_value=True))
        monkeypatch.setattr(
            graph.Graph, "is_start_node_present", mocker.MagicMock(return_value=True)
        )
        monkeypatch.setattr(
            graph.Graph, "success_node_validation", mocker.MagicMock(return_value=False)
        )
        monkeypatch.setattr(
            graph.Graph, "fail_node_validation", mocker.MagicMock(return_value=True)
        )
        new_graph = get_new_graph()
        new_graph.check_graph()


def test_validate_raises_exception_if_fail_node_validation_fails(mocker, monkeypatch):
    with pytest.raises(Exception):
        monkeypatch.setattr(
            graph.Graph, "missing_neighbors", mocker.MagicMock(return_value=[])
        )
        monkeypatch.setattr(graph.Graph, "is_dag", mocker.MagicMock(return_value=True))
        monkeypatch.setattr(
            graph.Graph, "is_start_node_present", mocker.MagicMock(return_value=True)
        )
        monkeypatch.setattr(
            graph.Graph, "success_node_validation", mocker.MagicMock(return_value=True)
        )
        monkeypatch.setattr(
            graph.Graph, "fail_node_validation", mocker.MagicMock(return_value=False)
        )
        new_graph = get_new_graph()
        new_graph.check_graph()


def test_validate_raises_exception_if_missing_neighbors(mocker, monkeypatch):
    with pytest.raises(Exception):
        monkeypatch.setattr(
            graph.Graph, "missing_neighbors", mocker.MagicMock(return_value=["missing"])
        )
        monkeypatch.setattr(graph.Graph, "is_dag", mocker.MagicMock(return_value=True))
        monkeypatch.setattr(
            graph.Graph, "is_start_node_present", mocker.MagicMock(return_value=True)
        )
        monkeypatch.setattr(
            graph.Graph, "success_node_validation", mocker.MagicMock(return_value=True)
        )
        monkeypatch.setattr(
            graph.Graph, "fail_node_validation", mocker.MagicMock(return_value=True)
        )
        new_graph = get_new_graph()
        new_graph.check_graph()


@pytest.fixture(name="mocked_graph")
def create_mocked_graph(mocker):
    return graph.Graph(start_at="start")


def test_is_dag_returns_true_when_acyclic(mocked_graph):
    start_node = StubNode(name="start", internal_name="start", next_node="middle")

    middle_node = StubNode(name="middle", internal_name="middle", next_node="success")

    success_node = SuccessNode(name="success", internal_name="success")

    fail_node = FailNode(name="fail", internal_name="fail")

    test_graph = mocked_graph
    for node in [start_node, middle_node, success_node, fail_node]:
        test_graph.add_node(node)

    assert test_graph.is_dag()


def test_is_dag_returns_true_when_on_failure_points_to_non_terminal_node_and_later_node(
    mocked_graph,
):
    test_graph = mocked_graph

    start_node_config = {"next_node": "middle", "on_failure": ""}
    start_node = StubNode(
        name="start", internal_name="start", next_node="middle", on_failure=""
    )

    middle_node_config = {"next_node": "success", "on_failure": "fail"}
    middle_node = StubNode(
        name="middle", internal_name="middle", next_node="success", on_failure="fail"
    )

    success_node = SuccessNode(name="success", internal_name="success")

    fail_node = FailNode(name="fail", internal_name="fail")

    test_graph = mocked_graph
    for node in [start_node, middle_node, success_node, fail_node]:
        test_graph.add_node(node)

    assert test_graph.is_dag()


def test_is_dag_returns_false_when_cyclic_in_next_nodes(mocked_graph):
    test_graph = mocked_graph

    start_node_config = {"next_node": "b", "on_failure": "fail"}
    start_node = StubNode(
        name="start", internal_name="start", next_node="b", on_failure="fail"
    )

    bnode = StubNode(name="b", internal_name="b", next_node="c", on_failure="fail")

    cnode = StubNode(name="c", internal_name="c", next_node="d", on_failure="fail")

    dnode = StubNode(name="d", internal_name="d", next_node="b", on_failure="fail")

    fail_node = FailNode(name="fail", internal_name="fail")

    nodes = [start_node, bnode, cnode, dnode, fail_node]
    test_graph = mocked_graph
    for node in nodes:
        test_graph.add_node(node)

    assert not test_graph.is_dag()


def test_is_dag_returns_false_when_fail_points_to_previous_node(mocked_graph):
    test_graph = mocked_graph

    start_node = StubNode(
        name="start", internal_name="start", next_node="b", on_failure="fail"
    )
    bnode = StubNode(name="b", internal_name="b", next_node="c", on_failure="fail")

    cnode = StubNode(name="c", internal_name="c", next_node="c", on_failure="fail")

    fail_node = FailNode(name="fail", internal_name="fail")
    nodes = [start_node, bnode, cnode, fail_node]
    test_graph = mocked_graph
    for node in nodes:
        test_graph.add_node(node)

    assert not test_graph.is_dag()


def test_missing_neighbors_empty_list_no_neigbors_missing(mocked_graph):
    test_graph = mocked_graph

    start_node = StubNode(
        name="start", internal_name="start", next_node="middle", on_failure="fail"
    )

    middle_node = StubNode(
        name="middle", internal_name="middle", next_node="success", on_failure="fail"
    )

    success_node = SuccessNode(name="success", internal_name="success")

    fail_node = FailNode(name="fail", internal_name="fail")

    nodes = [start_node, middle_node, success_node, fail_node]
    test_graph = mocked_graph
    for node in nodes:
        test_graph.add_node(node)

    missing_nodes = test_graph.missing_neighbors()
    assert len(missing_nodes) == 0


def test_missing_neighbors_list_of_missing_neighbor_one_missing_next(mocked_graph):
    test_graph = mocked_graph

    start_config = {"next_node": "middle", "on_failure": "fail"}
    start_node = StubNode(
        name="start", internal_name="start", next_node="middle", on_failure="fail"
    )

    middle_config = {"next_node": "success", "on_failure": "fail"}
    middle_node = StubNode(
        name="middle", internal_name="middle", next_node="success", on_failure="fail"
    )

    fail_node = FailNode(name="fail", internal_name="fail")

    nodes = [start_node, middle_node, fail_node]
    test_graph = mocked_graph
    for node in nodes:
        test_graph.add_node(node)

    missing_nodes = test_graph.missing_neighbors()
    assert len(missing_nodes) == 1
    assert missing_nodes[0] == "success"


def test_missing_list_of_missing_neighbor_one_missing_on_failure(mocked_graph):
    test_graph = mocked_graph

    start_node = StubNode(
        name="start", internal_name="start", next_node="middle", on_failure="fail"
    )

    middle_node = StubNode(
        name="middle", internal_name="middle", next_node="success", on_failure="fail"
    )

    success_node = SuccessNode(name="success", internal_name="success")

    FailNode(name="fail", internal_name="fail")

    nodes = [start_node, middle_node, success_node]
    test_graph = mocked_graph
    for node in nodes:
        test_graph.add_node(node)

    missing_nodes = test_graph.missing_neighbors()
    assert len(missing_nodes) == 1
    assert missing_nodes[0] == "fail"


def test_missing_list_of_missing_neighbor_two_missing(mocked_graph):
    test_graph = mocked_graph
    start_node = StubNode(
        name="start", internal_name="start", next_node="middle", on_failure="fail"
    )

    StubNode(
        name="middle", internal_name="middle", next_node="success", on_failure="fail"
    )

    success_node = SuccessNode(name="success", internal_name="success")

    FailNode(name="fail", internal_name="fail")

    nodes = [
        start_node,
        success_node,
    ]

    test_graph = mocked_graph
    for node in nodes:
        test_graph.add_node(node)
    missing_nodes = test_graph.missing_neighbors()
    assert len(missing_nodes) == 2
    assert "middle" in missing_nodes
    assert "fail" in missing_nodes
