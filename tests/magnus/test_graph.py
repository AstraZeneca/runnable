import pytest

from magnus import defaults  # pylint: disable=import-error
from magnus import exceptions  # pylint: disable=import-error
from magnus import graph  # pylint: disable=import-error
from magnus.nodes import BaseNode


def get_new_graph(start_at='this', internal_branch_name='i_name'):
    return graph.Graph(start_at=start_at, internal_branch_name=internal_branch_name)


@pytest.fixture
def new_graph():
    return get_new_graph()


class Node:
    def __init__(self, name='a', internal_name='a.b', node_type='task'):
        self.name = name
        self.internal_name = internal_name
        self.node_type = node_type


@pytest.fixture
def dummy_node(name='a', internal_name='a.b', node_type='task'):
    return Node(name, internal_name, node_type)


def test_init():
    new_graph = graph.Graph(start_at='this', internal_branch_name='i_name')
    assert new_graph.start_at == 'this'
    assert new_graph.internal_branch_name == 'i_name'


def test_init_default():
    new_graph = graph.Graph(start_at='this')
    assert new_graph.start_at == 'this'
    assert new_graph.internal_branch_name is None
    assert len(new_graph.nodes) == 0


def test_get_node_by_name_raises_exception_if_no_match(new_graph):
    with pytest.raises(exceptions.NodeNotFoundError):
        new_graph.get_node_by_name('a')


def test_get_node_by_name_returns_node_if_match(new_graph, dummy_node):
    new_graph.nodes.append(dummy_node)
    assert dummy_node == new_graph.get_node_by_name('a')


def test_get_node_by_internal_name_raises_exception_if_no_match(new_graph):
    with pytest.raises(exceptions.NodeNotFoundError):
        new_graph.get_node_by_internal_name('a')


def test_get_node_by_internal_name_returns_node_if_match(new_graph, dummy_node):
    new_graph.nodes.append(dummy_node)
    assert dummy_node == new_graph.get_node_by_internal_name('a.b')


def test_add_node_adds_to_nodes(new_graph, dummy_node):
    new_graph.add_node(dummy_node)
    assert len(new_graph.nodes) == 1
    assert new_graph.nodes[0] == dummy_node


def test_get_success_node_fails_if_none_present(new_graph):
    with pytest.raises(Exception):
        new_graph.get_success_node()


def test_get_success_node_returns_success_node_if_present(new_graph):
    new_node = Node(node_type='success')
    new_graph.nodes.append(new_node)

    assert new_graph.get_success_node() == new_node


def test_get_fail_node_fails_if_none_present(new_graph):
    with pytest.raises(Exception):
        new_graph.get_fail_node()


def test_get_fail_node_returns_success_node_if_present(new_graph):
    new_node = Node(node_type='fail')
    new_graph.nodes.append(new_node)

    assert new_graph.get_fail_node() == new_node


def test_is_start_node_present_returns_false_if_node_absent(monkeypatch, mocker):
    monkeypatch.setattr(graph.Graph, 'get_node_by_name', mocker.MagicMock(
        side_effect=exceptions.NodeNotFoundError('a')))
    new_graph = get_new_graph(start_at='a')
    assert new_graph.is_start_node_present() == False


def test_is_start_node_present_returns_true_if_node_present(mocker, monkeypatch):
    monkeypatch.setattr(graph.Graph, 'get_node_by_name', mocker.MagicMock())
    new_graph = get_new_graph(start_at='a')
    assert new_graph.is_start_node_present() == True


def test_success_node_validation_returns_false_if_neq_1(new_graph):
    assert new_graph.success_node_validation() == False


def test_success_node_validation_returns_false_if_gr_1(new_graph):
    node = Node(node_type='success')
    new_graph.nodes.append(node)
    new_graph.nodes.append(node)
    assert new_graph.success_node_validation() == False


def test_success_node_validation_returns_true_if_eq_1(new_graph):
    node = Node(node_type='success')
    new_graph.nodes.append(node)
    assert new_graph.success_node_validation() == True


def test_fail_node_validation_returns_false_if_neq_1(new_graph):
    assert new_graph.fail_node_validation() == False


def test_fail_node_validation_returns_false_if_gr_1(new_graph):
    node = Node(node_type='fail')
    new_graph.nodes.append(node)
    new_graph.nodes.append(node)
    assert new_graph.fail_node_validation() == False


def test_fail_node_validation_returns_true_if_eq_1(new_graph):
    node = Node(node_type='fail')
    new_graph.nodes.append(node)
    assert new_graph.fail_node_validation() == True


def test_validate_does_not_raise_exception_if_all_pass(monkeypatch, mocker):
    try:
        monkeypatch.setattr(graph.Graph, 'missing_neighbors', mocker.MagicMock(return_value=[]))
        monkeypatch.setattr(graph.Graph, 'is_dag', mocker.MagicMock(return_value=True))
        monkeypatch.setattr(graph.Graph, 'is_start_node_present', mocker.MagicMock(return_value=True))
        monkeypatch.setattr(graph.Graph, 'success_node_validation', mocker.MagicMock(return_value=True))
        monkeypatch.setattr(graph.Graph, 'fail_node_validation', mocker.MagicMock(return_value=True))
        new_graph = get_new_graph()
        new_graph.validate()
    except BaseException:
        assert False


def test_validate_raises_exception_if_is_dag_fails(mocker, monkeypatch):
    with pytest.raises(Exception):
        monkeypatch.setattr(graph.Graph, 'missing_neighbors', mocker.MagicMock(return_value=[]))
        monkeypatch.setattr(graph.Graph, 'is_dag', mocker.MagicMock(return_value=False))
        monkeypatch.setattr(graph.Graph, 'is_start_node_present', mocker.MagicMock(return_value=True))
        monkeypatch.setattr(graph.Graph, 'success_node_validation', mocker.MagicMock(return_value=True))
        monkeypatch.setattr(graph.Graph, 'fail_node_validation', mocker.MagicMock(return_value=True))
        new_graph = get_new_graph()
        new_graph.validate()


def test_validate_raises_exception_if_is_start_node_present_fails(mocker, monkeypatch):
    with pytest.raises(Exception):
        monkeypatch.setattr(graph.Graph, 'missing_neighbors', mocker.MagicMock(return_value=[]))
        monkeypatch.setattr(graph.Graph, 'is_dag', mocker.MagicMock(return_value=True))
        monkeypatch.setattr(graph.Graph, 'is_start_node_present', mocker.MagicMock(return_value=False))
        monkeypatch.setattr(graph.Graph, 'success_node_validation', mocker.MagicMock(return_value=True))
        monkeypatch.setattr(graph.Graph, 'fail_node_validation', mocker.MagicMock(return_value=True))
        new_graph = get_new_graph()
        new_graph.validate()


def test_validate_raises_exception_if_success_node_validation_fails(mocker, monkeypatch):
    with pytest.raises(Exception):
        monkeypatch.setattr(graph.Graph, 'missing_neighbors', mocker.MagicMock(return_value=[]))
        monkeypatch.setattr(graph.Graph, 'is_dag', mocker.MagicMock(return_value=True))
        monkeypatch.setattr(graph.Graph, 'is_start_node_present', mocker.MagicMock(return_value=True))
        monkeypatch.setattr(graph.Graph, 'success_node_validation', mocker.MagicMock(return_value=False))
        monkeypatch.setattr(graph.Graph, 'fail_node_validation', mocker.MagicMock(return_value=True))
        new_graph = get_new_graph()
        new_graph.validate()


def test_validate_raises_exception_if_fail_node_validation_fails(mocker, monkeypatch):
    with pytest.raises(Exception):
        monkeypatch.setattr(graph.Graph, 'missing_neighbors', mocker.MagicMock(return_value=[]))
        monkeypatch.setattr(graph.Graph, 'is_dag', mocker.MagicMock(return_value=True))
        monkeypatch.setattr(graph.Graph, 'is_start_node_present', mocker.MagicMock(return_value=True))
        monkeypatch.setattr(graph.Graph, 'success_node_validation', mocker.MagicMock(return_value=True))
        monkeypatch.setattr(graph.Graph, 'fail_node_validation', mocker.MagicMock(return_value=False))
        new_graph = get_new_graph()
        new_graph.validate()


def test_validate_raises_exception_if_missing_neighbors(mocker, monkeypatch):
    with pytest.raises(Exception):
        monkeypatch.setattr(graph.Graph, 'missing_neighbors', mocker.MagicMock(return_value=['missing']))
        monkeypatch.setattr(graph.Graph, 'is_dag', mocker.MagicMock(return_value=True))
        monkeypatch.setattr(graph.Graph, 'is_start_node_present', mocker.MagicMock(return_value=True))
        monkeypatch.setattr(graph.Graph, 'success_node_validation', mocker.MagicMock(return_value=True))
        monkeypatch.setattr(graph.Graph, 'fail_node_validation', mocker.MagicMock(return_value=True))
        new_graph = get_new_graph()
        new_graph.validate()


def test_create_graph_inits_graph_with_defaults(mocker, monkeypatch):
    dag_config = {
        'start_at': 'step1'
    }
    graph_init = mocker.MagicMock(return_value=None)
    monkeypatch.setattr(graph.Graph, '__init__', graph_init)
    monkeypatch.setattr(graph.Graph, 'validate', mocker.MagicMock())

    graph.create_graph(dag_config, internal_branch_name='i_name')
    graph_init.assert_called_once_with(start_at='step1', description=None,
                                       max_time=defaults.MAX_TIME, internal_branch_name='i_name')


def test_create_graph_inits_graph_with_given_config(mocker, monkeypatch):
    dag_config = {
        'start_at': 'step1',
        'description': 'test',
        'max_time': 1
    }
    graph_init = mocker.MagicMock(return_value=None)
    monkeypatch.setattr(graph.Graph, '__init__', graph_init)
    monkeypatch.setattr(graph.Graph, 'validate', mocker.MagicMock())

    graph.create_graph(dag_config, internal_branch_name='i_name')
    graph_init.assert_called_once_with(start_at='step1', description='test',
                                       max_time=1, internal_branch_name='i_name')


def test_create_graph_inits_graph_populates_nodes(mocker, monkeypatch):
    dag_config = {
        'start_at': 'step1',
        'steps': {
            'step1': {
                'type': 'test'
            }
        }
    }
    graph_init = mocker.MagicMock(return_value=None)
    monkeypatch.setattr(graph.Graph, '__init__', graph_init)
    monkeypatch.setattr(graph.Graph, 'validate', mocker.MagicMock())
    monkeypatch.setattr(graph.Graph, 'add_node', mocker.MagicMock())

    monkeypatch.setattr('magnus.nodes.validate_node', mocker.MagicMock())

    mock_driver_manager = mocker.MagicMock()

    monkeypatch.setattr(graph.driver, 'DriverManager', mock_driver_manager)
    graph.create_graph(dag_config, internal_branch_name=None)

    _, kwargs = mock_driver_manager.call_args
    assert kwargs['invoke_kwds']['name'] == 'step1'
    assert kwargs['invoke_kwds']['internal_name'] == 'step1'


def test_create_graph_inits_graph_populates_nodes_with_internal_branch(mocker, monkeypatch):
    dag_config = {
        'start_at': 'step1',
        'steps': {
            'step1': {
                'type': 'test'
            }
        }
    }
    graph_init = mocker.MagicMock(return_value=None)
    monkeypatch.setattr(graph.Graph, '__init__', graph_init)
    monkeypatch.setattr(graph.Graph, 'validate', mocker.MagicMock())
    monkeypatch.setattr(graph.Graph, 'add_node', mocker.MagicMock())

    monkeypatch.setattr('magnus.nodes.validate_node', mocker.MagicMock())

    mock_driver_manager = mocker.MagicMock()

    monkeypatch.setattr(graph.driver, 'DriverManager', mock_driver_manager)
    graph.create_graph(dag_config, internal_branch_name='i_name')

    _, kwargs = mock_driver_manager.call_args
    assert kwargs['invoke_kwds']['name'] == 'step1'
    assert kwargs['invoke_kwds']['internal_name'] == 'i_name.step1'


def test_create_graph_raises_exception_if_node_fails(mocker, monkeypatch):
    dag_config = {
        'start_at': 'step1',
        'steps': {
            'step1': {
                'type': 'test'
            }
        }
    }
    graph_init = mocker.MagicMock(return_value=None)
    monkeypatch.setattr(graph.Graph, '__init__', graph_init)
    monkeypatch.setattr(graph.Graph, 'validate', mocker.MagicMock())
    monkeypatch.setattr(graph.Graph, 'add_node', mocker.MagicMock())

    monkeypatch.setattr('magnus.nodes.validate_node', mocker.MagicMock(return_value=['wrong']))

    with pytest.raises(Exception):
        graph.create_graph(dag_config, internal_branch_name=None)


@pytest.fixture(name='mocked_graph')
def create_mocked_graph(mocker):
    mocked_graph_init = mocker.MagicMock(return_value=None)
    mocker.patch.object(graph.Graph, '__init__', mocked_graph_init)
    return graph.Graph()


@pytest.fixture(name='mocked_basenode')
def create_mocked_basenode(mocker):
    mocked_basenode_init = mocker.MagicMock(return_value=None)
    mocker.patch.object(BaseNode, "__init__", mocked_basenode_init)
    return BaseNode


def test_is_dag_returns_true_when_acyclic(mocked_graph, mocked_basenode):
    test_graph = mocked_graph
    start_node = mocked_basenode()
    start_node.name = 'start'
    start_node.config = {'next': 'middle', 'on_failure': 'fail_node'}
    middle_node = mocked_basenode()
    middle_node.name = 'middle'
    middle_node.config = {'next': 'success', 'on_failure': 'fail_node'}
    success_node = mocked_basenode()
    success_node.name = 'success'
    success_node.config = {}
    fail_node = mocked_basenode()
    fail_node.name = 'fail_node'
    fail_node.config = {}

    test_graph.nodes = [
        start_node,
        middle_node,
        success_node,
        fail_node
    ]

    assert test_graph.is_dag()


def test_is_dag_returns_true_when_on_failure_points_to_non_terminal_node_and_later_node(mocked_graph, mocked_basenode):
    test_graph = mocked_graph

    start_node = mocked_basenode()
    start_node.name = 'start'
    start_node.config = {'next': 'middle', 'on_failure': 'middle'}

    middle_node = mocked_basenode()
    middle_node.name = 'middle'
    middle_node.config = {'next': 'success', 'on_failure': 'fail_node'}

    success_node = mocked_basenode()
    success_node.name = 'success'
    success_node.config = {}

    fail_node = mocked_basenode()
    fail_node.name = 'fail_node'
    fail_node.config = {}

    test_graph.nodes = [
        start_node,
        middle_node,
        success_node,
        fail_node
    ]
    assert test_graph.is_dag()


def test_is_dag_returns_false_when_cyclic_in_next_nodes(mocked_graph, mocked_basenode):
    test_graph = mocked_graph

    start_node = mocked_basenode()
    start_node.name = 'start'
    start_node.config = {'next': 'b', 'on_failure': 'fail_node'}

    bnode = mocked_basenode()
    bnode.name = 'b'
    bnode.config = {'next': 'c', 'on_failure': 'fail_node'}

    cnode = mocked_basenode()
    cnode.name = 'c'
    cnode.config = {'next': 'd', 'on_failure': 'fail_node'}

    dnode = mocked_basenode()
    dnode.name = 'd'
    dnode.config = {'next': 'b', 'on_failure': 'fail_node'}

    fail_node = mocked_basenode()
    fail_node.name = 'fail_node'
    fail_node.config = {}

    test_graph.nodes = [
        start_node,
        bnode,
        cnode,
        dnode,
        fail_node
    ]

    assert not test_graph.is_dag()


def test_is_dag_returns_false_when_fail_points_to_previous_node(mocked_graph, mocked_basenode):
    test_graph = mocked_graph

    start_node = mocked_basenode()
    start_node.name = 'start'
    start_node.config = {'next': 'b', 'on_failure': 'fail_node'}

    bnode = mocked_basenode()
    bnode.name = 'b'
    bnode.config = {'next': 'c', 'on_failure': 'fail_node'}

    cnode = mocked_basenode()
    cnode.name = 'c'
    cnode.config = {'next': 'finish', 'on_failure': 'b'}

    finish_node = mocked_basenode()
    finish_node.name = 'finish'
    finish_node.config = {}

    fail_node = mocked_basenode()
    fail_node.name = 'fail_node'
    fail_node.config = {}

    test_graph.nodes = [
        start_node,
        bnode,
        cnode,
        finish_node,
        fail_node
    ]

    assert not test_graph.is_dag()


def test_is_dag_returns_false_when_downstream_fail_node_revisits_earlier_node(mocked_graph, mocked_basenode):
    test_graph = mocked_graph

    start_node = mocked_basenode()
    start_node.name = 'start'
    start_node.config = {'next': 'b', 'on_fail': 'fail_node'}

    bnode = mocked_basenode()
    bnode.name = 'b'
    bnode.config = {'next': 'c', 'on_failure': 'fail_node'}

    cnode = mocked_basenode()
    cnode.name = 'c'
    cnode.config = {'next': 'finish', 'on_failure': 'd'}

    dnode = mocked_basenode()
    dnode.name = 'd'
    dnode.config = {'next': 'b', 'on_failure': 'fail_node'}

    finish_node = mocked_basenode()
    finish_node.name = 'finish'
    finish_node.config = {}

    fail_node = mocked_basenode()
    fail_node.name = 'fail_mode'
    fail_node.config = {}

    test_graph.nodes = [
        start_node,
        bnode,
        cnode,
        dnode,
        finish_node,
        fail_node
    ]

    assert not test_graph.is_dag()


def test_missing_neighbors_empty_list_no_neigbors_missing(mocked_graph, mocked_basenode):
    test_graph = mocked_graph
    start_node = mocked_basenode()
    start_node.name = 'start'
    start_node.config = {'next': 'middle', 'on_failure': 'fail_node'}
    middle_node = mocked_basenode()
    middle_node.name = 'middle'
    middle_node.config = {'next': 'success', 'on_failure': 'fail_node'}
    success_node = mocked_basenode()
    success_node.name = 'success'
    success_node.config = {}
    fail_node = mocked_basenode()
    fail_node.name = 'fail_node'
    fail_node.config = {}

    test_graph.nodes = [
        start_node,
        middle_node,
        success_node,
        fail_node
    ]

    missing_nodes = test_graph.missing_neighbors()
    assert len(missing_nodes) == 0


def test_missing_neighbors_list_of_missing_neighbor_one_missing_next(mocked_graph, mocked_basenode):
    test_graph = mocked_graph
    start_node = mocked_basenode()
    start_node.name = 'start'
    start_node.config = {'next': 'middle', 'on_failure': 'fail_node'}
    middle_node = mocked_basenode()
    middle_node.name = 'middle'
    middle_node.config = {'next': 'success', 'on_failure': 'fail_node'}
    fail_node = mocked_basenode()
    fail_node.name = 'fail_node'
    fail_node.config = {}

    test_graph.nodes = [
        start_node,
        middle_node,
        fail_node
    ]

    missing_nodes = test_graph.missing_neighbors()
    assert len(missing_nodes) == 1
    assert missing_nodes[0] == 'success'


def test_missing_list_of_missing_neighbor_one_missing_on_failure(mocked_graph, mocked_basenode):
    test_graph = mocked_graph
    start_node = mocked_basenode()
    start_node.name = 'start'
    start_node.config = {'next': 'middle', 'on_failure': 'fail_node'}
    middle_node = mocked_basenode()
    middle_node.name = 'middle'
    middle_node.config = {'next': 'success', 'on_failure': 'fail_node'}
    success_node = mocked_basenode()
    success_node.name = 'success'
    success_node.config = {}

    test_graph.nodes = [
        start_node,
        middle_node,
        success_node
    ]

    missing_nodes = test_graph.missing_neighbors()
    assert len(missing_nodes) == 1
    assert missing_nodes[0] == 'fail_node'


def test_missing_list_of_missing_neighbor_two_missing(mocked_graph, mocked_basenode):
    test_graph = mocked_graph
    start_node = mocked_basenode()
    start_node.name = 'start'
    start_node.config = {'next': 'middle', 'on_failure': 'fail_node'}
    success_node = mocked_basenode()
    success_node.name = 'success'
    success_node.config = {}

    test_graph.nodes = [
        start_node,
        success_node,
    ]

    missing_nodes = test_graph.missing_neighbors()
    assert len(missing_nodes) == 2
    assert 'middle' in missing_nodes
    assert 'fail_node' in missing_nodes
