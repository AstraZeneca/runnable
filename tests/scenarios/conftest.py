import pytest

from magnus import graph


@pytest.fixture
def as_is_node():
    def _closure(name, next_node):
        step_config = {
            'command': "does not matter",
            'command_type': 'python-function',
            'type': 'as-is',
            'next': next_node
        }
        return graph.create_node(name=name, step_config=step_config)
    return _closure


@pytest.fixture
def exception_node():
    def _closure(name, next_node):
        step_config = {
            'command': "exit 1",
            'command_type': 'shell',
            'type': 'task',
            'next': next_node
        }
        return graph.create_node(name=name, step_config=step_config)
    return _closure


@pytest.fixture
def success_graph(as_is_node):
    def _closure():
        dag = graph.Graph(start_at='first')
        dag.add_node(as_is_node('first', 'second'))
        dag.add_node(as_is_node('second', 'success'))
        dag.add_terminal_nodes()
        return dag
    return _closure


@pytest.fixture
def fail_graph(exception_node):
    def _closure():
        dag = graph.Graph(start_at='first')
        dag.add_node(exception_node('first', 'success'))
        dag.add_terminal_nodes()
        return dag
    return _closure


@pytest.fixture
def on_fail_graph(as_is_node, exception_node):
    def _closure():
        dag = graph.Graph(start_at='first')
        first_node = exception_node('first', 'second')
        first_node.config['on_failure'] = 'second'
        dag.add_node(first_node)
        dag.add_node(as_is_node('second', 'success'))
        dag.add_terminal_nodes()
        return dag
    return _closure
