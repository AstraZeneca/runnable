import pytest

from magnus import graph


@pytest.fixture
def as_is_node():
    def _closure(name, next_node, on_failure=''):
        step_config = {
            'command': "does not matter",
            'command_type': 'python-function',
            'type': 'as-is',
            'next': next_node,
            'on_failure': on_failure
        }
        return graph.create_node(name=name, step_config=step_config)
    return _closure


@pytest.fixture
def as_is_container_node():
    def _closure(name, next_node, on_failure=''):
        step_config = {
            'command': "does not matter",
            'command_type': 'python-function',
            'type': 'as-is',
            'next': next_node,
            'on_failure': on_failure,
            'executor_config': {
                'local-container':
                {
                    'run_in_local': True
                }
            }
        }
        return graph.create_node(name=name, step_config=step_config)
    return _closure


@pytest.fixture
def exception_node():
    def _closure(name, next_node, on_failure=''):
        step_config = {
            'command': "exit 1",
            'command_type': 'shell',
            'type': 'task',
            'next': next_node,
            'on_failure': on_failure
        }
        return graph.create_node(name=name, step_config=step_config)
    return _closure


@pytest.fixture
def parallel_node():
    def _closure(name, branch, next_node):
        step_config = {
            'type': 'parallel',
            'next': next_node,
            'branches': {
                'a': branch()._to_dict(),
                'b': branch()._to_dict()
            }
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
def success_container_graph(as_is_container_node):
    def _closure():
        dag = graph.Graph(start_at='first')
        dag.add_node(as_is_container_node('first', 'second'))
        dag.add_node(as_is_container_node('second', 'success'))
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
        first_node = exception_node('first', 'second', 'third')
        dag.add_node(first_node)
        dag.add_node(as_is_node('second', 'third'))
        dag.add_node(as_is_node('third', 'success'))
        dag.add_terminal_nodes()
        return dag
    return _closure


@pytest.fixture
def parallel_success_graph(as_is_node, parallel_node, success_graph):
    def _closure():
        dag = graph.Graph(start_at='first')
        dag.add_node(as_is_node('first', 'second'))
        dag.add_node(parallel_node(name='second', branch=success_graph, next_node='success'))
        dag.add_terminal_nodes()
        return dag
    return _closure


@pytest.fixture
def parallel_fail_graph(as_is_node, parallel_node, fail_graph):
    def _closure():
        dag = graph.Graph(start_at='first')
        dag.add_node(as_is_node('first', 'second'))
        dag.add_node(parallel_node(name='second', branch=fail_graph, next_node='success'))
        dag.add_terminal_nodes()
        return dag
    return _closure
