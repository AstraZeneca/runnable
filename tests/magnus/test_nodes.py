import os

import pytest

from magnus import defaults  # pylint: disable=import-error
from magnus import nodes  # pylint: disable=import-error
from magnus import graph


def test_base_execution_execute_command_raises_not_implemented_error():
    base_execution_type = nodes.BaseExecutionType()

    with pytest.raises(NotImplementedError):
        base_execution_type.execute_command(command='fail')


def test_get_command_class_returns_the_correct_subclasses():
    class DummyExecutionType(nodes.BaseExecutionType):
        execution_type = 'dummy'

    obj = nodes.get_command_class(command_type='dummy')
    assert isinstance(obj, DummyExecutionType)


def test_get_command_class_raises_exception_for_invalid_class():
    with pytest.raises(Exception):
        nodes.get_command_class(command_type='dummy1')


def test_python_execute_command_raises_exception_if_function_fails(mocker, monkeypatch):
    dummy_func = mocker.MagicMock(side_effect=Exception())

    class DummyModule:
        def __init__(self):
            self.func = dummy_func
    monkeypatch.setattr(nodes.utils, 'get_module_and_func_names', mocker.MagicMock(return_value=('idk', 'func')))
    monkeypatch.setattr(nodes.importlib, 'import_module', mocker.MagicMock(return_value=DummyModule()))

    monkeypatch.setattr(nodes.utils, 'filter_arguments_for_func', mocker.MagicMock(return_value={'a': 1}))

    py_exec = nodes.PythonExecutionType()
    with pytest.raises(Exception):
        py_exec.execute_command(command='test')


def test_python_execute_command_calls_with_no_parameters_if_none_sent(mocker, monkeypatch):
    dummy_func = mocker.MagicMock(return_value=None)

    class DummyModule:
        def __init__(self):
            self.func = dummy_func

    monkeypatch.setattr(nodes.utils, 'get_module_and_func_names', mocker.MagicMock(return_value=('idk', 'func')))
    monkeypatch.setattr(nodes.importlib, 'import_module', mocker.MagicMock(return_value=DummyModule()))

    monkeypatch.setattr(nodes.utils, 'filter_arguments_for_func', mocker.MagicMock(return_value={}))

    py_exec = nodes.PythonExecutionType()
    py_exec.execute_command(command='test')
    dummy_func.assert_called_once()


def test_python_execute_command_calls_with_parameters_if_sent_by_filter(mocker, monkeypatch):
    dummy_func = mocker.MagicMock(return_value=None)

    class DummyModule:
        def __init__(self):
            self.func = dummy_func

    monkeypatch.setattr(nodes.utils, 'get_module_and_func_names', mocker.MagicMock(return_value=('idk', 'func')))
    monkeypatch.setattr(nodes.importlib, 'import_module', mocker.MagicMock(return_value=DummyModule()))

    monkeypatch.setattr(nodes.utils, 'filter_arguments_for_func', mocker.MagicMock(return_value={'a': 1}))

    py_exec = nodes.PythonExecutionType()
    py_exec.execute_command(command='test')
    dummy_func.assert_called_once_with(a=1)


def test_python_execute_command_sends_no_mapped_variable_if_not_present_in_signature(mocker, monkeypatch):
    dummy_func = mocker.MagicMock(return_value=None)

    class DummyModule:
        def __init__(self):
            self.func = dummy_func

    monkeypatch.setattr(nodes.utils, 'get_module_and_func_names', mocker.MagicMock(return_value=('idk', 'func')))
    monkeypatch.setattr(nodes.importlib, 'import_module', mocker.MagicMock(return_value=DummyModule()))

    monkeypatch.setattr(nodes.utils, 'filter_arguments_for_func', mocker.MagicMock(return_value={'a': 1}))

    py_exec = nodes.PythonExecutionType()
    py_exec.execute_command(command='test', map_variable={'map_name': 'map_value'})
    dummy_func.assert_called_once_with(a=1)


def test_python_execute_command_sends_mapped_variable_if_present_in_signature(mocker, monkeypatch):
    dummy_func = mocker.MagicMock(return_value=None)

    class DummyModule:
        def __init__(self):
            self.func = dummy_func

    monkeypatch.setattr(nodes.utils, 'get_module_and_func_names', mocker.MagicMock(return_value=('idk', 'func')))
    monkeypatch.setattr(nodes.importlib, 'import_module', mocker.MagicMock(return_value=DummyModule()))

    monkeypatch.setattr(nodes.utils, 'filter_arguments_for_func', mocker.MagicMock(
        return_value={'a': 1, 'map_name': 'map_value'}))

    py_exec = nodes.PythonExecutionType()
    py_exec.execute_command(command='test')
    dummy_func.assert_called_once_with(a=1, map_name='map_value')


def test_python_execute_command_raises_exception_if_return_value_is_not_dict(mocker, monkeypatch):
    dummy_func = mocker.MagicMock(return_value=1)

    class DummyModule:
        def __init__(self):
            self.func = dummy_func

    monkeypatch.setattr(nodes.utils, 'get_module_and_func_names', mocker.MagicMock(return_value=('idk', 'func')))
    monkeypatch.setattr(nodes.importlib, 'import_module', mocker.MagicMock(return_value=DummyModule()))

    monkeypatch.setattr(nodes.utils, 'filter_arguments_for_func', mocker.MagicMock(return_value={'a': 1}))

    py_exec = nodes.PythonExecutionType()
    with pytest.raises(Exception):
        py_exec.execute_command(command='test', map_variable='iterme')


def test_python_execute_command_sets_env_variable_of_return_values(mocker, monkeypatch):
    dummy_func = mocker.MagicMock(return_value={'a': 10})

    class DummyModule:
        def __init__(self):
            self.func = dummy_func

    monkeypatch.setattr(nodes.utils, 'get_module_and_func_names', mocker.MagicMock(return_value=('idk', 'func')))
    monkeypatch.setattr(nodes.importlib, 'import_module', mocker.MagicMock(return_value=DummyModule()))

    monkeypatch.setattr(nodes.utils, 'filter_arguments_for_func', mocker.MagicMock(return_value={'a': 1}))

    py_exec = nodes.PythonExecutionType()
    py_exec.execute_command(command='test', map_variable='iterme')

    assert defaults.PARAMETER_PREFIX + 'a' in os.environ
    assert os.environ[defaults.PARAMETER_PREFIX + 'a'] == '10'

    del os.environ[defaults.PARAMETER_PREFIX + 'a']


def test_base_node_command_friendly_name_replaces_whitespace_with_character():
    node = nodes.BaseNode(name='test', internal_name='test', config='test_config', execution_type=None)

    assert node.command_friendly_name() == 'test'

    node.internal_name = 'test '
    assert node.command_friendly_name() == 'test' + defaults.COMMAND_FRIENDLY_CHARACTER


def test_base_node_get_internal_name_from_command_name_replaces_character_with_whitespace():
    assert nodes.BaseNode.get_internal_name_from_command_name('test') == 'test'

    assert nodes.BaseNode.get_internal_name_from_command_name('test%') == 'test '


def test_base_node_get_step_log_name_returns_internal_name_if_no_map_variable():
    node = nodes.BaseNode(name='test', internal_name='test', config='test_config', execution_type=None)

    assert node.get_step_log_name() == 'test'


def test_base_node_get_step_log_name_returns_map_modified_internal_name_if_map_variable():
    node = nodes.BaseNode(name='test', internal_name='test.' + defaults.MAP_PLACEHOLDER,
                          config='test_config', execution_type=None)

    assert node.get_step_log_name(map_variable={'map_key': 'a'}) == 'test.a'


def test_base_node_get_step_log_name_returns_map_modified_internal_name_if_map_variable_multiple():
    node = nodes.BaseNode(
        name='test', internal_name='test.' + defaults.MAP_PLACEHOLDER + '.step.' + defaults.MAP_PLACEHOLDER,
        config='test_config', execution_type=None)

    assert node.get_step_log_name(map_variable={'map_key': 'a', 'map_key1': 'b'}) == 'test.a.step.b'


def test_base_node_get_branch_log_name_returns_null_if_not_set():
    node = nodes.BaseNode(name='test', internal_name='test', config='test_config', execution_type=None)

    assert node.get_branch_log_name() is None


def test_base_node_get_branch_log_name_returns_internal_name_if_set():
    node = nodes.BaseNode(name='test', internal_name='test', config='test_config',
                          execution_type=None, internal_branch_name='test_internal')

    assert node.get_branch_log_name() is 'test_internal'


def test_base_node_get_branch_log_name_returns_map_modified_internal_name_if_map_variable():
    node = nodes.BaseNode(name='test', internal_name='test_', config='test_config',
                          execution_type=None, internal_branch_name='test.' + defaults.MAP_PLACEHOLDER)

    assert node.get_branch_log_name(map_variable={'map_key': 'a'}) == 'test.a'


def test_base_node_get_branch_log_name_returns_map_modified_internal_name_if_map_variable_multiple():
    node = nodes.BaseNode(name='test', internal_name='test_', config='test_config',
                          execution_type=None,
                          internal_branch_name='test.' + defaults.MAP_PLACEHOLDER + '.step.' + defaults.MAP_PLACEHOLDER)

    assert node.get_branch_log_name(map_variable={'map_key': 'a', 'map_key1': 'b'}) == 'test.a.step.b'


def test_base_node_get_on_failure_node_returns_none_if_not_defined():
    node = nodes.BaseNode(name='test', internal_name='test', config={}, execution_type=None)

    assert node.get_on_failure_node() is None


def test_base_node_get_on_failure_node_returns_node_name_if_defined():
    node = nodes.BaseNode(name='test', internal_name='test', config={'on_failure': 'fail'}, execution_type=None)

    assert node.get_on_failure_node() == 'fail'


def test_base_node_get_catalog_settings_returns_none_if_not_defined():
    node = nodes.BaseNode(name='test', internal_name='test', config={}, execution_type=None)

    assert node.get_catalog_settings() is None


def test_base_node_get_catalog_settings_returns_node_name_if_defined():
    node = nodes.BaseNode(name='test', internal_name='test', config={'catalog': 'some settings'}, execution_type=None)

    assert node.get_catalog_settings() == 'some settings'


def test_base_node_get_branch_by_name_raises_exception():
    node = nodes.BaseNode(name='test', internal_name='test', config={'catalog': 'some settings'}, execution_type=None)

    with pytest.raises(Exception):
        node.get_branch_by_name('fail')


def test_base_node_get_next_node_returns_config_next():
    node = nodes.BaseNode(name='test', internal_name='test', config={'next': 'IamNext'}, execution_type=None)

    assert node.get_next_node() == 'IamNext'


def test_base_node_get_mode_config_returns_mode_config_if_present():
    node = nodes.BaseNode(name='test', internal_name='test', config={'mode_config': 'some settings'},
                          execution_type=None)
    assert node.get_mode_config() == 'some settings'


def test_base_node_get_mode_config_returns_empty_dict_if_not_present():
    node = nodes.BaseNode(name='test', internal_name='test', config={}, execution_type=None)
    assert node.get_mode_config() == {}


def test_base_node_get_max_attempts_returns_max_attempts_as_in_config():
    node = nodes.BaseNode(name='test', internal_name='test', config={'retry': 2}, execution_type=None)
    assert node.get_max_attempts() == 2


def test_base_node_get_max_attempts_returns_max_attempts_as_1_if_not_in_config():
    node = nodes.BaseNode(name='test', internal_name='test', config={}, execution_type=None)
    assert node.get_max_attempts() == 1


def test_base_node_execute_raises_not_implemented_error():
    node = nodes.BaseNode(name='test', internal_name='test', config={}, execution_type=None)

    with pytest.raises(NotImplementedError):
        node.execute(executor='test')


def test_base_node_execute_as_graph_raises_not_implemented_error():
    node = nodes.BaseNode(name='test', internal_name='test', config={}, execution_type=None)

    with pytest.raises(NotImplementedError):
        node.execute_as_graph(executor='test')


def test_validate_node_gets_specs_from_default_specs(mocker, monkeypatch):
    mock_load_yaml = mocker.MagicMock()

    monkeypatch.setattr(nodes.utils, 'load_yaml', mock_load_yaml)

    node = nodes.BaseNode(name='test', internal_name='test', config={}, execution_type=None)

    with pytest.raises(Exception):
        nodes.validate_node(node)
    args, _ = mock_load_yaml.call_args
    assert args[0].endswith(defaults.NODE_SPEC_FILE)


def test_validate_node_raises_exception_for_unspecified_node(mocker, monkeypatch):
    dummy_specs = {'test': {}}
    monkeypatch.setattr(nodes.utils, 'load_yaml', mocker.MagicMock(return_value=dummy_specs))

    node = nodes.BaseNode(name='test', internal_name='test', config={}, execution_type=None)

    node.node_type = 'test1'
    with pytest.raises(Exception):
        nodes.validate_node(node)


def test_validate_node_does_not_raise_exception_for_specified_node(mocker, monkeypatch):
    dummy_specs = {'dummy': {}}
    monkeypatch.setattr(nodes.utils, 'load_yaml', mocker.MagicMock(return_value=dummy_specs))

    node = nodes.BaseNode(name='test', internal_name='test', config={}, execution_type=None)

    node.node_type = 'dummy'

    nodes.validate_node(node)


def test_validate_node_sends_message_back_if_dot_present_in_name(mocker, monkeypatch):
    dummy_specs = {'dummy': {}}
    monkeypatch.setattr(nodes.utils, 'load_yaml', mocker.MagicMock(return_value=dummy_specs))

    node = nodes.BaseNode(name='test.', internal_name='test', config={}, execution_type=None)

    node.node_type = 'dummy'
    messages = nodes.validate_node(node)

    assert len(messages) == 1
    assert messages[0] == 'Node names cannot have . in them'


def test_validate_node_sends_message_back_if_character_present_in_name(mocker, monkeypatch):
    dummy_specs = {'dummy': {}}
    monkeypatch.setattr(nodes.utils, 'load_yaml', mocker.MagicMock(return_value=dummy_specs))

    node = nodes.BaseNode(name='test%', internal_name='test', config={}, execution_type=None)

    node.node_type = 'dummy'
    messages = nodes.validate_node(node)

    assert len(messages) == 1
    assert messages[0] == "Node names cannot have '%' in them"


def test_validate_node_messages_empty_if_name_is_valid(mocker, monkeypatch):
    dummy_specs = {'dummy': {}}
    monkeypatch.setattr(nodes.utils, 'load_yaml', mocker.MagicMock(return_value=dummy_specs))

    node = nodes.BaseNode(name='test', internal_name='test', config={}, execution_type=None)

    node.node_type = 'dummy'
    messages = nodes.validate_node(node)

    assert len(messages) == 0


def test_validate_node_sends_messages_if_required_are_not_present(mocker, monkeypatch):
    dummy_specs = {'dummy': {'required': ['dummy_required']}}
    monkeypatch.setattr(nodes.utils, 'load_yaml', mocker.MagicMock(return_value=dummy_specs))

    node = nodes.BaseNode(name='test', internal_name='test', config={'dummy_required1': True}, execution_type=None)

    node.node_type = 'dummy'
    messages = nodes.validate_node(node)

    assert len(messages) == 1
    assert messages[0] == 'test should have dummy_required field'


def test_validate_node_sends_empty_if_required_present(mocker, monkeypatch):
    dummy_specs = {'dummy': {'required': ['dummy_required']}}
    monkeypatch.setattr(nodes.utils, 'load_yaml', mocker.MagicMock(return_value=dummy_specs))

    node = nodes.BaseNode(name='test', internal_name='test', config={'dummy_required': True}, execution_type=None)

    node.node_type = 'dummy'
    messages = nodes.validate_node(node)

    assert len(messages) == 0


def test_validate_node_sends_messages_if_error_on_are_present(mocker, monkeypatch):
    dummy_specs = {'dummy': {'error_on': ['dummy_required']}}
    monkeypatch.setattr(nodes.utils, 'load_yaml', mocker.MagicMock(return_value=dummy_specs))

    node = nodes.BaseNode(name='test', internal_name='test', config={'dummy_required': True}, execution_type=None)

    node.node_type = 'dummy'
    messages = nodes.validate_node(node)

    assert len(messages) == 1
    assert messages[0] == 'test should not have dummy_required field'


def test_validate_node_sends_empty_if_error_on_not_present(mocker, monkeypatch):
    dummy_specs = {'dummy': {'error_on': ['dummy_required']}}
    monkeypatch.setattr(nodes.utils, 'load_yaml', mocker.MagicMock(return_value=dummy_specs))

    node = nodes.BaseNode(name='test', internal_name='test', config={'dummy_required1': True}, execution_type=None)

    node.node_type = 'dummy'
    messages = nodes.validate_node(node)

    assert len(messages) == 0


def test_task_node_mocks_if_mock_is_true(mocker, monkeypatch):
    mock_attempt_log = mocker.MagicMock()

    mock_executor = mocker.MagicMock()
    mock_executor.run_log_store.create_attempt_log = mocker.MagicMock(return_value=mock_attempt_log)

    task_node = nodes.TaskNode(name='test', internal_name='test', config={'command': 'nocommand'}, execution_type=None)

    task_node.execute(executor=mock_executor, mock=True)

    assert mock_attempt_log.status == defaults.SUCCESS


def test_task_node_sets_attempt_log_fail_in_exception_of_execution(mocker, monkeypatch):
    mock_attempt_log = mocker.MagicMock()

    mock_executor = mocker.MagicMock()
    mock_executor.run_log_store.create_attempt_log = mocker.MagicMock(return_value=mock_attempt_log)

    task_node = nodes.TaskNode(name='test', internal_name='test', config={'command': 'nocommand'}, execution_type=None)

    task_node.execution_type = mocker.MagicMock()
    task_node.execution_type.execute_command = mocker.MagicMock(side_effect=Exception())

    task_node.execute(executor=mock_executor)

    assert mock_attempt_log.status == defaults.FAIL


def test_task_node_sets_attempt_log_success_in_no_exception_of_execution(mocker, monkeypatch):
    mock_attempt_log = mocker.MagicMock()

    mock_executor = mocker.MagicMock()
    mock_executor.run_log_store.create_attempt_log = mocker.MagicMock(return_value=mock_attempt_log)

    task_node = nodes.TaskNode(name='test', internal_name='test', config={'command': 'nocommand'}, execution_type=None)

    task_node.execution_type = mocker.MagicMock()
    task_node.execution_type.execute_command = mocker.MagicMock()

    task_node.execute(executor=mock_executor)

    assert mock_attempt_log.status == defaults.SUCCESS


def test_task_node_sends_map_variable_if_sent_to_execution(mocker, monkeypatch):
    mock_attempt_log = mocker.MagicMock()
    mock_execution_command = mocker.MagicMock()

    mock_executor = mocker.MagicMock()
    mock_executor.run_log_store.create_attempt_log = mocker.MagicMock(return_value=mock_attempt_log)

    task_node = nodes.TaskNode(name='test', internal_name='test', config={'command': 'nocommand'}, execution_type=None)

    task_node.execution_type = mocker.MagicMock()
    task_node.execution_type.execute_command = mock_execution_command

    task_node.execute(executor=mock_executor, map_variable={'map_key': 'a'})

    assert mock_attempt_log.status == defaults.SUCCESS
    mock_execution_command.assert_called_once_with('nocommand', map_variable={'map_key': 'a'})


def test_task_node_execute_as_graph_raises_exception():
    task_node = nodes.TaskNode(name='test', internal_name='test', config={'command': 'nocommand'}, execution_type=None)

    with pytest.raises(Exception):
        task_node.execute_as_graph(None)


def test_get_node_class_returns_the_correct_subclasses():
    class DummyNode(nodes.BaseNode):
        node_type = 'dummy'

    obj = nodes.get_node_class(node_type='dummy')
    assert obj == DummyNode


def test_get_node_class_raises_exception_if_node_class_not_found():
    with pytest.raises(Exception):
        nodes.get_node_class(node_type='dummy1')


def test_fail_node_sets_branch_log_fail(mocker, monkeypatch):
    mock_attempt_log = mocker.MagicMock()
    mock_branch_log = mocker.MagicMock()

    mock_executor = mocker.MagicMock()
    mock_executor.run_log_store.create_attempt_log = mocker.MagicMock(return_value=mock_attempt_log)
    mock_executor.run_log_store.get_branch_log = mocker.MagicMock(return_value=mock_branch_log)

    node = nodes.FailNode(name='test', internal_name='test', config={'command': 'nocommand'}, execution_type=None)

    node.execute(executor=mock_executor)

    assert mock_attempt_log.status == defaults.SUCCESS
    assert mock_branch_log.status == defaults.FAIL


def test_fail_node_sets_attempt_log_success_even_in_exception(mocker, monkeypatch):
    mock_attempt_log = mocker.MagicMock()

    mock_executor = mocker.MagicMock()
    mock_executor.run_log_store.create_attempt_log = mocker.MagicMock(return_value=mock_attempt_log)
    mock_executor.run_log_store.get_branch_log = mocker.MagicMock(side_effect=Exception())

    node = nodes.FailNode(name='test', internal_name='test', config={'command': 'nocommand'}, execution_type=None)

    node.execute(executor=mock_executor)

    assert mock_attempt_log.status == defaults.SUCCESS


def test_fail_node_execute_as_graph_raises_exception():
    fail_node = nodes.FailNode(name='test', internal_name='test', config={'command': 'nocommand'}, execution_type=None)

    with pytest.raises(Exception):
        fail_node.execute_as_graph(None)


def test_success_node_sets_branch_log_success(mocker, monkeypatch):
    mock_attempt_log = mocker.MagicMock()
    mock_branch_log = mocker.MagicMock()

    mock_executor = mocker.MagicMock()
    mock_executor.run_log_store.create_attempt_log = mocker.MagicMock(return_value=mock_attempt_log)
    mock_executor.run_log_store.get_branch_log = mocker.MagicMock(return_value=mock_branch_log)

    node = nodes.SuccessNode(name='test', internal_name='test', config={'command': 'nocommand'}, execution_type=None)

    node.execute(executor=mock_executor)

    assert mock_attempt_log.status == defaults.SUCCESS
    assert mock_branch_log.status == defaults.SUCCESS


def test_success_node_sets_attempt_log_success_even_in_exception(mocker, monkeypatch):
    mock_attempt_log = mocker.MagicMock()

    mock_executor = mocker.MagicMock()
    mock_executor.run_log_store.create_attempt_log = mocker.MagicMock(return_value=mock_attempt_log)
    mock_executor.run_log_store.get_branch_log = mocker.MagicMock(side_effect=Exception())

    node = nodes.SuccessNode(name='test', internal_name='test', config={'command': 'nocommand'}, execution_type=None)

    node.execute(executor=mock_executor)

    assert mock_attempt_log.status == defaults.SUCCESS


def test_success_node_execute_as_graph_raises_exception():
    success_node = nodes.SuccessNode(name='test', internal_name='test',
                                     config={'command': 'nocommand'}, execution_type=None)

    with pytest.raises(Exception):
        success_node.execute_as_graph(None)


def test_parallel_node_raises_exception_for_empty_branches():
    with pytest.raises(Exception):
        nodes.ParallelNode(name='test', internal_name='test', config={'branches': {}}, execution_type='python')


def test_parallel_node_get_sub_graphs_creates_graphs(mocker, monkeypatch):
    mock_create_graph = mocker.MagicMock(return_value='agraphobject')

    monkeypatch.setattr(nodes, 'create_graph', mock_create_graph)

    parallel_config = {
        'branches': {
            'a': {},
            'b': {}
        }
    }
    node = nodes.ParallelNode(name='test', internal_name='test', config=parallel_config, execution_type='python')
    assert mock_create_graph.call_count == 2
    assert len(node.branches.items()) == 2


def test_parallel_node_get_branch_by_name_raises_exception_if_branch_not_found(mocker, monkeypatch):
    monkeypatch.setattr(nodes.ParallelNode, 'get_sub_graphs', mocker.MagicMock())

    node = nodes.ParallelNode(name='test', internal_name='test', config={}, execution_type='python')

    with pytest.raises(Exception):
        node.get_branch_by_name('a')


def test_parallel_node_get_branch_by_name_returns_branch_if_found(mocker, monkeypatch):
    monkeypatch.setattr(nodes.ParallelNode, 'get_sub_graphs', mocker.MagicMock())

    node = nodes.ParallelNode(name='test', internal_name='test', config={}, execution_type='python')
    node.branches = {'a': 'somegraph'}

    assert node.get_branch_by_name('a') == 'somegraph'


def test_parallel_node_execute_raises_exception(mocker, monkeypatch):
    monkeypatch.setattr(nodes.ParallelNode, 'get_sub_graphs', mocker.MagicMock())

    node = nodes.ParallelNode(name='test', internal_name='test', config={}, execution_type='python')

    with pytest.raises(Exception):
        node.execute(executor='test')


def test_nodes_map_node_raises_exception_if_config_not_have_iterate_on():
    with pytest.raises(Exception):
        nodes.MapNode(name='test', internal_name='test', config={}, execution_type='test')


def test_nodes_map_node_raises_exception_if_config_not_have_iterate_as():
    with pytest.raises(Exception):
        nodes.MapNode(name='test', internal_name='test', config={'iterate_on': 'y'}, execution_type='test')


def test_nodes_map_node_names_the_branch_as_defaults_place_holder(monkeypatch, mocker):
    monkeypatch.setattr(nodes.MapNode, 'get_sub_graph', mocker.MagicMock())

    node = nodes.MapNode(name='test', internal_name='test', config={
                         'iterate_on': 'a', 'iterate_as': 'y_i'}, execution_type='test')

    assert node.branch_placeholder_name == defaults.MAP_PLACEHOLDER


def test_nodes_map_get_sub_graph_calls_create_graph_with_correct_naming(mocker, monkeypatch):
    mock_create_graph = mocker.MagicMock()
    monkeypatch.setattr(nodes, 'create_graph', mock_create_graph)

    _ = nodes.MapNode(name='test', internal_name='test', config={
        'iterate_on': 'a', 'iterate_as': 'y_i', 'branch': {}}, execution_type='test')

    mock_create_graph.assert_called_once_with({}, internal_branch_name='test.' + defaults.MAP_PLACEHOLDER)


def test_nodes_map_get_branch_by_name_returns_a_sub_graph(mocker, monkeypatch):
    mock_create_graph = mocker.MagicMock(return_value='a')
    monkeypatch.setattr(nodes, 'create_graph', mock_create_graph)

    node = nodes.MapNode(name='test', internal_name='test', config={
        'iterate_on': 'a', 'iterate_as': 'y_i', 'branch': {}}, execution_type='test')

    assert node.get_branch_by_name('anyname') == 'a'


def test_nodes_map_node_execute_raises_exception(mocker, monkeypatch):
    monkeypatch.setattr(nodes.MapNode, 'get_sub_graph', mocker.MagicMock())

    node = nodes.MapNode(name='test', internal_name='test', config={
                         'iterate_on': 'a', 'iterate_as': 'y_i'}, execution_type='test')

    with pytest.raises(Exception):
        node.execute('dummy')


def test_nodes_dag_node_raises_exception_if_dag_definition_is_not_present():
    with pytest.raises(Exception):
        nodes.DagNode(name='test', internal_name='test', config={}, execution_type='test')


def test_node_dag_node_get_sub_graph_raises_exception_if_dag_block_not_present(mocker, monkeypatch):
    mock_load_yaml = mocker.MagicMock(return_value={})

    monkeypatch.setattr(nodes.utils, 'load_yaml', mock_load_yaml)

    with pytest.raises(Exception):
        nodes.DagNode(name='test', internal_name='test', config={'dag_definition': 'a'}, execution_type='test')


def test_nodes_dag_node_get_sub_graph_calls_create_graph_with_correct_parameters(mocker, monkeypatch):
    mock_load_yaml = mocker.MagicMock(return_value={'dag': 'a'})
    mock_create_graph = mocker.MagicMock(return_value='branch')

    monkeypatch.setattr(nodes.utils, 'load_yaml', mock_load_yaml)
    monkeypatch.setattr(nodes, 'create_graph', mock_create_graph)

    _ = nodes.DagNode(name='test', internal_name='test', config={'dag_definition': 'a'}, execution_type='test')

    mock_create_graph.assert_called_once_with('a', internal_branch_name='test.' + defaults.DAG_BRANCH_NAME)


def test_nodes_dag_node_get_branch_by_name_raises_exception_if_branch_name_is_invalid(mocker, monkeypatch):
    monkeypatch.setattr(nodes.DagNode, 'get_sub_graph', mocker.MagicMock(return_value='branch'))

    node = nodes.DagNode(name='test', internal_name='test', config={'dag_definition': 'a'}, execution_type='test')

    with pytest.raises(Exception):
        node.get_branch_by_name('test')


def test_nodes_dag_node_get_branch_by_name_returns_if_branch_name_is_valid(mocker, monkeypatch):
    monkeypatch.setattr(nodes.DagNode, 'get_sub_graph', mocker.MagicMock(return_value='branch'))

    node = nodes.DagNode(name='test', internal_name='test', config={'dag_definition': 'a'}, execution_type='test')

    assert node.get_branch_by_name('test.' + defaults.DAG_BRANCH_NAME) == 'branch'


def test_nodes_dag_node_execute_raises_exception(mocker, monkeypatch):
    monkeypatch.setattr(nodes.DagNode, 'get_sub_graph', mocker.MagicMock(return_value='branch'))

    node = nodes.DagNode(name='test', internal_name='test', config={'dag_definition': 'a'}, execution_type='test')

    with pytest.raises(Exception):
        node.execute('dummy')


def test_nodes_as_is_node_defaults_render_string_to_none():
    node = nodes.AsISNode(name='test', internal_name='test', config={}, execution_type='test')

    assert node.render_string is None


def test_nodes_as_is_node_accepts_what_is_given():
    node = nodes.AsISNode(name='test', internal_name='test', config={'render_string': 'test'}, execution_type='test')

    assert node.render_string == 'test'


def test_as_is_node_execute_as_graph_raises_exception():
    as_is_node = nodes.AsISNode(name='test', internal_name='test',
                                config={'command': 'nocommand'}, execution_type=None)

    with pytest.raises(Exception):
        as_is_node.execute_as_graph(None)


def test_as_is_node_sets_attempt_log_success(mocker, monkeypatch):
    mock_attempt_log = mocker.MagicMock()

    mock_executor = mocker.MagicMock()
    mock_executor.run_log_store.create_attempt_log = mocker.MagicMock(return_value=mock_attempt_log)

    node = nodes.AsISNode(name='test', internal_name='test', config={}, execution_type=None)

    node.execute(executor=mock_executor)

    assert mock_attempt_log.status == defaults.SUCCESS


def test_is_terminal_node_when_has_next():
    node = nodes.BaseNode(name='test', internal_name='test_', config={'next': 'yes'},
                          execution_type=None, internal_branch_name='test_')

    assert not node.is_terminal_node()


def test_is_terminal_node_when_no_next():
    node = nodes.BaseNode(name='test', internal_name='test_', config={'none': 'no'},
                          execution_type=None, internal_branch_name='test_')

    assert node.is_terminal_node()


def test_get_neighbors_no_neighbors():
    node = nodes.BaseNode(name='test', internal_name='test_', config={},
                          execution_type=None, internal_branch_name='test_')
    assert node.get_neighbors() == []


def test_get_neighbors_only_next():
    node = nodes.BaseNode(name='test', internal_name='test_', config={'next': 'a'},
                          execution_type=None, internal_branch_name='test_')
    neighbors = node.get_neighbors()
    assert len(neighbors) == 1
    assert neighbors[0] == 'a'


def test_get_neighbors_both_next_and_on_failure():
    node = nodes.BaseNode(name='test', internal_name='test_', config={'next': 'a', 'on_failure': 'b'},
                          execution_type=None, internal_branch_name='test_')
    neighbors = node.get_neighbors()
    assert len(neighbors) == 2
    assert neighbors[0] == 'a'
    assert neighbors[1] == 'b'
