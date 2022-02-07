import os

import pytest

from magnus import tasks


def test_base_task_execute_command_raises_not_implemented_error():
    base_execution_type = tasks.BaseTaskType(command=None)

    with pytest.raises(NotImplementedError):
        base_execution_type.execute_command()


def test_base_task_get_parameters_gets_from_utils(mocker, monkeypatch):
    mock_get_user_set_parameters = mocker.MagicMock()

    monkeypatch.setattr(tasks.utils, 'get_user_set_parameters', mock_get_user_set_parameters)

    base_execution_type = tasks.BaseTaskType(command=None)

    base_execution_type.get_parameters()
    mock_get_user_set_parameters.assert_called_once_with(remove=False)


def test_python_task_command_raises_exception_if_function_fails(mocker, monkeypatch):
    dummy_func = mocker.MagicMock(side_effect=Exception())

    class DummyModule:
        def __init__(self):
            self.func = dummy_func
    monkeypatch.setattr(tasks.utils, 'get_module_and_func_names', mocker.MagicMock(return_value=('idk', 'func')))
    monkeypatch.setattr(tasks.importlib, 'import_module', mocker.MagicMock(return_value=DummyModule()))

    monkeypatch.setattr(tasks.utils, 'filter_arguments_for_func', mocker.MagicMock(return_value={'a': 1}))

    py_exec = tasks.PythonTaskType(command='test')
    with pytest.raises(Exception):
        py_exec.execute_command()


# def test_python_execute_command_calls_with_no_parameters_if_none_sent(mocker, monkeypatch):
#     dummy_func = mocker.MagicMock(return_value=None)

#     class DummyModule:
#         def __init__(self):
#             self.func = dummy_func

#     monkeypatch.setattr(tasks.utils, 'get_module_and_func_names', mocker.MagicMock(return_value=('idk', 'func')))
#     monkeypatch.setattr(tasks.importlib, 'import_module', mocker.MagicMock(return_value=DummyModule()))

#     monkeypatch.setattr(nodes.utils, 'filter_arguments_for_func', mocker.MagicMock(return_value={}))

#     py_exec = nodes.PythonExecutionType()
#     py_exec.execute_command(command='test')
#     dummy_func.assert_called_once()


# def test_python_execute_command_calls_with_parameters_if_sent_by_filter(mocker, monkeypatch):
#     dummy_func = mocker.MagicMock(return_value=None)

#     class DummyModule:
#         def __init__(self):
#             self.func = dummy_func

#     monkeypatch.setattr(nodes.utils, 'get_module_and_func_names', mocker.MagicMock(return_value=('idk', 'func')))
#     monkeypatch.setattr(nodes.importlib, 'import_module', mocker.MagicMock(return_value=DummyModule()))

#     monkeypatch.setattr(nodes.utils, 'filter_arguments_for_func', mocker.MagicMock(return_value={'a': 1}))

#     py_exec = nodes.PythonExecutionType()
#     py_exec.execute_command(command='test')
#     dummy_func.assert_called_once_with(a=1)


# def test_python_execute_command_sends_no_mapped_variable_if_not_present_in_signature(mocker, monkeypatch):
#     dummy_func = mocker.MagicMock(return_value=None)

#     class DummyModule:
#         def __init__(self):
#             self.func = dummy_func

#     monkeypatch.setattr(nodes.utils, 'get_module_and_func_names', mocker.MagicMock(return_value=('idk', 'func')))
#     monkeypatch.setattr(nodes.importlib, 'import_module', mocker.MagicMock(return_value=DummyModule()))

#     monkeypatch.setattr(nodes.utils, 'filter_arguments_for_func', mocker.MagicMock(return_value={'a': 1}))

#     py_exec = nodes.PythonExecutionType()
#     py_exec.execute_command(command='test', map_variable={'map_name': 'map_value'})
#     dummy_func.assert_called_once_with(a=1)


# def test_python_execute_command_sends_mapped_variable_if_present_in_signature(mocker, monkeypatch):
#     dummy_func = mocker.MagicMock(return_value=None)

#     class DummyModule:
#         def __init__(self):
#             self.func = dummy_func

#     monkeypatch.setattr(nodes.utils, 'get_module_and_func_names', mocker.MagicMock(return_value=('idk', 'func')))
#     monkeypatch.setattr(nodes.importlib, 'import_module', mocker.MagicMock(return_value=DummyModule()))

#     monkeypatch.setattr(nodes.utils, 'filter_arguments_for_func', mocker.MagicMock(
#         return_value={'a': 1, 'map_name': 'map_value'}))

#     py_exec = nodes.PythonExecutionType()
#     py_exec.execute_command(command='test')
#     dummy_func.assert_called_once_with(a=1, map_name='map_value')


# def test_python_execute_command_raises_exception_if_return_value_is_not_dict(mocker, monkeypatch):
#     dummy_func = mocker.MagicMock(return_value=1)

#     class DummyModule:
#         def __init__(self):
#             self.func = dummy_func

#     monkeypatch.setattr(nodes.utils, 'get_module_and_func_names', mocker.MagicMock(return_value=('idk', 'func')))
#     monkeypatch.setattr(nodes.importlib, 'import_module', mocker.MagicMock(return_value=DummyModule()))

#     monkeypatch.setattr(nodes.utils, 'filter_arguments_for_func', mocker.MagicMock(return_value={'a': 1}))

#     py_exec = nodes.PythonExecutionType()
#     with pytest.raises(Exception):
#         py_exec.execute_command(command='test', map_variable='iterme')


# def test_python_execute_command_sets_env_variable_of_return_values(mocker, monkeypatch):
#     dummy_func = mocker.MagicMock(return_value={'a': 10})

#     class DummyModule:
#         def __init__(self):
#             self.func = dummy_func

#     monkeypatch.setattr(nodes.utils, 'get_module_and_func_names', mocker.MagicMock(return_value=('idk', 'func')))
#     monkeypatch.setattr(nodes.importlib, 'import_module', mocker.MagicMock(return_value=DummyModule()))

#     monkeypatch.setattr(nodes.utils, 'filter_arguments_for_func', mocker.MagicMock(return_value={'a': 1}))

#     py_exec = nodes.PythonExecutionType()
#     py_exec.execute_command(command='test', map_variable='iterme')

#     assert defaults.PARAMETER_PREFIX + 'a' in os.environ
#     assert os.environ[defaults.PARAMETER_PREFIX + 'a'] == '10'

#     del os.environ[defaults.PARAMETER_PREFIX + 'a']
