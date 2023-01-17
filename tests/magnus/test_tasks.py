import os

import pytest

from magnus import defaults, tasks


def test_base_task_execute_command_raises_not_implemented_error():
    base_execution_type = tasks.BaseTaskType(command=None)

    with pytest.raises(NotImplementedError):
        base_execution_type.execute_command()


def test_base_task_get_parameters_gets_from_utils(mocker, monkeypatch):
    mock_get_user_set_parameters = mocker.MagicMock()

    monkeypatch.setattr(tasks.utils, 'get_user_set_parameters', mock_get_user_set_parameters)

    base_execution_type = tasks.BaseTaskType(command=None)

    base_execution_type._get_parameters()
    mock_get_user_set_parameters.assert_called_once_with(remove=False)


def test_base_task_set_parameters_does_nothing_if_no_parameters_sent():
    base_execution_type = tasks.BaseTaskType(command=None)
    base_execution_type._set_parameters()


def test_base_task_set_parameters_raises_exception_if_parameters_are_not_dict():
    base_execution_type = tasks.BaseTaskType(command=None)
    with pytest.raises(Exception):
        base_execution_type._set_parameters(parameters=[1, 2, 3])


def test_base_task_set_parameters_sets_environ_vars_if_sent(mocker, monkeypatch):
    mock_os_environ = {}

    monkeypatch.setattr(tasks.os, 'environ', mock_os_environ)

    base_execution_type = tasks.BaseTaskType(command=None)
    base_execution_type._set_parameters(parameters={'x': 10})

    assert mock_os_environ[defaults.PARAMETER_PREFIX + 'x'] == '10'


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


def test_python_task_command_calls_with_no_parameters_if_none_sent(mocker, monkeypatch):
    dummy_func = mocker.MagicMock(return_value=None)

    class DummyModule:
        def __init__(self):
            self.func = dummy_func

    monkeypatch.setattr(tasks.utils, 'get_module_and_func_names', mocker.MagicMock(return_value=('idk', 'func')))
    monkeypatch.setattr(tasks.importlib, 'import_module', mocker.MagicMock(return_value=DummyModule()))

    monkeypatch.setattr(tasks.utils, 'filter_arguments_for_func', mocker.MagicMock(return_value={}))

    py_exec = tasks.PythonTaskType(command='test')
    py_exec.execute_command()
    dummy_func.assert_called_once()


def test_python_task_command_calls_with_parameters_if_sent_by_filter(mocker, monkeypatch):
    dummy_func = mocker.MagicMock(return_value=None)

    class DummyModule:
        def __init__(self):
            self.func = dummy_func

    monkeypatch.setattr(tasks.utils, 'get_module_and_func_names', mocker.MagicMock(return_value=('idk', 'func')))
    monkeypatch.setattr(tasks.importlib, 'import_module', mocker.MagicMock(return_value=DummyModule()))

    monkeypatch.setattr(tasks.utils, 'filter_arguments_for_func', mocker.MagicMock(return_value={'a': 1}))

    py_exec = tasks.PythonTaskType(command='test')
    py_exec.execute_command()
    dummy_func.assert_called_once_with(a=1)


def test_python_task_command_sends_no_mapped_variable_if_not_present_in_signature(mocker, monkeypatch):
    dummy_func = mocker.MagicMock(return_value=None)

    class DummyModule:
        def __init__(self):
            self.func = dummy_func

    monkeypatch.setattr(tasks.utils, 'get_module_and_func_names', mocker.MagicMock(return_value=('idk', 'func')))
    monkeypatch.setattr(tasks.importlib, 'import_module', mocker.MagicMock(return_value=DummyModule()))

    monkeypatch.setattr(tasks.utils, 'filter_arguments_for_func', mocker.MagicMock(return_value={'a': 1}))

    py_exec = tasks.PythonTaskType(command='test',)
    py_exec.execute_command(map_variable={'map_name': 'map_value'})
    dummy_func.assert_called_once_with(a=1)


def test_python_task_command_sends_mapped_variable_if_present_in_signature(mocker, monkeypatch):
    dummy_func = mocker.MagicMock(return_value=None)

    class DummyModule:
        def __init__(self):
            self.func = dummy_func

    monkeypatch.setattr(tasks.utils, 'get_module_and_func_names', mocker.MagicMock(return_value=('idk', 'func')))
    monkeypatch.setattr(tasks.importlib, 'import_module', mocker.MagicMock(return_value=DummyModule()))

    monkeypatch.setattr(tasks.utils, 'filter_arguments_for_func', mocker.MagicMock(
        return_value={'a': 1, 'map_name': 'map_value'}))

    py_exec = tasks.PythonTaskType(command='test')
    py_exec.execute_command()
    dummy_func.assert_called_once_with(a=1, map_name='map_value')


def test_python_task_command_raises_exception_if_return_value_is_not_dict(mocker, monkeypatch):
    dummy_func = mocker.MagicMock(return_value=1)

    class DummyModule:
        def __init__(self):
            self.func = dummy_func

    monkeypatch.setattr(tasks.utils, 'get_module_and_func_names', mocker.MagicMock(return_value=('idk', 'func')))
    monkeypatch.setattr(tasks.importlib, 'import_module', mocker.MagicMock(return_value=DummyModule()))

    monkeypatch.setattr(tasks.utils, 'filter_arguments_for_func', mocker.MagicMock(return_value={'a': 1}))

    py_exec = tasks.PythonTaskType(command='test')
    with pytest.raises(Exception):
        py_exec.execute_command(map_variable='iterme')


def test_python_task_command_sets_env_variable_of_return_values(mocker, monkeypatch):
    dummy_func = mocker.MagicMock(return_value={'a': 10})

    class DummyModule:
        def __init__(self):
            self.func = dummy_func

    monkeypatch.setattr(tasks.utils, 'get_module_and_func_names', mocker.MagicMock(return_value=('idk', 'func')))
    monkeypatch.setattr(tasks.importlib, 'import_module', mocker.MagicMock(return_value=DummyModule()))

    monkeypatch.setattr(tasks.utils, 'filter_arguments_for_func', mocker.MagicMock(return_value={'a': 1}))

    py_exec = tasks.PythonTaskType(command='test')
    py_exec.execute_command(map_variable='iterme')

    assert defaults.PARAMETER_PREFIX + 'a' in os.environ
    assert os.environ[defaults.PARAMETER_PREFIX + 'a'] == '10'

    del os.environ[defaults.PARAMETER_PREFIX + 'a']


def test_python_lambda_task_type_execute_command_raises_for_under_and_dunder():
    lambda_exec = tasks.PythonLambdaTaskType(command='_ and __')

    with pytest.raises(Exception):
        lambda_exec.execute_command()


def test_notebook_raises_exception_if_command_is_not_a_notebook():
    with pytest.raises(Exception):
        task_exec = tasks.NotebookTaskType(command='path to notebook')


def test_notebook_raises_exception_if_papermill_is_not_installed(mocker, monkeypatch):
    monkeypatch.setattr(tasks, 'pm', None)

    task_exec = tasks.NotebookTaskType(command='path to notebook.ipynb')

    with pytest.raises(Exception):
        task_exec.execute_command()
