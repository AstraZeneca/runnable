import contextlib
import os

import pytest
from pydantic import BaseModel

from runnable import defaults, tasks


@pytest.fixture
def configuration():
    return {"node_name": "dummy", "task_type": "dummy"}


def test_base_task_execute_command_raises_not_implemented_error(configuration):
    base_execution_type = tasks.BaseTaskType(**configuration)

    with pytest.raises(NotImplementedError):
        base_execution_type.execute_command()


def test_python_task_command_raises_exception_if_function_fails(mocker, monkeypatch, configuration):
    dummy_func = mocker.MagicMock(side_effect=Exception())

    class DummyModule:
        def __init__(self):
            self.func = dummy_func

    monkeypatch.setattr(tasks.utils, "get_module_and_attr_names", mocker.MagicMock(return_value=("idk", "func")))
    monkeypatch.setattr(tasks.importlib, "import_module", mocker.MagicMock(return_value=DummyModule()))

    monkeypatch.setattr(tasks.BaseTaskType, "output_to_file", mocker.MagicMock(return_value=contextlib.nullcontext()))

    monkeypatch.setattr(tasks.parameters, "filter_arguments_for_func", mocker.MagicMock(return_value={"a": 1}))

    configuration["command"] = "dummy"
    py_exec = tasks.PythonTaskType(**configuration)
    with pytest.raises(Exception):
        py_exec.execute_command()


def test_python_task_command_calls_with_no_parameters_if_none_sent(mocker, monkeypatch, configuration):
    dummy_func = mocker.MagicMock(return_value=None)

    class DummyModule:
        def __init__(self):
            self.func = dummy_func

    monkeypatch.setattr(tasks.utils, "get_module_and_attr_names", mocker.MagicMock(return_value=("idk", "func")))
    monkeypatch.setattr(tasks.importlib, "import_module", mocker.MagicMock(return_value=DummyModule()))
    monkeypatch.setattr(tasks.BaseTaskType, "output_to_file", mocker.MagicMock(return_value=contextlib.nullcontext()))
    monkeypatch.setattr(tasks.parameters, "filter_arguments_for_func", mocker.MagicMock(return_value={}))

    configuration["command"] = "dummy"
    py_exec = tasks.PythonTaskType(**configuration)

    py_exec.execute_command()
    dummy_func.assert_called_once()


def test_python_task_command_calls_with_parameters_if_sent_by_filter(mocker, monkeypatch, configuration):
    dummy_func = mocker.MagicMock(return_value=None)

    class DummyModule:
        def __init__(self):
            self.func = dummy_func

    monkeypatch.setattr(tasks.utils, "get_module_and_attr_names", mocker.MagicMock(return_value=("idk", "func")))
    monkeypatch.setattr(tasks.importlib, "import_module", mocker.MagicMock(return_value=DummyModule()))

    monkeypatch.setattr(tasks.BaseTaskType, "output_to_file", mocker.MagicMock(return_value=contextlib.nullcontext()))
    monkeypatch.setattr(tasks.parameters, "filter_arguments_for_func", mocker.MagicMock(return_value={"a": 1}))

    configuration["command"] = "dummy"
    py_exec = tasks.PythonTaskType(**configuration)
    py_exec.execute_command()
    dummy_func.assert_called_once_with(a=1)


def test_python_task_command_sends_no_mapped_variable_if_not_present_in_signature(mocker, monkeypatch, configuration):
    dummy_func = mocker.MagicMock(return_value=None)

    class DummyModule:
        def __init__(self):
            self.func = dummy_func

    monkeypatch.setattr(tasks.utils, "get_module_and_attr_names", mocker.MagicMock(return_value=("idk", "func")))
    monkeypatch.setattr(tasks.importlib, "import_module", mocker.MagicMock(return_value=DummyModule()))
    monkeypatch.setattr(tasks.BaseTaskType, "output_to_file", mocker.MagicMock(return_value=contextlib.nullcontext()))
    monkeypatch.setattr(tasks.parameters, "filter_arguments_for_func", mocker.MagicMock(return_value={"a": 1}))

    configuration["command"] = "dummy"
    py_exec = tasks.PythonTaskType(**configuration)
    py_exec.execute_command(map_variable={"map_name": "map_value"})
    dummy_func.assert_called_once_with(a=1)


def test_python_task_command_sends_mapped_variable_if_present_in_signature(mocker, monkeypatch, configuration):
    dummy_func = mocker.MagicMock(return_value=None)

    class DummyModule:
        def __init__(self):
            self.func = dummy_func

    monkeypatch.setattr(tasks.utils, "get_module_and_attr_names", mocker.MagicMock(return_value=("idk", "func")))
    monkeypatch.setattr(tasks.importlib, "import_module", mocker.MagicMock(return_value=DummyModule()))
    monkeypatch.setattr(tasks.BaseTaskType, "output_to_file", mocker.MagicMock(return_value=contextlib.nullcontext()))
    monkeypatch.setattr(
        tasks.parameters, "filter_arguments_for_func", mocker.MagicMock(return_value={"a": 1, "map_name": "map_value"})
    )

    configuration["command"] = "dummy"
    py_exec = tasks.PythonTaskType(**configuration)
    py_exec.execute_command()
    dummy_func.assert_called_once_with(a=1, map_name="map_value")


def test_notebook_raises_exception_if_command_is_not_a_notebook():
    with pytest.raises(Exception):
        tasks.NotebookTaskType(command="path to notebook")


def test_notebook_raises_exception_if_ploomber_is_not_installed(mocker, monkeypatch):
    task_exec = tasks.NotebookTaskType(command="test.ipynb", node_name="dummy")

    with pytest.raises(Exception):
        task_exec.execute_command()
