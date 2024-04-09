import pytest


from runnable import tasks


@pytest.fixture
def configuration():
    return {"node_name": "dummy", "task_type": "dummy"}


def test_base_task_execute_command_raises_not_implemented_error(configuration):
    base_execution_type = tasks.BaseTaskType(**configuration)

    with pytest.raises(NotImplementedError):
        base_execution_type.execute_command()


def test_notebook_raises_exception_if_command_is_not_a_notebook():
    with pytest.raises(Exception):
        tasks.NotebookTaskType(command="path to notebook")
