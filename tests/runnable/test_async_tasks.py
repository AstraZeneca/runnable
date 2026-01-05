import pytest

from runnable.tasks import AsyncPythonTaskType


def test_async_python_task_type_initialization():
    """Test AsyncPythonTaskType can be instantiated with command."""
    task = AsyncPythonTaskType(command="examples.common.functions.hello")
    assert task.task_type == "async-python"
    assert task.command == "examples.common.functions.hello"


def test_async_python_task_type_sync_raises():
    """Test AsyncPythonTaskType.execute_command raises RuntimeError."""
    task = AsyncPythonTaskType(command="test.func")
    with pytest.raises(RuntimeError, match="requires async execution"):
        task.execute_command()


def test_async_python_task_sdk_initialization():
    """Test AsyncPythonTask SDK class can be instantiated."""
    from examples.common.functions import async_hello
    from runnable import AsyncPythonTask

    task = AsyncPythonTask(name="async_task", function=async_hello, returns=["result"])
    assert task.command_type == "async-python"
    assert task.command == "examples.common.functions.async_hello"


def test_async_pipeline_creation():
    """Test AsyncPipeline can be created with async tasks."""
    from examples.common.functions import async_hello
    from runnable import AsyncPipeline, AsyncPythonTask

    pipeline = AsyncPipeline(
        steps=[
            AsyncPythonTask(
                name="async_task", function=async_hello, returns=["result"]
            ),
        ],
        name="test_async_pipeline",
    )
    assert pipeline.name == "test_async_pipeline"
    assert len(pipeline.steps) == 1
