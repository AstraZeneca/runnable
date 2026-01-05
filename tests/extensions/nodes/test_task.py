from datetime import datetime

from runnable import defaults
from runnable.datastore import StepAttempt, StepLog
from runnable.tasks import BaseTaskType


# --- Mock Task Implementation ---
class MockTask(BaseTaskType):
    task_type: str = "mock"
    command: str = "test_command"

    def execute_command(self, iter_variable=None):
        return StepAttempt(status=defaults.SUCCESS, start_time=str(datetime.now()))

    def get_summary(self):
        return {"type": self.task_type, "command": self.command}


# --- Test Cases ---
def test_task_node_basic_properties():
    """Test TaskNode basic properties and initialization"""
    task = MockTask()
    node = TaskNode(
        name="test_task",
        internal_name="test.task",
        executable=task,
        next_node="next_step",
    )
    assert node.name == "test_task"
    assert node.internal_name == "test.task"
    assert node.node_type == "task"
    assert node.next_node == "next_step"
    assert isinstance(node.executable, BaseTaskType)


def test_task_node_parse_from_config(mocker):
    """Test TaskNode configuration parsing"""
    config = {
        "name": "test_task",
        "internal_name": "test.task",
        "next": "next_step",
        "command_type": "mock",
        "command": "test_command",
        "next_node": "next_step",
    }
    mock_create = mocker.patch(
        "extensions.nodes.task.create_task", return_value=MockTask()
    )
    node = TaskNode.parse_from_config(config)
    assert node.name == "test_task"
    assert node.internal_name == "test.task"
    assert node.next_node == "next_step"
    assert isinstance(node.executable, BaseTaskType)
    mock_create.assert_called_once()


def test_task_node_get_summary(mocker):
    """Test TaskNode summary generation"""
    task = MockTask()
    node = TaskNode(
        name="test_task",
        internal_name="test.task",
        executable=task,
        next_node="next_step",
    )
    mocker.patch.object(TaskNode, "_get_catalog_settings", return_value={"foo": "bar"})
    summary = node.get_summary()
    assert summary["name"] == "test_task"
    assert summary["type"] == "task"
    assert summary["executable"]["type"] == "mock"
    assert summary["executable"]["command"] == "test_command"
    assert "catalog" in summary
    assert summary["catalog"] == {"foo": "bar"}


def test_task_node_execution(mocker):
    """Test TaskNode execution with real and mock context"""
    task = MockTask()
    node = TaskNode(
        name="test_task",
        internal_name="test.task",
        executable=task,
        next_node="next_step",
    )
    mock_context = mocker.Mock()
    mock_context.run_id = "test_run"
    mock_context.run_log_store.get_step_log.return_value = StepLog(
        name="test_step", internal_name="test.step", status=defaults.SUCCESS
    )
    mocker.patch.object(
        TaskNode,
        "_context",
        new_callable=mocker.PropertyMock,
        return_value=mock_context,
    )
    step_log = node.execute()
    assert step_log.status == defaults.SUCCESS
    mock_context.run_log_store.get_step_log.assert_called_once_with(
        "test.task", "test_run"
    )


def test_task_node_mock_execution(mocker):
    """Test TaskNode mock execution (no real command run)"""
    task = MockTask()
    node = TaskNode(
        name="test_task",
        internal_name="test.task",
        executable=task,
        next_node="next_step",
    )
    mock_context = mocker.Mock()
    mock_context.run_id = "test_run"
    mock_context.run_log_store.get_step_log.return_value = StepLog(
        name="test_step", internal_name="test.step", status=defaults.SUCCESS
    )
    mock_context.retry_indicator = ""
    mocker.patch.object(
        TaskNode,
        "_context",
        new_callable=mocker.PropertyMock,
        return_value=mock_context,
    )
    step_log = node.execute(mock=True)
    assert step_log.status == defaults.SUCCESS
    assert len(step_log.attempts) == 1
    assert step_log.attempts[0].status == defaults.SUCCESS


def test_task_node_with_map_variable(mocker):
    """Test TaskNode execution with map variable"""
    from runnable.defaults import IterableParameterModel, MapVariableModel

    task = MockTask()
    node = TaskNode(
        name="test_task",
        internal_name="test.task",
        executable=task,
        next_node="next_step",
    )
    # Create IterableParameterModel for testing
    iter_variable = IterableParameterModel(
        map_variable={"test_var": MapVariableModel(value='"test_value"')}
    )
    mock_context = mocker.Mock()
    mock_context.run_id = "test_run"
    mock_context.run_log_store.get_step_log.return_value = StepLog(
        name="test_step", internal_name="test.step", status=defaults.SUCCESS
    )
    mocker.patch.object(
        TaskNode,
        "_context",
        new_callable=mocker.PropertyMock,
        return_value=mock_context,
    )
    node.execute(iter_variable=iter_variable)
    mock_context.run_log_store.get_step_log.assert_called_once_with(
        node._get_step_log_name(iter_variable), "test_run"
    )


def test_task_node_with_attempt_number(mocker):
    """Test TaskNode execution with specific attempt number"""
    task = MockTask()
    node = TaskNode(
        name="test_task",
        internal_name="test.task",
        executable=task,
        next_node="next_step",
    )
    mock_context = mocker.Mock()
    mock_context.run_id = "test_run"
    mock_context.run_log_store.get_step_log.return_value = StepLog(
        name="test_step", internal_name="test.step", status=defaults.SUCCESS
    )
    mocker.patch.object(
        TaskNode,
        "_context",
        new_callable=mocker.PropertyMock,
        return_value=mock_context,
    )
    step_log = node.execute(attempt_number=2)
    assert len(step_log.attempts) == 1
    assert step_log.attempts[0].attempt_number == 2


from extensions.nodes.task import TaskNode
