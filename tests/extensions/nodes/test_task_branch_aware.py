from datetime import datetime
from unittest.mock import Mock, patch

from runnable import defaults
from runnable.datastore import StepAttempt
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
def test_parse_from_config_passes_internal_branch_name(mocker):
    """Test TaskNode.parse_from_config passes internal_branch_name to task."""

    config = {
        "name": "test_task",
        "internal_name": "test.task",
        "next_node": "next_step",
        "command_type": "python",
        "command": "test_function"
    }

    # Mock create_task to capture the task_config
    mock_create = mocker.patch(
        'extensions.nodes.task.create_task', return_value=MockTask()
    )

    # Create TaskNode - currently does not pass internal_branch_name
    task_node = TaskNode.parse_from_config(config)

    # Verify create_task was called
    mock_create.assert_called_once()
    called_config = mock_create.call_args[0][0]

    # The config should not yet include internal_branch_name
    # (this test will guide us to add it)
    assert "internal_branch_name" not in called_config


def test_task_node_inherits_internal_branch_name_from_base_node(mocker):
    """Test TaskNode has internal_branch_name from BaseNode."""

    config = {
        "name": "test_task",
        "internal_name": "test.task",
        "next_node": "next_step",
        "command_type": "python",
        "command": "test_function"
    }

    mocker.patch(
        'extensions.nodes.task.create_task', return_value=MockTask()
    )

    task_node = TaskNode.parse_from_config(config)

    # TaskNode should have internal_branch_name attribute (inherited from BaseNode)
    assert hasattr(task_node, 'internal_branch_name')
    assert task_node.internal_branch_name == ""  # Default value from BaseNode


def test_parse_from_config_includes_internal_branch_name_in_task_config(mocker):
    """Test parse_from_config passes internal_branch_name to create_task."""

    config = {
        "name": "test_task",
        "internal_name": "test.task",
        "next_node": "next_step",
        "command_type": "python",
        "command": "test_function",
        "internal_branch_name": "map.iteration_1"
    }

    # Mock create_task to capture the task_config
    mock_create = mocker.patch(
        'extensions.nodes.task.create_task', return_value=MockTask()
    )

    # Create TaskNode with internal_branch_name in config
    task_node = TaskNode.parse_from_config(config)

    # Verify create_task was called
    mock_create.assert_called_once()
    called_config = mock_create.call_args[0][0]

    # The task config should include internal_branch_name
    assert "internal_branch_name" in called_config
    assert called_config["internal_branch_name"] == "map.iteration_1"


# Import at the end to avoid circular imports
from extensions.nodes.task import TaskNode
