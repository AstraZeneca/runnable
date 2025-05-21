from unittest.mock import Mock

import pytest

from runnable import defaults
from runnable.datastore import StepLog


def make_mock_context():
    mock_context = Mock()
    mock_context.run_id = "test_run"
    # Step log
    mock_context.run_log_store.get_step_log.return_value = StepLog(
        name="success_step", internal_name="success.step", status="SUCCESS"
    )
    # Branch log
    branch_log = Mock()
    branch_log.status = None
    mock_context.run_log_store.get_branch_log.return_value = branch_log
    mock_context.run_log_store.add_branch_log.return_value = None
    return mock_context


def test_success_node_basic_properties():
    node = SuccessNode(
        name="success",
        internal_name="success.step",
    )
    assert node.name == "success"
    assert node.internal_name == "success.step"
    assert node.node_type == "success"


def test_success_node_parse_from_config():
    config = {"name": "success", "internal_name": "success.step"}
    node = SuccessNode.parse_from_config(config)
    assert node.name == "success"
    assert node.internal_name == "success.step"
    assert node.node_type == "success"


def test_success_node_get_summary():
    node = SuccessNode(
        name="success",
        internal_name="success.step",
    )
    summary = node.get_summary()
    assert summary["name"] == "success"
    assert summary["type"] == "success"


def test_success_node_execute(mocker):
    node = SuccessNode(
        name="success",
        internal_name="success.step",
    )
    mock_context = make_mock_context()
    mocker.patch.object(
        SuccessNode,
        "_context",
        new_callable=mocker.PropertyMock,
        return_value=mock_context,
    )
    step_log = node.execute()
    # Step log status should be set to SUCCESS
    assert step_log.status == defaults.SUCCESS
    # There should be one attempt, and its status should be SUCCESS
    assert len(step_log.attempts) == 1
    assert step_log.attempts[0].status == defaults.SUCCESS
    # Branch log status should be set to SUCCESS
    branch_log = mock_context.run_log_store.get_branch_log.return_value
    assert branch_log.status == defaults.SUCCESS
    # add_branch_log should be called
    mock_context.run_log_store.add_branch_log.assert_called_once_with(
        branch_log, "test_run"
    )


def test_success_node_execute_with_attempt_number(mocker):
    node = SuccessNode(
        name="success",
        internal_name="success.step",
    )
    mock_context = make_mock_context()
    mocker.patch.object(
        SuccessNode,
        "_context",
        new_callable=mocker.PropertyMock,
        return_value=mock_context,
    )
    step_log = node.execute(attempt_number=3)
    assert len(step_log.attempts) == 1
    assert step_log.attempts[0].attempt_number == 3


from extensions.nodes.success import SuccessNode
