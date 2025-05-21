from unittest.mock import Mock

import pytest

from runnable import defaults
from runnable.datastore import StepLog


def make_mock_context():
    mock_context = Mock()
    mock_context.run_id = "test_run"
    # Step log
    mock_context.run_log_store.get_step_log.return_value = StepLog(
        name="fail_step", internal_name="fail.step", status="FAIL"
    )
    # Branch log
    branch_log = Mock()
    branch_log.status = None
    mock_context.run_log_store.get_branch_log.return_value = branch_log
    mock_context.run_log_store.add_branch_log.return_value = None
    return mock_context


def test_fail_node_basic_properties():
    node = FailNode(
        name="fail",
        internal_name="fail.step",
    )
    assert node.name == "fail"
    assert node.internal_name == "fail.step"
    assert node.node_type == "fail"


def test_fail_node_parse_from_config():
    config = {"name": "fail", "internal_name": "fail.step"}
    node = FailNode.parse_from_config(config)
    assert node.name == "fail"
    assert node.internal_name == "fail.step"
    assert node.node_type == "fail"


def test_fail_node_get_summary():
    node = FailNode(
        name="fail",
        internal_name="fail.step",
    )
    summary = node.get_summary()
    assert summary["name"] == "fail"
    assert summary["type"] == "fail"


def test_fail_node_execute(mocker):
    node = FailNode(
        name="fail",
        internal_name="fail.step",
    )
    mock_context = make_mock_context()
    mocker.patch.object(
        FailNode,
        "_context",
        new_callable=mocker.PropertyMock,
        return_value=mock_context,
    )
    step_log = node.execute()
    # Step log status should be set to SUCCESS (as per implementation)
    assert step_log.status == defaults.SUCCESS
    # There should be one attempt, and its status should be SUCCESS
    assert len(step_log.attempts) == 1
    assert step_log.attempts[0].status == defaults.SUCCESS
    # Branch log status should be set to FAIL
    branch_log = mock_context.run_log_store.get_branch_log.return_value
    assert branch_log.status == defaults.FAIL
    # add_branch_log should be called
    mock_context.run_log_store.add_branch_log.assert_called_once_with(
        branch_log, "test_run"
    )


def test_fail_node_execute_with_attempt_number(mocker):
    node = FailNode(
        name="fail",
        internal_name="fail.step",
    )
    mock_context = make_mock_context()
    mocker.patch.object(
        FailNode,
        "_context",
        new_callable=mocker.PropertyMock,
        return_value=mock_context,
    )
    step_log = node.execute(attempt_number=5)
    assert len(step_log.attempts) == 1
    assert step_log.attempts[0].attempt_number == 5


from extensions.nodes.fail import FailNode
