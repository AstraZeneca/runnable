from unittest.mock import Mock

from runnable import defaults
from runnable.datastore import StepLog


def make_mock_context():
    mock_context = Mock()
    mock_context.run_id = "test_run"
    mock_context.run_log_store.get_step_log.return_value = StepLog(
        name="stub_step", internal_name="stub.step", status="SUCCESS"
    )
    return mock_context


def test_stub_node_basic_properties():
    node = StubNode(
        name="stub",
        internal_name="stub.step",
        next_node="success",
    )
    assert node.name == "stub"
    assert node.internal_name == "stub.step"
    assert node.node_type == "stub"
    assert node.next_node == "success"


def test_stub_node_parse_from_config():
    config = {"name": "stub", "internal_name": "stub.step", "next_node": "success"}
    node = StubNode.parse_from_config(config)
    assert node.name == "stub"
    assert node.internal_name == "stub.step"
    assert node.node_type == "stub"
    assert node.next_node == "success"


def test_stub_node_get_summary():
    node = StubNode(
        name="stub",
        internal_name="stub.step",
        next_node="success",
    )
    summary = node.get_summary()
    assert summary["name"] == "stub"
    assert summary["type"] == "stub"


def test_stub_node_execute(mocker):
    node = StubNode(
        name="stub",
        internal_name="stub.step",
        next_node="success",
    )
    mock_context = make_mock_context()
    mocker.patch.object(
        StubNode,
        "_context",
        new_callable=mocker.PropertyMock,
        return_value=mock_context,
    )
    step_log = node.execute()
    assert step_log.status == defaults.SUCCESS
    assert len(step_log.attempts) == 1
    assert step_log.attempts[0].status == defaults.SUCCESS


def test_stub_node_execute_with_attempt_number(mocker):
    node = StubNode(
        name="stub",
        internal_name="stub.step",
        next_node="success",
    )
    mock_context = make_mock_context()
    mocker.patch.object(
        StubNode,
        "_context",
        new_callable=mocker.PropertyMock,
        return_value=mock_context,
    )
    step_log = node.execute(attempt_number=7)
    assert len(step_log.attempts) == 1
    assert step_log.attempts[0].attempt_number == 7


from extensions.nodes.stub import StubNode
