from extensions.nodes import fail, stub, success
from runnable import sdk


def test_success_init():
    test_success = sdk.Success()

    assert test_success.name == "success"
    assert test_success.create_node() == success.SuccessNode(
        name="success", internal_name="success"
    )


def test_fail_init():
    test_fail = sdk.Fail()

    assert test_fail.name == "fail"
    assert test_fail.create_node() == fail.FailNode(name="fail", internal_name="fail")


def test_stub_node_makes_next_success_if_terminate_with_success():
    test_stub = sdk.Stub(name="stub", terminate_with_success=True)

    assert test_stub.name == "stub"
    assert test_stub.create_node() == stub.StubNode(
        name="stub", next_node="success", internal_name="stub"
    )


def test_stub_node_takes_given_next_node():
    test_stub = sdk.Stub(name="stub", next_node="test")

    assert test_stub.create_node() == stub.StubNode(
        name="stub", next_node="test", internal_name="stub"
    )
