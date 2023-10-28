from rich import print

from magnus import Stub, Pipeline, Task, defaults, entrypoints, utils


def test_sequence():
    first = Stub(name="first", next="second")
    second = Stub(name="second", terminate_with_success=True)

    pipeline = Pipeline(steps=[first, second], start_at="first", add_terminal_nodes=True)

    run_log = pipeline.execute()

    assert len(run_log.steps) == 3
