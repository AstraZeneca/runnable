import json
import os

from pydantic import BaseModel

from magnus import Pipeline, Task


class EggsModel(BaseModel):
    ham: str


def emit_metrics():
    """
    A function that populates environment variables with metrics.

    Any environment variable with prefix "MAGNUS_TRACK_" will be
    understood as a metric.

    Numeric metrics can be set as strings but would be stored to
    int/float. Boolean metrics are not supported.
    """
    os.environ["MAGNUS_TRACK_spam"] = "hello"
    os.environ["MAGNUS_TRACK_eggs"] = json.dumps(
        EggsModel(ham="world").model_dump(by_alias=True),
    )
    os.environ["MAGNUS_TRACK_answer"] = "42.0"  # Would be stored as float(42)


def main():
    metrics = Task(
        name="Emit Metrics",
        command="examples.concepts.experiment_tracking_env.emit_metrics",
        terminate_with_success=True,
    )

    pipeline = Pipeline(
        steps=[metrics],
        start_at=metrics,
        add_terminal_nodes=True,
    )

    pipeline.execute()  # (1)


if __name__ == "__main__":
    main()
