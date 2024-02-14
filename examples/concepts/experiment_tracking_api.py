"""
A simple example of using experiment tracking service to track experiments.

You can run this pipeline by:
    python run examples/concepts/experiment_tracking_api.py
"""

from pydantic import BaseModel

from magnus import Pipeline, Task, track_this


class EggsModel(BaseModel):
    ham: str


def emit_metrics():
    """
    A function that populates experiment tracker with metrics.

    track_this can take many keyword arguments.
    Nested structures are supported by pydantic models.
    """
    track_this(spam="hello", eggs=EggsModel(ham="world"))
    track_this(answer=42.0)
    track_this(is_it_true=False)


def main():
    metrics = Task(
        name="Emit Metrics",
        command="examples.concepts.experiment_tracking_api.emit_metrics",
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
