"""
A simple example of using experiment tracking service to track experiments.
In this example, we integrate with mlflow as our experiment tracking service.

The mlflow server is expected to be running at: http://127.0.0.1:8080

You can run this pipeline by:
    python run examples/concepts/experiment_tracking_integration.py
"""

from pydantic import BaseModel

from runnable import Pipeline, Task, track_this


class EggsModel(BaseModel):
    ham: str


def emit_metrics():
    """
    A function that populates experiment tracker with metrics.

    track_this can take many keyword arguments.
    Nested structures are supported by pydantic models.
    """
    track_this(spam="hello", eggs=EggsModel(ham="world"))
    track_this(is_it_true=False)

    track_this(answer=0.0)
    track_this(step=1, answer=20.0)
    track_this(step=2, answer=40.0)
    track_this(step=3, answer=60.0)


def main():
    metrics = Task(
        name="Emit Metrics",
        command="examples.concepts.experiment_tracking_integration.emit_metrics",
        terminate_with_success=True,
    )

    pipeline = Pipeline(
        steps=[metrics],
        start_at=metrics,
        add_terminal_nodes=True,
    )

    pipeline.execute(configuration_file="examples/configs/mlflow-config.yaml", tag="demo-magnus")


if __name__ == "__main__":
    main()
