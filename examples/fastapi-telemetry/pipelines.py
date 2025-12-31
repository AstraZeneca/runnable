"""Example pipelines for FastAPI integration."""

import time

from runnable import Pipeline, PythonTask, pickled


def compute(x: int = 10) -> int:
    """Simple compute function."""
    print(f"Computing: x={x}")
    time.sleep(1)  # Simulate work
    result = x * 2
    print(f"Computed: {result}")
    return result


def finalize(result: int) -> str:
    """Finalize the result."""
    print(f"Finalizing: result={result}")
    time.sleep(0.5)
    output = f"Final result: {result}"
    print(f"Finalized: {output}")
    return output


def build_example_pipeline() -> Pipeline:
    """Build an example pipeline."""
    return Pipeline(
        steps=[
            PythonTask(
                function=compute,
                name="compute",
                returns=[pickled("result")],
            ),
            PythonTask(
                function=finalize,
                name="finalize",
                returns=[pickled("final")],
            ),
        ]
    )


# Pipeline registry
PIPELINE_REGISTRY = {
    "example": {
        "builder": build_example_pipeline,
    },
}
