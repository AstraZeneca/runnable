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


def example_pipeline():
    """
    Build and return an example pipeline.

    This function is called by the SDK to get the pipeline definition.
    It must be a module-level function that returns a Pipeline when called.
    """
    pipeline = Pipeline(
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

    pipeline.execute()

    return pipeline


# Pipeline registry - each entry maps to a builder function
# The builder function must be importable and callable with no args
PIPELINE_REGISTRY = {
    "example": {
        "module": "pipelines",
        "function": "example_pipeline",
    },
}
