"""
Simple local test to verify telemetry is working.

Run with:
    uv run python examples/telemetry-local/simple_telemetry_test.py

You should see colored/indented spans showing:
- pipeline span wrapping the execution
- task spans for each step with inputs/outputs
"""

from runnable import Pipeline, PythonTask, pickled


def step_one(x: int = 5) -> int:
    """First step - doubles the input."""
    print(f"Step one: received x={x}")
    result = x * 2
    print(f"Step one: returning {result}")
    return result


def step_two(doubled: int) -> str:
    """Second step - formats the result."""
    print(f"Step two: received doubled={doubled}")
    result = f"Final result: {doubled}"
    print(f"Step two: returning '{result}'")
    return result


def main():
    """Run a simple pipeline and observe telemetry output."""
    pipeline = Pipeline(
        steps=[
            PythonTask(
                function=step_one,
                name="step_one",
                returns=[pickled("doubled")],
            ),
            PythonTask(
                function=step_two,
                name="step_two",
                returns=[pickled("final_result")],
            ),
        ]
    )

    # Execute the pipeline
    pipeline.execute()

    return pipeline


if __name__ == "__main__":
    import logfire

    # Configure logfire to output to console (no cloud sending)
    # Must be done before main() to catch all spans
    logfire.configure(
        send_to_logfire=False,
        console=logfire.ConsoleOptions(
            colors="auto",
            span_style="indented",
            include_timestamps=True,
            verbose=True,
        ),
    )

    print("=" * 60)
    print("Running pipeline with telemetry enabled")
    print("You should see spans for pipeline and each task")
    print("=" * 60)
    print()

    main()

    print()
    print("=" * 60)
    print("Pipeline completed!")
    print("=" * 60)
