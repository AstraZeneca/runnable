"""Example pipelines for FastAPI integration.

Each pipeline builder returns a Pipeline object (does NOT call execute).
The FastAPI endpoint handles execution.
"""

import time
from functools import partial
from typing import List

from pydantic import BaseModel

from runnable import (
    Map,
    Parallel,
    Pipeline,
    PythonTask,
    ShellTask,
    Stub,
    metric,
    pickled,
)

# =============================================================================
# Pipeline: example - Simple two-step compute pipeline
# =============================================================================


def compute(x: int = 10) -> int:
    """Simple compute function."""
    print(f"Computing: x={x}")
    time.sleep(1)
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


def example_pipeline() -> Pipeline:
    """Simple two-step pipeline: compute -> finalize."""
    pipeline = Pipeline(
        steps=[
            PythonTask(function=compute, name="compute", returns=["result"]),
            PythonTask(function=finalize, name="finalize", returns=["final"]),
        ]
    )
    pipeline.execute()
    return pipeline


# =============================================================================
# Pipeline: hello - Simplest possible pipeline
# =============================================================================


def hello():
    """The most basic function."""
    print("Hello World!")


def hello_pipeline() -> Pipeline:
    """Single step hello world pipeline."""
    pipeline = Pipeline(steps=[PythonTask(name="hello", function=hello)])
    pipeline.execute()
    return pipeline


# =============================================================================
# Pipeline: parameters - Demonstrates parameter passing
# =============================================================================


class ModelConfig(BaseModel):
    learning_rate: float
    epochs: int


def generate_data(rows: int = 100):
    """Generate training data."""
    print(f"Generating {rows} rows of data...")
    time.sleep(0.5)
    data = [[i, i * 2] for i in range(rows)]
    config = ModelConfig(learning_rate=0.01, epochs=10)
    accuracy = 0.95
    print(f"Generated data with config: {config}")
    return data, config, accuracy


def train_model(data: list, config: ModelConfig, accuracy: float):
    """Train model with data and config."""
    print(f"Training with {len(data)} samples")
    print(f"Config: lr={config.learning_rate}, epochs={config.epochs}")
    time.sleep(1)
    final_accuracy = accuracy + 0.02
    print(f"Training complete. Accuracy: {final_accuracy}")
    return final_accuracy


def parameters_pipeline() -> Pipeline:
    """Pipeline demonstrating parameter passing between tasks."""
    pipeline = Pipeline(
        steps=[
            PythonTask(
                function=generate_data,
                name="generate_data",
                returns=[pickled("data"), "config", metric("accuracy")],
            ),
            PythonTask(
                function=train_model,
                name="train_model",
                returns=[metric("final_accuracy")],
            ),
        ]
    )
    pipeline.execute()
    return pipeline


# =============================================================================
# Pipeline: parallel - Demonstrates parallel execution
# =============================================================================


def branch_a_work():
    """Work done in branch A."""
    print("Branch A: Starting work...")
    time.sleep(1)
    print("Branch A: Complete!")


def branch_b_work():
    """Work done in branch B."""
    print("Branch B: Starting work...")
    time.sleep(0.8)
    print("Branch B: Complete!")


def aggregate():
    """Aggregate results from branches."""
    print("Aggregating results from parallel branches...")
    time.sleep(0.3)
    print("Aggregation complete!")


def parallel_pipeline() -> Pipeline:
    """Pipeline with parallel branches."""
    branch_a = Pipeline(
        steps=[PythonTask(function=branch_a_work, name="branch_a_work")]
    )
    branch_b = Pipeline(
        steps=[PythonTask(function=branch_b_work, name="branch_b_work")]
    )

    pipeline = Pipeline(
        steps=[
            Parallel(
                name="parallel_step",
                branches={"branch_a": branch_a, "branch_b": branch_b},
            ),
            PythonTask(function=aggregate, name="aggregate"),
        ]
    )
    pipeline.execute()
    return pipeline


# =============================================================================
# Pipeline: map - Demonstrates map/reduce pattern
# =============================================================================


def process_item(item: int) -> int:
    """Process a single item."""
    print(f"Processing item: {item}")
    time.sleep(0.3)
    result = item * 10
    print(f"Processed: {item} -> {result}")
    return result


def reduce_results(processed: List[int], items: List[int]):
    """Reduce all processed results."""
    print(f"Reducing {len(processed)} results...")
    total = sum(processed)
    print(f"Items: {items}")
    print(f"Processed: {processed}")
    print(f"Total: {total}")


def map_pipeline() -> Pipeline:
    """Pipeline demonstrating map/reduce over items."""

    pipeline = Pipeline(
        steps=[
            Map(
                name="process_items",
                iterate_on="items",
                iterate_as="item",
                branch=Pipeline(
                    steps=[
                        PythonTask(
                            function=process_item,
                            name="process_item",
                            returns=["processed"],
                        )
                    ]
                ),
            ),
            PythonTask(function=reduce_results, name="reduce"),
        ]
    )
    pipeline.execute()
    return pipeline


# =============================================================================
# Pipeline: shell - Mix of Python and Shell tasks
# =============================================================================


def prepare_env():
    """Prepare environment."""
    print("Preparing environment...")
    time.sleep(0.3)
    return "ready"


def shell_pipeline() -> Pipeline:
    """Pipeline with mix of Python and Shell tasks."""
    pipeline = Pipeline(
        steps=[
            PythonTask(function=prepare_env, name="prepare", returns=["status"]),
            ShellTask(
                name="shell_echo",
                command="echo 'Status is: '$status && echo 'Shell task complete!'",
            ),
            Stub(name="cleanup"),
        ]
    )
    pipeline.execute()
    return pipeline


# =============================================================================
# Pipeline Registry
# =============================================================================

PIPELINE_REGISTRY = {
    "example": partial(example_pipeline),
    "hello": partial(hello_pipeline),
    "parameters": partial(parameters_pipeline),
    "parallel": partial(parallel_pipeline),
    "map": partial(map_pipeline),
    "shell": partial(shell_pipeline),
}
