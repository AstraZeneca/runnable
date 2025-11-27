"""
Chapter 7: Running Anywhere

Demonstrate that the same pipeline code runs in different environments
without modification. The key insight: your code stays the same, only
the configuration changes.
"""

from runnable import Pipeline, PythonTask, pickled
from functions import load_data, preprocess_data, train_model, evaluate_model


def main():
    """The exact same pipeline from Chapter 4 - no code changes!"""
    print("=" * 50)
    print("Chapter 7: Running Anywhere")
    print("=" * 50)

    # This is the EXACT same pipeline from Chapter 4
    # No modifications needed to run in different environments
    pipeline = Pipeline(steps=[
        PythonTask(
            function=load_data,
            name="load_data",
            returns=[pickled("df")]
        ),
        PythonTask(
            function=preprocess_data,
            name="preprocess",
            returns=[pickled("preprocessed_data")]
        ),
        PythonTask(
            function=train_model,
            name="train",
            returns=[pickled("model_data")]
        ),
        PythonTask(
            function=evaluate_model,
            name="evaluate",
            returns=[pickled("evaluation_results")]
        )
    ])

    # Execute the pipeline
    # The environment is determined by configuration, not code
    pipeline.execute()

    print("\n" + "=" * 50)
    print("Running anywhere benefits:")
    print("- 💻 Same code runs on laptop, containers, or cloud")
    print("- 🔧 Environment controlled by configuration files")
    print("- 🚀 No code changes for different deployments")
    print("- 🎯 Develop locally, deploy anywhere")
    print("- 🔄 Easy migration between platforms")
    print("=" * 50)

    print("\n" + "=" * 50)
    print("Example: Run this pipeline in different ways:")
    print()
    print("1. Local execution (default):")
    print("   uv run examples/tutorials/getting-started/07_running_anywhere.py")
    print()
    print("2. With containers (if Docker available):")
    print("   uv run examples/tutorials/getting-started/07_running_anywhere.py \\")
    print("     --config examples/configs/local-container.yaml")
    print()
    print("3. With custom catalog location:")
    print("   uv run examples/tutorials/getting-started/07_running_anywhere.py \\")
    print("     --config examples/configs/custom-storage.yaml")
    print()
    print("Same code. Different environments. Zero changes.")
    print("=" * 50)

    return pipeline


if __name__ == "__main__":
    main()
