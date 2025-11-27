"""
Chapter 4: Connecting the Workflow

Break the monolithic function into a proper multi-step ML pipeline
with automatic data flow between steps.
"""

from runnable import Pipeline, PythonTask, pickled
from functions import load_data, preprocess_data, train_model, evaluate_model


def main():
    """Transform monolithic function into multi-step pipeline."""
    print("=" * 50)
    print("Chapter 4: Connecting the Workflow")
    print("=" * 50)

    # Same functions, now as separate pipeline steps
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

    pipeline.execute()

    print("\n" + "=" * 50)
    print("Pipeline benefits:")
    print("- 🔗 Automatic data flow between steps")
    print("- ⚡ Can resume from any failed step")
    print("- 📊 Individual step tracking and timing")
    print("- 🔍 Intermediate results preserved")
    print("- 🎯 Better debugging and development")
    print("=" * 50)

    return pipeline

if __name__ == "__main__":
    main()
