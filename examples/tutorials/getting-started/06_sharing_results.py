"""
Chapter 6: Sharing Results

Show how to store and retrieve model artifacts and metrics that persist
across runs. This enables sharing models with teammates and tracking
performance over time.
"""

import pickle
import json
from pathlib import Path
from runnable import Pipeline, PythonTask, Catalog, pickled, metric
from functions import load_data, preprocess_data, train_model, evaluate_model


def save_model_artifact(model_data):
    """Save trained model to a file for sharing."""
    # Save model to a file that will be stored in catalog
    with open("trained_model.pkl", "wb") as f:
        pickle.dump(model_data, f)

    print(f"Model saved: trained_model.pkl")
    print(f"Feature count: {len(model_data['feature_names'])}")

    return model_data


def save_evaluation_metrics(evaluation_results):
    """Save evaluation metrics to files for tracking and sharing."""
    # Extract key metrics
    accuracy = evaluation_results['accuracy']
    report = evaluation_results['classification_report']

    # Save detailed report as JSON
    with open("evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # Save a summary for quick viewing
    summary = {
        "accuracy": accuracy,
        "precision": report['weighted avg']['precision'],
        "recall": report['weighted avg']['recall'],
        "f1-score": report['weighted avg']['f1-score']
    }

    with open("metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Metrics saved:")
    print(f"  - Accuracy: {accuracy:.4f}")
    print(f"  - Precision: {summary['precision']:.4f}")
    print(f"  - Recall: {summary['recall']:.4f}")
    print(f"  - F1-Score: {summary['f1-score']:.4f}")

    # Return metrics for tracking (as tuple to match returns specification)
    return (
        accuracy,
        summary['precision'],
        summary['recall'],
        summary['f1-score']
    )


def load_and_verify_model():
    """Demonstrate loading a previously saved model."""
    # Load the saved model
    with open("trained_model.pkl", "rb") as f:
        model_data = pickle.load(f)

    print(f"Model loaded successfully!")
    print(f"Model type: {type(model_data['model']).__name__}")
    print(f"Feature count: {len(model_data['feature_names'])}")

    return {"model_verified": True}


def main():
    """Demonstrate persistent storage of models and metrics."""
    print("=" * 50)
    print("Chapter 6: Sharing Results")
    print("=" * 50)

    pipeline = Pipeline(steps=[
        # Load and preprocess data
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
        # Train model
        PythonTask(
            function=train_model,
            name="train",
            returns=[pickled("model_data")]
        ),
        # Save model artifact to catalog
        PythonTask(
            function=save_model_artifact,
            name="save_model",
            catalog=Catalog(put=["trained_model.pkl"]),  # Store model file
            returns=[pickled("model_data")]
        ),
        # Evaluate and save metrics
        PythonTask(
            function=evaluate_model,
            name="evaluate",
            returns=[pickled("evaluation_results")]
        ),
        PythonTask(
            function=save_evaluation_metrics,
            name="save_metrics",
            catalog=Catalog(put=["evaluation_report.json", "metrics_summary.json"]),
            returns=[
                metric("accuracy"),
                metric("precision"),
                metric("recall"),
                metric("f1_score")
            ]
        ),
        # Verify model can be loaded (simulating another run or teammate)
        PythonTask(
            function=load_and_verify_model,
            name="verify_model",
            catalog=Catalog(get=["trained_model.pkl"]),  # Get the saved model
            returns=[pickled("verification")]
        )
    ])

    pipeline.execute()

    print("\n" + "=" * 50)
    print("Persistent storage benefits:")
    print("- 📦 Model artifacts saved and shareable")
    print("- 📊 Metrics tracked across runs")
    print("- 🤝 Teammates can reuse your trained models")
    print("- 📈 Performance history for comparison")
    print("- 🔍 All results stored in .catalog/")
    print("=" * 50)

    return pipeline


if __name__ == "__main__":
    main()
