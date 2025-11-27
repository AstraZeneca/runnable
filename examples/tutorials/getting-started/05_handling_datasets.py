"""
Chapter 5: Handling Large Datasets

Show how to use Catalog for efficient file storage instead of keeping
everything in memory. This is crucial for large datasets and sharing
intermediate results.
"""

import pandas as pd
from runnable import Pipeline, PythonTask, Catalog, pickled
from functions import load_data, preprocess_data, train_model, evaluate_model


def load_data_to_file(data_path="data.csv"):
    """Load data and save to file for catalog management."""
    df = load_data(data_path)
    df.to_csv("dataset.csv", index=False)
    print(f"Dataset saved: {len(df)} rows")
    return {"rows": len(df), "columns": len(df.columns)}


def preprocess_from_file(test_size=0.2, random_state=42):
    """Load data from file, preprocess, and save results to files."""
    df = pd.read_csv("dataset.csv")
    print(f"Loaded dataset: {len(df)} rows")

    preprocessed = preprocess_data(df, test_size=test_size, random_state=random_state)

    # Save preprocessed data to files instead of passing in memory
    preprocessed['X_train'].to_csv("X_train.csv", index=False)
    preprocessed['X_test'].to_csv("X_test.csv", index=False)
    preprocessed['y_train'].to_csv("y_train.csv", index=False)
    preprocessed['y_test'].to_csv("y_test.csv", index=False)

    print(f"Preprocessed data saved: {len(preprocessed['X_train'])} training samples")

    return {
        "train_samples": len(preprocessed['X_train']),
        "test_samples": len(preprocessed['X_test'])
    }


def train_from_files(n_estimators=100, random_state=42):
    """Load preprocessed data from files and train model."""
    # Load preprocessed data from files
    X_train = pd.read_csv("X_train.csv")
    y_train = pd.read_csv("y_train.csv")['target']

    print(f"Training on {len(X_train)} samples")

    preprocessed_data = {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': None,  # Not needed for training
        'y_test': None
    }

    model_data = train_model(preprocessed_data, n_estimators=n_estimators, random_state=random_state)

    # Return model for pickle-based passing (models are usually small enough)
    return model_data


def evaluate_from_files(model_data):
    """Load test data from files and evaluate model."""
    # Load test data from files
    X_test = pd.read_csv("X_test.csv")
    y_test = pd.read_csv("y_test.csv")['target']

    print(f"Evaluating on {len(X_test)} test samples")

    preprocessed_data = {
        'X_train': None,  # Not needed for evaluation
        'y_train': None,
        'X_test': X_test,
        'y_test': y_test
    }

    results = evaluate_model(model_data, preprocessed_data)
    return results


def main():
    """Demonstrate file-based data management with Catalog."""
    print("=" * 50)
    print("Chapter 5: Handling Large Datasets")
    print("=" * 50)

    pipeline = Pipeline(steps=[
        # Load data and store the dataset file
        PythonTask(
            function=load_data_to_file,
            name="load_data",
            catalog=Catalog(put=["dataset.csv"]),
            returns=[pickled("dataset_info")]
        ),
        # Preprocess and store all intermediate files
        PythonTask(
            function=preprocess_from_file,
            name="preprocess",
            catalog=Catalog(
                get=["dataset.csv"],  # Get the dataset
                put=["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]  # Store results
            ),
            returns=[pickled("preprocess_info")]
        ),
        # Train model using files
        PythonTask(
            function=train_from_files,
            name="train",
            catalog=Catalog(get=["X_train.csv", "y_train.csv"]),
            returns=[pickled("model_data")]
        ),
        # Evaluate using files
        PythonTask(
            function=evaluate_from_files,
            name="evaluate",
            catalog=Catalog(get=["X_test.csv", "y_test.csv"]),
            returns=[pickled("evaluation_results")]
        )
    ])

    pipeline.execute()

    print("\n" + "=" * 50)
    print("File-based catalog benefits:")
    print("- 💾 Handles datasets larger than memory")
    print("- 🔄 Intermediate files automatically managed")
    print("- 📦 Files stored safely in .runnable/")
    print("- 🚀 Can resume without reloading large data")
    print("- 🤝 Files can be shared across runs")
    print("=" * 50)

    return pipeline


if __name__ == "__main__":
    main()
