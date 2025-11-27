"""
Core ML functions for the getting started tutorial.

These functions represent a realistic ML workflow that progressively
gets wrapped with Runnable patterns throughout the tutorial.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import json
import os
from pathlib import Path


def create_sample_dataset(n_samples=1000, n_features=20, random_state=42):
    """Create a sample classification dataset."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        random_state=random_state
    )

    # Convert to DataFrame for more realistic data handling
    feature_names = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y

    return df


def load_data(data_path="data.csv"):
    """Load dataset from file or create if doesn't exist."""
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    else:
        # Create sample data if file doesn't exist
        df = create_sample_dataset()
        df.to_csv(data_path, index=False)
        return df


def preprocess_data(df, test_size=0.2, random_state=42):
    """Preprocess data for training."""
    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Simple preprocessing - could be much more complex in real scenarios
    # For now, just ensure no missing values
    X_train = X_train.fillna(X_train.mean())
    X_test = X_test.fillna(X_train.mean())  # Use training means

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }


def train_model(preprocessed_data, n_estimators=100, random_state=42):
    """Train a Random Forest model."""
    X_train = preprocessed_data['X_train']
    y_train = preprocessed_data['y_train']

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state
    )

    model.fit(X_train, y_train)

    return {
        'model': model,
        'feature_names': list(X_train.columns)
    }


def evaluate_model(model_data, preprocessed_data):
    """Evaluate the trained model."""
    model = model_data['model']
    X_test = preprocessed_data['X_test']
    y_test = preprocessed_data['y_test']

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    return {
        'accuracy': accuracy,
        'classification_report': report,
        'predictions': y_pred.tolist(),
        'probabilities': y_pred_proba.tolist()
    }


def save_model(model_data, file_path="model.pkl"):
    """Save trained model to file."""
    with open(file_path, 'wb') as f:
        pickle.dump(model_data, f)
    return file_path


def save_results(evaluation_results, file_path="results.json"):
    """Save evaluation results to file."""
    with open(file_path, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    return file_path


# The "starting point" function that combines everything
def train_ml_model_basic():
    """
    Basic ML training function - works locally but has typical problems:
    - Hardcoded parameters
    - No tracking of runs
    - Results get overwritten
    - No reproducibility guarantees
    """
    print("Loading data...")
    df = load_data("data.csv")

    print("Preprocessing...")
    preprocessed = preprocess_data(df, test_size=0.2, random_state=42)

    print("Training model...")
    model_data = train_model(preprocessed, n_estimators=100, random_state=42)

    print("Evaluating...")
    results = evaluate_model(model_data, preprocessed)

    print(f"Accuracy: {results['accuracy']:.4f}")

    # Save everything (gets overwritten each run!)
    save_model(model_data, "model.pkl")
    save_results(results, "results.json")

    return results


if __name__ == "__main__":
    # This is the "before Runnable" version
    results = train_ml_model_basic()
    print("Done! Check model.pkl and results.json")
