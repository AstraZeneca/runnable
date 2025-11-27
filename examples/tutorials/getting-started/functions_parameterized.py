"""
Parameterized versions of ML functions for Chapter 3.

These functions accept parameters instead of having hardcoded values,
making them flexible for different experiments.
"""

from functions import load_data, preprocess_data, train_model, evaluate_model, save_model, save_results


def train_ml_model_flexible(
    data_path="data.csv",
    test_size=0.2,
    n_estimators=100,
    random_state=42,
    model_path="model.pkl",
    results_path="results.json"
):
    """
    Flexible ML training function that accepts parameters.

    Same logic as train_ml_model_basic, but now configurable!
    """
    print("Loading data...")
    df = load_data(data_path)

    print("Preprocessing...")
    preprocessed = preprocess_data(df, test_size=test_size, random_state=random_state)

    print(f"Training model with {n_estimators} estimators...")
    model_data = train_model(preprocessed, n_estimators=n_estimators, random_state=random_state)

    print("Evaluating...")
    results = evaluate_model(model_data, preprocessed)

    print(f"Accuracy: {results['accuracy']:.4f}")

    # Save with custom paths
    save_model(model_data, model_path)
    save_results(results, results_path)

    return results
