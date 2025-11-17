"""
Data Science Pipeline 101 - Core Functions

This module contains all the functions used in our data science pipeline tutorial.
Each function represents a step in a typical ML workflow: load → explore → preprocess → train → evaluate.
"""

import json
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(dataset_name: str = "wine"):
    """
    Load a dataset for our ML pipeline.

    Args:
        dataset_name: Name of the dataset to load (currently supports 'wine')

    Returns:
        Tuple of (features_df, target_series)
    """
    logger.info(f"Loading {dataset_name} dataset...")

    if dataset_name == "wine":
        data = load_wine()
        features_df = pd.DataFrame(data.data, columns=data.feature_names)
        target_series = pd.Series(data.target, name="target")

        logger.info(
            f"Dataset loaded: {features_df.shape[0]} samples, {features_df.shape[1]} features"
        )
        logger.info(f"Target classes: {np.unique(target_series).tolist()}")

        return features_df, target_series
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")


def explore_data(features: pd.DataFrame, target: pd.Series):
    """
    Perform basic exploratory data analysis.

    Args:
        features: Feature dataframe
        target: Target series

    Returns:
        Dictionary with EDA insights
    """
    logger.info("Starting exploratory data analysis...")

    # Basic statistics
    eda_insights = {
        "n_samples": len(features),
        "n_features": len(features.columns),
        "n_classes": len(target.unique()),
        "class_distribution": target.value_counts().to_dict(),
        "missing_values": features.isnull().sum().sum(),
        "feature_stats": {
            "mean": features.mean().to_dict(),
            "std": features.std().to_dict(),
        },
    }

    # Create a simple visualization
    plt.figure(figsize=(10, 6))

    # Class distribution plot
    plt.subplot(1, 2, 1)
    target.value_counts().plot(kind="bar")
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")

    # Feature correlation heatmap (simplified - just show first few features)
    plt.subplot(1, 2, 2)
    corr_matrix = features.iloc[:, :5].corr()  # First 5 features to keep it readable
    plt.imshow(corr_matrix, cmap="coolwarm", aspect="auto")
    plt.colorbar()
    plt.title("Feature Correlation (First 5 Features)")
    plt.xticks(range(5), corr_matrix.columns, rotation=45)
    plt.yticks(range(5), corr_matrix.columns)

    plt.tight_layout()
    plt.savefig("eda_report.png", dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(
        f"EDA complete: {eda_insights['n_samples']} samples, {eda_insights['missing_values']} missing values"
    )

    return eda_insights


def preprocess_data(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Preprocess the data: scale features and split into train/test sets.

    Args:
        features: Feature dataframe
        target: Target series
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    logger.info(f"Preprocessing data with test_size={test_size}...")

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logger.info(
        f"Data split: {len(X_train_scaled)} train, {len(X_test_scaled)} test samples"
    )
    logger.info("Features scaled using StandardScaler")

    print(type(X_train_scaled), type(y_train.values))
    print(X_train_scaled.shape, y_train.values.shape)

    return X_train_scaled, X_test_scaled, y_train.values, y_test.values


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_type: str = "random_forest",
    n_estimators: int = 100,
    max_depth=None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    max_iter: int = 1000,
    C: float = 1.0,
):
    """
    Train a machine learning model.

    Args:
        X_train: Training features
        y_train: Training targets
        model_type: Type of model to train
        **model_params: Model hyperparameters

    Returns:
        Trained model object
    """
    logger.info(f"Training {model_type} model...")

    if model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
        )
    elif model_type == "logistic_regression":
        model = LogisticRegression(max_iter=max_iter, C=C, random_state=42)
    else:
        raise ValueError(f"Model type {model_type} not supported")

    model.fit(X_train, y_train)

    logger.info(f"Model training complete: {model_type}")

    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model and generate metrics.

    Args:
        model: Trained model object
        X_test: Test features
        y_test: Test targets

    Returns:
        Dictionary containing evaluation metrics
    """
    logger.info("Evaluating model performance...")

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Create evaluation report
    evaluation_results = {
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": conf_matrix.tolist(),
    }

    # Save detailed report as JSON
    with open("evaluation_report.json", "w") as f:
        json.dump(evaluation_results, f, indent=2)

    # Create confusion matrix visualization
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()

    # Add labels
    classes = range(len(conf_matrix))
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    # Add text annotations
    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(
                j,
                i,
                conf_matrix[i, j],
                horizontalalignment="center",
                color="white" if conf_matrix[i, j] > conf_matrix.max() / 2 else "black",
            )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Model evaluation complete: Accuracy = {accuracy:.3f}")

    return evaluation_results, accuracy
