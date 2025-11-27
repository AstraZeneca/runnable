"""
ML Model Comparison Tutorial - Core Functions

This module demonstrates how to use Runnable's parallel execution to train
and compare multiple machine learning models simultaneously.

Features demonstrated:
- Parallel model training
- Model comparison and evaluation
- Performance visualization
- Model persistence and catalog management
"""

import json
import logging
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_digits, load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_dataset(dataset_name: str = "breast_cancer", test_size: float = 0.2, random_state: int = 42):
    """
    Load and prepare a dataset for model comparison.

    Args:
        dataset_name: Name of the dataset to load
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, dataset_info)
    """
    logger.info(f"Loading {dataset_name} dataset...")

    # Load dataset
    if dataset_name == "breast_cancer":
        data = load_breast_cancer()
    elif dataset_name == "wine":
        data = load_wine()
    elif dataset_name == "digits":
        data = load_digits()
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")

    # Create DataFrames
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='target')

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    dataset_info = {
        "name": dataset_name,
        "n_samples": len(X),
        "n_features": X.shape[1],
        "n_classes": len(np.unique(y)),
        "class_names": data.target_names.tolist() if hasattr(data, 'target_names') else None,
        "feature_names": data.feature_names.tolist(),
        "train_size": len(X_train_scaled),
        "test_size": len(X_test_scaled)
    }

    logger.info(f"Dataset loaded: {dataset_info['n_samples']} samples, {dataset_info['n_features']} features, {dataset_info['n_classes']} classes")

    return X_train_scaled, X_test_scaled, y_train.values, y_test.values, dataset_info


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray,
                       n_estimators: int = 100, max_depth = None,
                       min_samples_split: int = 2, random_state: int = 42):
    """Train a Random Forest model."""
    start_time = time.time()
    logger.info("Training Random Forest model...")

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    # Get feature importance
    feature_importance = model.feature_importances_

    model_info = {
        "name": "Random Forest",
        "algorithm": "ensemble",
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "training_time": training_time,
        "feature_importance": feature_importance.tolist()
    }

    logger.info(f"Random Forest training complete in {training_time:.2f} seconds")
    return model, model_info


def train_logistic_regression(X_train: np.ndarray, y_train: np.ndarray,
                             C: float = 1.0, max_iter: int = 1000, random_state: int = 42):
    """Train a Logistic Regression model."""
    start_time = time.time()
    logger.info("Training Logistic Regression model...")

    model = LogisticRegression(
        C=C,
        max_iter=max_iter,
        random_state=random_state,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    model_info = {
        "name": "Logistic Regression",
        "algorithm": "linear",
        "C": C,
        "max_iter": max_iter,
        "training_time": training_time,
        "n_features": X_train.shape[1]
    }

    logger.info(f"Logistic Regression training complete in {training_time:.2f} seconds")
    return model, model_info


def train_svm(X_train: np.ndarray, y_train: np.ndarray,
              C: float = 1.0, kernel: str = "rbf", random_state: int = 42):
    """Train an SVM model."""
    start_time = time.time()
    logger.info("Training SVM model...")

    model = SVC(
        C=C,
        kernel=kernel,
        random_state=random_state,
        probability=True  # Enable probability estimates
    )

    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    model_info = {
        "name": "SVM",
        "algorithm": "kernel",
        "C": C,
        "kernel": kernel,
        "training_time": training_time,
        "n_support": model.n_support_.tolist()
    }

    logger.info(f"SVM training complete in {training_time:.2f} seconds")
    return model, model_info


def train_knn(X_train: np.ndarray, y_train: np.ndarray,
              n_neighbors: int = 5, weights: str = "uniform"):
    """Train a K-Nearest Neighbors model."""
    start_time = time.time()
    logger.info("Training K-Nearest Neighbors model...")

    model = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    model_info = {
        "name": "K-Nearest Neighbors",
        "algorithm": "instance_based",
        "n_neighbors": n_neighbors,
        "weights": weights,
        "training_time": training_time
    }

    logger.info(f"KNN training complete in {training_time:.2f} seconds")
    return model, model_info


def evaluate_model(model, model_info: dict, X_test: np.ndarray, y_test: np.ndarray):
    """
    Evaluate a trained model and return comprehensive metrics.

    Args:
        model: Trained scikit-learn model
        model_info: Information about the model
        X_test: Test features
        y_test: Test targets

    Returns:
        Dictionary containing evaluation results
    """
    start_time = time.time()
    logger.info(f"Evaluating {model_info['name']} model...")

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Get prediction probabilities if available
    try:
        model.predict_proba(X_test)
        has_probabilities = True
    except AttributeError:
        has_probabilities = False

    evaluation_time = time.time() - start_time

    evaluation_results = {
        "model_name": model_info['name'],
        "algorithm": model_info['algorithm'],
        "accuracy": accuracy,
        "precision": report['macro avg']['precision'],
        "recall": report['macro avg']['recall'],
        "f1_score": report['macro avg']['f1-score'],
        "classification_report": report,
        "training_time": model_info['training_time'],
        "evaluation_time": evaluation_time,
        "has_probabilities": has_probabilities,
        "predictions": y_pred.tolist(),
        "model_params": {k: v for k, v in model_info.items()
                        if k not in ['name', 'algorithm', 'training_time']}
    }

    logger.info(f"{model_info['name']} evaluation complete: Accuracy = {accuracy:.3f}")

    return evaluation_results


def compare_models(rf_results: dict, lr_results: dict, svm_results: dict, knn_results: dict, dataset_info: dict):
    """
    Compare all model results and generate comprehensive comparison report.

    Args:
        rf_results: Random Forest evaluation results
        lr_results: Logistic Regression evaluation results
        svm_results: SVM evaluation results
        knn_results: KNN evaluation results
        dataset_info: Information about the dataset

    Returns:
        Comparison results and generates visualization files
    """
    logger.info("Comparing model performance...")

    # Collect all results
    all_results = [rf_results, lr_results, svm_results, knn_results]

    # Create comparison DataFrame
    comparison_data = []
    for result in all_results:
        comparison_data.append({
            'Model': result['model_name'],
            'Algorithm': result['algorithm'],
            'Accuracy': result['accuracy'],
            'Precision': result['precision'],
            'Recall': result['recall'],
            'F1-Score': result['f1_score'],
            'Training Time (s)': result['training_time'],
            'Evaluation Time (s)': result['evaluation_time']
        })

    comparison_df = pd.DataFrame(comparison_data)

    # Find best model
    best_model_idx = comparison_df['Accuracy'].idxmax()
    best_model = comparison_df.iloc[best_model_idx]

    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Model Comparison Results - {dataset_info["name"].title()} Dataset', fontsize=16)

    # 1. Accuracy comparison
    axes[0, 0].bar(comparison_df['Model'], comparison_df['Accuracy'], color='skyblue')
    axes[0, 0].set_title('Model Accuracy Comparison')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].set_ylim(0, 1)

    # Add accuracy values on bars
    for i, v in enumerate(comparison_df['Accuracy']):
        axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

    # 2. Training time comparison
    axes[0, 1].bar(comparison_df['Model'], comparison_df['Training Time (s)'], color='lightcoral')
    axes[0, 1].set_title('Training Time Comparison')
    axes[0, 1].set_ylabel('Training Time (seconds)')
    axes[0, 1].tick_params(axis='x', rotation=45)

    # 3. F1-Score comparison
    axes[1, 0].bar(comparison_df['Model'], comparison_df['F1-Score'], color='lightgreen')
    axes[1, 0].set_title('F1-Score Comparison')
    axes[1, 0].set_ylabel('F1-Score')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].set_ylim(0, 1)

    # 4. Multi-metric radar chart
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    ax_radar = plt.subplot(2, 2, 4, projection='polar')
    colors = ['blue', 'red', 'green', 'orange']

    for i, result in enumerate(all_results):
        values = [result['accuracy'], result['precision'], result['recall'], result['f1_score']]
        values += values[:1]  # Complete the circle

        ax_radar.plot(angles, values, 'o-', linewidth=2, label=result['model_name'], color=colors[i])
        ax_radar.fill(angles, values, alpha=0.1, color=colors[i])

    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(metrics)
    ax_radar.set_ylim(0, 1)
    ax_radar.set_title('Multi-Metric Performance Comparison')
    ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    plt.tight_layout()
    plt.savefig('model_comparison_results.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Save comparison results
    comparison_summary = {
        "dataset_info": dataset_info,
        "comparison_table": comparison_df.to_dict('records'),
        "best_model": {
            "name": best_model['Model'],
            "accuracy": best_model['Accuracy'],
            "algorithm": best_model['Algorithm']
        },
        "summary_stats": {
            "avg_accuracy": comparison_df['Accuracy'].mean(),
            "std_accuracy": comparison_df['Accuracy'].std(),
            "fastest_training": comparison_df.loc[comparison_df['Training Time (s)'].idxmin(), 'Model'],
            "slowest_training": comparison_df.loc[comparison_df['Training Time (s)'].idxmax(), 'Model']
        }
    }

    # Save to JSON
    with open('model_comparison_report.json', 'w') as f:
        json.dump(comparison_summary, f, indent=2)

    logger.info(f"Model comparison complete. Best model: {best_model['Model']} (Accuracy: {best_model['Accuracy']:.3f})")

    return comparison_summary, best_model['Accuracy']


# Wrapper functions for proper parameter mapping in the pipeline

def evaluate_rf_model(rf_model, rf_model_info, X_test: np.ndarray, y_test: np.ndarray):
    """Wrapper for evaluating Random Forest model."""
    return evaluate_model(rf_model, rf_model_info, X_test, y_test)


def evaluate_lr_model(lr_model, lr_model_info, X_test: np.ndarray, y_test: np.ndarray):
    """Wrapper for evaluating Logistic Regression model."""
    return evaluate_model(lr_model, lr_model_info, X_test, y_test)


def evaluate_svm_model(svm_model, svm_model_info, X_test: np.ndarray, y_test: np.ndarray):
    """Wrapper for evaluating SVM model."""
    return evaluate_model(svm_model, svm_model_info, X_test, y_test)


def evaluate_knn_model(knn_model, knn_model_info, X_test: np.ndarray, y_test: np.ndarray):
    """Wrapper for evaluating KNN model."""
    return evaluate_model(knn_model, knn_model_info, X_test, y_test)
