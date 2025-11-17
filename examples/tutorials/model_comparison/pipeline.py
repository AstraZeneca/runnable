"""
ML Model Comparison Pipeline

This tutorial demonstrates how to use Runnable's parallel execution to train
and compare multiple machine learning models simultaneously.

Key Runnable Features Demonstrated:
- Parallel execution for simultaneous model training
- Parameter passing and model configuration
- Model comparison and results aggregation
- File catalog for storing models and reports
- Metrics collection for performance tracking

Pipeline Flow:
Load Data → Preprocess → [RF | LR | SVM | KNN] → Compare Results
                         \    |    |    /
                          \   |    |   /
                           Parallel Training

Run this pipeline with:
    uv run --group tutorial examples/tutorials/model_comparison/pipeline.py

The pipeline will:
1. Load and preprocess a dataset
2. Train 4 different models in parallel:
   - Random Forest
   - Logistic Regression
   - Support Vector Machine
   - K-Nearest Neighbors
3. Evaluate each model's performance
4. Generate comparison visualizations and reports
"""

from examples.tutorials.model_comparison.functions import (
    compare_models,
    evaluate_knn_model,
    evaluate_lr_model,
    evaluate_rf_model,
    evaluate_svm_model,
    load_dataset,
    train_knn,
    train_logistic_regression,
    train_random_forest,
    train_svm,
)
from runnable import Catalog, Parallel, Pipeline, PythonTask, metric, pickled


def main():
    """
    Create and execute the model comparison pipeline.
    """

    # Task 1: Load and preprocess the dataset
    load_data_task = PythonTask(
        function=load_dataset,
        name="load_dataset",
        returns=[
            pickled("X_train"),
            pickled("X_test"),
            pickled("y_train"),
            pickled("y_test"),
            pickled("dataset_info"),
        ],
    )

    # Parallel Training Tasks - Each model trains independently

    # Random Forest Training Branch
    train_rf_task = PythonTask(
        function=train_random_forest,
        name="train_random_forest",
        returns=[
            pickled("rf_model"),
            pickled("rf_model_info"),
        ],
    )

    evaluate_rf_task = PythonTask(
        function=evaluate_rf_model,
        name="evaluate_random_forest",
        returns=[pickled("rf_results")],
    )

    rf_pipeline = Pipeline(steps=[train_rf_task, evaluate_rf_task])

    # Logistic Regression Training Branch
    train_lr_task = PythonTask(
        function=train_logistic_regression,
        name="train_logistic_regression",
        returns=[
            pickled("lr_model"),
            pickled("lr_model_info"),
        ],
    )

    evaluate_lr_task = PythonTask(
        function=evaluate_lr_model,
        name="evaluate_logistic_regression",
        returns=[pickled("lr_results")],
    )

    lr_pipeline = Pipeline(steps=[train_lr_task, evaluate_lr_task])

    # SVM Training Branch
    train_svm_task = PythonTask(
        function=train_svm,
        name="train_svm",
        returns=[
            pickled("svm_model"),
            pickled("svm_model_info"),
        ],
    )

    evaluate_svm_task = PythonTask(
        function=evaluate_svm_model,
        name="evaluate_svm",
        returns=[pickled("svm_results")],
    )

    svm_pipeline = Pipeline(steps=[train_svm_task, evaluate_svm_task])

    # KNN Training Branch
    train_knn_task = PythonTask(
        function=train_knn,
        name="train_knn",
        returns=[
            pickled("knn_model"),
            pickled("knn_model_info"),
        ],
    )

    evaluate_knn_task = PythonTask(
        function=evaluate_knn_model,
        name="evaluate_knn",
        returns=[pickled("knn_results")],
    )

    knn_pipeline = Pipeline(steps=[train_knn_task, evaluate_knn_task])

    # Parallel execution of all model training pipelines
    parallel_training = Parallel(
        name="train_all_models",
        branches={
            "random_forest": rf_pipeline,
            "logistic_regression": lr_pipeline,
            "svm": svm_pipeline,
            "knn": knn_pipeline,
        }
    )

    # Final comparison task that aggregates all results
    compare_models_task = PythonTask(
        function=compare_models,
        name="compare_models",
        returns=[
            pickled("comparison_summary"),
            metric("best_model_accuracy"),  # Track the best accuracy as a metric
        ],
        catalog=Catalog(
            put=["model_comparison_results.png", "model_comparison_report.json"]
        ),
        terminate_with_success=True,
    )

    # Create the main pipeline
    pipeline = Pipeline(
        steps=[
            load_data_task,
            parallel_training,
            compare_models_task,
        ]
    )

    # Execute the pipeline with parameters from YAML file
    pipeline.execute(
        parameters_file="examples/tutorials/model_comparison/parameters.yaml"
    )

    return pipeline


if __name__ == "__main__":
    main()
