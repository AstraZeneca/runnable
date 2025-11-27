"""
Data Science Pipeline 101

A complete machine learning pipeline demonstrating core Runnable features:
- Parameter passing between tasks
- Object serialization (datasets, models)
- File catalog for outputs (plots, reports)
- Metrics collection
- Configuration via YAML

This tutorial shows a typical data science workflow:
Load Data → Explore → Preprocess → Train → Evaluate

Run this pipeline with:
    uv run examples/tutorials/data_science_101/pipeline.py

The pipeline will:
1. Load the wine dataset from scikit-learn
2. Perform exploratory data analysis
3. Preprocess the data (scaling, train/test split)
4. Train a machine learning model
5. Evaluate the model and generate reports
"""

from examples.tutorials.data_science_101.functions import (
    evaluate_model,
    explore_data,
    load_data,
    preprocess_data,
    train_model,
)
from runnable import Catalog, Pipeline, PythonTask, metric, pickled


def main():
    """
    Create and execute the data science pipeline.
    """

    # Task 1: Load the dataset
    load_data_task = PythonTask(
        function=load_data,
        name="load_data",
        returns=[
            pickled("features"),  # DataFrame gets automatically pickled
            pickled("target"),  # Series gets automatically pickled
        ],
    )

    # Task 2: Explore the data and generate EDA report
    explore_data_task = PythonTask(
        function=explore_data,
        name="explore_data",
        returns=[pickled("eda_insights")],
        catalog=Catalog(
            put=["eda_report.png"]  # Save the EDA plot
        ),
    )

    # Task 3: Preprocess the data (scaling, train/test split)
    preprocess_data_task = PythonTask(
        function=preprocess_data,
        name="preprocess_data",
        returns=[
            pickled("X_train"),
            pickled("X_test"),
            pickled("y_train"),
            pickled("y_test"),
        ],
    )

    # Task 4: Train the model
    train_model_task = PythonTask(
        function=train_model,
        name="train_model",
        returns=[pickled("model")],
    )

    # Task 5: Evaluate the model and generate reports
    evaluate_model_task = PythonTask(
        function=evaluate_model,
        name="evaluate_model",
        returns=[
            pickled("evaluation_results"),
            metric("accuracy"),  # Extract accuracy as a metric
        ],
        catalog=Catalog(put=["evaluation_report.json", "confusion_matrix.png"]),
    )

    # Create the pipeline
    pipeline = Pipeline(
        steps=[
            load_data_task,
            explore_data_task,
            preprocess_data_task,
            train_model_task,
            evaluate_model_task,
        ]
    )

    # Execute the pipeline with parameters from YAML file
    pipeline.execute(
        parameters_file="examples/tutorials/data_science_101/parameters.yaml"
    )

    return pipeline


if __name__ == "__main__":
    main()
