# Data Science Pipeline 101

A comprehensive tutorial demonstrating a complete machine learning workflow using Runnable. This example showcases core Runnable features through a practical data science pipeline.

## What You'll Learn

This tutorial demonstrates:
- **Parameter passing** between tasks using YAML configuration
- **Object serialization** for datasets and models (pandas DataFrames, scikit-learn models)
- **File catalog** for storing plots and reports
- **Metrics collection** for tracking model performance
- **Pipeline orchestration** with clear task dependencies

## Pipeline Overview

The pipeline implements a typical data science workflow:

```
Load Data → Explore → Preprocess → Train → Evaluate
```

### Pipeline Steps

1. **Load Data**: Load the wine dataset from scikit-learn
2. **Explore Data**: Perform EDA and generate visualizations
3. **Preprocess Data**: Scale features and split into train/test sets
4. **Train Model**: Train a Random Forest or Logistic Regression model
5. **Evaluate Model**: Calculate metrics and generate reports

## Running the Tutorial

### Prerequisites

Install the tutorial dependencies:

```bash
uv sync --group tutorial
```

### Execute the Pipeline

```bash
uv run --group tutorial examples/tutorials/data_science_101/pipeline.py
```

### What Gets Generated

After running the pipeline, you'll find:

**Output Files:**
- `eda_report.png` - Exploratory data analysis visualizations
- `confusion_matrix.png` - Model performance visualization
- `evaluation_report.json` - Detailed evaluation metrics

**Catalog Contents** (`.catalog/<run_id>/`):
- All intermediate data objects (datasets, model, etc.)
- Execution logs for each task
- Copies of all output files

**Run Logs** (`.run_log_store/`):
- Complete pipeline execution history
- Parameter tracking and metrics

## Configuration

Modify `parameters.yaml` to experiment with different settings:

```yaml
# Try different models
model_type: "logistic_regression"  # or "random_forest"

# Adjust model parameters
n_estimators: 200  # for Random Forest
max_depth: 10      # for Random Forest
C: 0.1             # for Logistic Regression

# Change data split
test_size: 0.3
random_state: 123
```

## Key Runnable Features Demonstrated

### 1. Parameter Flow
Parameters from the YAML file flow automatically to all tasks:

```python
# In functions.py
def train_model(X_train, y_train, model_type: str = "random_forest",
               n_estimators: int = 100, max_depth = None):
    # Parameters automatically injected from YAML
```

### 2. Object Serialization
Complex objects are automatically pickled and passed between tasks:

```python
# Task returns are automatically handled
returns=[
    pickled("features"),  # pandas DataFrame
    pickled("model"),     # scikit-learn model
]
```

### 3. File Catalog
Files are automatically stored in the catalog for persistence:

```python
catalog=Catalog(
    put=["eda_report.png", "confusion_matrix.png"]
)
```

### 4. Metrics Collection
Track important metrics across pipeline runs:

```python
returns=[
    metric("accuracy"),  # Tracked in run logs
]
```

## Expected Results

The wine dataset is well-suited for classification, so you should see:
- **High accuracy** (often 95%+ or perfect)
- **Clean confusion matrix** with few misclassifications
- **Balanced class distribution** in the EDA report

## Extending the Tutorial

Try these modifications to learn more:

1. **Add Cross-Validation**: Create a task that performs k-fold validation
2. **Model Comparison**: Use parallel execution to compare multiple models
3. **Hyperparameter Tuning**: Use map functionality to test different parameters
4. **Feature Selection**: Add a task for feature importance analysis
5. **Different Dataset**: Modify `load_data()` to use other scikit-learn datasets

## File Structure

```
examples/tutorials/data_science_101/
├── README.md           # This file
├── pipeline.py         # Main pipeline definition
├── functions.py        # All task functions
└── parameters.yaml     # Configuration parameters
```

This tutorial provides a solid foundation for building more complex ML pipelines with Runnable!
