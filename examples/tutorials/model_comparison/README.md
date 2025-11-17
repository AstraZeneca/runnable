# ML Model Comparison Tutorial

A comprehensive tutorial demonstrating how to use Runnable's parallel execution to train and compare multiple machine learning models simultaneously.

## What You'll Learn

This tutorial showcases advanced Runnable features:
- **Parallel execution** for simultaneous model training
- **Pipeline branching** with independent model training paths
- **Results aggregation** from parallel branches
- **Advanced visualizations** with comprehensive model comparison
- **Performance metrics tracking** across different algorithms

## Pipeline Architecture

```
                Load Data → Preprocess
                           ↓
                    ┌─────────────────┐
                    │ Parallel Training │
                    └─────────────────┘
                   ↓      ↓      ↓      ↓
               Random   Logistic  SVM    KNN
               Forest   Regression
                   ↓      ↓      ↓      ↓
               Evaluate Evaluate Evaluate Evaluate
                    \      |      |     /
                     \     |      |    /
                      Compare Models
```

### Models Compared

1. **Random Forest** - Ensemble method with multiple decision trees
2. **Logistic Regression** - Linear method with regularization
3. **Support Vector Machine** - Kernel-based classification
4. **K-Nearest Neighbors** - Instance-based learning

## Running the Tutorial

### Prerequisites

Install the tutorial dependencies:

```bash
uv sync --group tutorial
```

### Execute the Pipeline

```bash
uv run --group tutorial examples/tutorials/model_comparison/pipeline.py
```

### What Gets Generated

**Comprehensive Visualizations:**
- `model_comparison_results.png` - Multi-panel comparison dashboard:
  - Accuracy comparison bar chart
  - Training time comparison
  - F1-Score comparison
  - Multi-metric radar chart

**Detailed Report:**
- `model_comparison_report.json` - Complete comparison data:
  - Individual model performance metrics
  - Best model identification
  - Statistical summaries
  - Training time analysis

**Catalog Contents** (`.catalog/<run_id>/`):
- All trained models (pickled)
- Model metadata and configuration
- Individual evaluation results
- Dataset information
- Execution logs for each parallel branch

## Configuration Options

Modify `parameters.yaml` to experiment with different settings:

### Dataset Selection
```yaml
dataset_name: "breast_cancer"  # "wine", "digits"
test_size: 0.2
random_state: 42
```

### Model Hyperparameters
```yaml
# Random Forest
n_estimators: 100
max_depth: null
min_samples_split: 2

# Logistic Regression & SVM
C: 1.0
max_iter: 1000

# SVM Specific
kernel: "rbf"  # "linear", "poly", "sigmoid"

# K-Nearest Neighbors
n_neighbors: 5
weights: "uniform"  # "distance"
```

## Expected Results

On the **Breast Cancer dataset** (569 samples, 30 features):
- **High accuracy** across all models (typically 95%+)
- **Logistic Regression and SVM** often tie for best performance
- **Random Forest** provides good performance with feature importance
- **KNN** typically fastest training, but may have lower accuracy

Performance characteristics:
- **Training Speed**: KNN > SVM > Random Forest > Logistic Regression
- **Prediction Speed**: Logistic Regression > SVM > Random Forest > KNN
- **Interpretability**: Logistic Regression > Random Forest > SVM > KNN

## Key Runnable Features Demonstrated

### 1. Parallel Execution
```python
parallel_training = Parallel(
    name="train_all_models",
    branches={
        "random_forest": rf_pipeline,
        "logistic_regression": lr_pipeline,
        "svm": svm_pipeline,
        "knn": knn_pipeline,
    }
)
```

### 2. Pipeline Branching
Each model has its own independent pipeline:
```python
rf_pipeline = Pipeline(steps=[train_rf_task, evaluate_rf_task])
```

### 3. Results Aggregation
The comparison task receives results from all parallel branches:
```python
compare_models_task = PythonTask(
    function=compare_models,  # Receives rf_results, lr_results, svm_results, knn_results
    returns=[pickled("comparison_summary"), metric("best_model_accuracy")]
)
```

### 4. Advanced Visualization
Generates comprehensive comparison dashboard with:
- Multiple chart types (bar charts, radar plot)
- Performance metrics comparison
- Training time analysis
- Statistical summaries

## Performance Analysis

The tutorial automatically tracks and compares:
- **Accuracy, Precision, Recall, F1-Score**
- **Training and evaluation times**
- **Statistical significance of differences**
- **Algorithm family performance** (ensemble vs linear vs kernel vs instance-based)

## Extending the Tutorial

### Add More Models
1. Create new training functions (e.g., `train_gradient_boosting`)
2. Add evaluation wrappers
3. Include in parallel branches
4. Update comparison function

### Custom Metrics
1. Modify evaluation functions to include custom metrics
2. Update visualization to display new metrics
3. Add statistical tests for significance

### Cross-Validation
1. Replace simple train/test split with k-fold CV
2. Use Runnable's map functionality for fold processing
3. Aggregate CV results across models

## Troubleshooting

**Memory Issues**: For large datasets, consider:
- Reducing model complexity (fewer trees, simpler kernels)
- Using incremental learning algorithms
- Implementing data streaming

**Slow Training**: To speed up execution:
- Reduce n_estimators for Random Forest
- Use linear kernel for SVM
- Decrease max_iter for iterative algorithms

**Poor Performance**: If accuracy is low:
- Check data preprocessing steps
- Tune hyperparameters
- Consider feature selection/engineering
- Verify data quality

This tutorial provides a solid foundation for building production-ready model comparison pipelines with Runnable!
