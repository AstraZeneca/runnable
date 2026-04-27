# Tracking Model Performance

You've trained models and stored files (Chapter 5), but how do you track **performance over time**? How do you compare today's 94% accuracy with yesterday's 91%? Let's add structured metrics tracking for model analytics.

## The Metrics Tracking Problem

After running your pipeline, you have files but no performance history:

```python
# Train a model...
pipeline.execute()
# Files are saved (Chapter 5 ✅) but:
# - What was the accuracy?
# - How does it compare to previous runs?
# - Which hyperparameters worked best?
# - Can teammates see model performance trends?
```

**The missing piece: structured performance tracking**

- Metrics buried in output logs
- No systematic performance comparison
- Hard to track model improvements over time
- Team can't easily compare model performance

## The Solution: Structured Metrics Tracking

Building on Chapter 5's file storage, let's add **metrics tracking** for performance analytics:

```python title="examples/tutorials/getting-started/06_sharing_results.py"
from runnable import Pipeline, PythonTask, Catalog, metric, pickled

def save_evaluation_metrics(evaluation_results):
    """Extract and return metrics for tracking."""
    accuracy = evaluation_results['accuracy']
    report = evaluation_results['classification_report']

    # Return metrics for structured tracking (not just files)
    return (
        accuracy,
        report['weighted avg']['precision'],
        report['weighted avg']['recall'],
        report['weighted avg']['f1-score']
    )

PythonTask(
    function=save_evaluation_metrics,
    name="save_metrics",
    returns=[
        metric("accuracy"),      # ← Tracked in run log for analytics
        metric("precision"),     # ← Not just saved to files
        metric("recall"),
        metric("f1_score")
    ]
)
```

**Try it:**

```bash
uv run examples/tutorials/getting-started/06_sharing_results.py
```

## Understanding `metric()` vs `pickled()` Returns

**Key insight:** Different return types serve different purposes:

```python
# Chapter 5 approach: File storage
returns=[pickled("model_data")]     # Stores complex objects
catalog=Catalog(put=["file.csv"])   # Stores files

# Chapter 6 approach: Metrics tracking
returns=[metric("accuracy")]        # Tracks performance numbers
# No catalog needed - metrics go to run log
```

**How `metric()` works:**

1. **Structured storage**: Metrics stored as key-value pairs in run log
2. **Easy comparison**: Query and compare across runs
3. **Analytics ready**: Perfect for tracking trends and performance
4. **Lightweight**: Just numbers, not large files or objects

**When to use what:**

- **`metric()`**: Performance numbers, hyperparameters, counts
- **`pickled()`**: Models, complex objects, datasets
- **`catalog=Catalog(put=[])`**: Files, reports, artifacts

## Tracking Metrics Over Time

Use `metric()` to track performance metrics:

```python
def save_evaluation_metrics(evaluation_results):
    """Save metrics for tracking."""
    accuracy = evaluation_results['accuracy']
    report = evaluation_results['classification_report']

    # Return metrics for tracking (metrics don't need files)
    return (
        accuracy,
        report['weighted avg']['precision'],
        report['weighted avg']['recall'],
        report['weighted avg']['f1-score']
    )

PythonTask(
    function=save_evaluation_metrics,
    name="save_metrics",
    returns=[
        metric("accuracy"),      # Tracked in run log automatically
        metric("precision"),     # No catalog needed for metrics
        metric("recall"),
        metric("f1_score")
    ]
)
```

**Metrics are special:**

- Automatically tracked in run logs
- Easy to compare across runs
- Can be visualized over time
- Help identify model improvements

## Comparing Metrics Across Runs

The real power: compare performance across different runs and experiments:

```bash
# Run different experiments
RUNNABLE_PRM_n_estimators=50 uv run 06_sharing_results.py   # Run 1
RUNNABLE_PRM_n_estimators=100 uv run 06_sharing_results.py  # Run 2
RUNNABLE_PRM_n_estimators=200 uv run 06_sharing_results.py  # Run 3

# Compare metrics across all runs
ls .run_log_store/
# curious-euler-0123/  # n_estimators=50
# happy-tesla-0124/    # n_estimators=100
# wise-darwin-0125/    # n_estimators=200
```

**Each run's metrics are tracked:**

```json
# .run_log_store/curious-euler-0123/run_log.json
{
  "run_id": "curious-euler-0123",
  "parameters": {"n_estimators": 50},
  "metrics": {
    "accuracy": 0.8234,
    "precision": 0.8156,
    "recall": 0.8234,
    "f1_score": 0.8189
  }
}
```

## Complete Pipeline with Metrics Tracking

Here's a pipeline that combines Chapter 5's file storage with metrics tracking:

```python title="examples/tutorials/getting-started/06_sharing_results.py"
--8<-- "examples/tutorials/getting-started/06_sharing_results.py:114:124"
```

## What You Get with Metrics Tracking

### 📊 **Structured Performance Data**

Unlike buried logs, metrics are stored in structured format:

```json
# Easy to parse and compare
{
  "run_id": "happy-euler-0123",
  "parameters": {"n_estimators": 100, "test_size": 0.2},
  "metrics": {
    "accuracy": 0.9234,
    "precision": 0.9156
  }
}
```

### 🎯 **Experiment Comparison**

Compare hyperparameters and results side by side:

```bash
# Run A: n_estimators=50  → accuracy: 0.8234
# Run B: n_estimators=100 → accuracy: 0.9156
# Run C: n_estimators=200 → accuracy: 0.9234

# Clear winner: Run C with n_estimators=200
```

### 📈 **Performance History**

Compare different runs:

```bash
# See all your runs
ls .run_log_store/

# Compare metrics from different runs
cat .run_log_store/run-1/run_log.json | grep accuracy
cat .run_log_store/run-2/run_log.json | grep accuracy
```

### 🤝 **Team Performance Tracking**

- Compare metrics across team members' experiments
- Track overall model performance improvements
- See which hyperparameters work best across the team
- Build shared knowledge of what approaches work

## Where Metrics Are Stored

**Metrics live in the run log** (not catalog like files from Chapter 5):

```bash
.run_log_store/             # Metrics stored here
  ├── run-id-123/
  │   └── run_log.json      # Contains structured metrics
  └── run-id-124/
      └── run_log.json      # Each run's metrics

# Quick metrics lookup
cat .run_log_store/*/run_log.json | grep -A 10 '"metrics"'
```

**Key difference from Chapter 5:**
- **Files** → `.catalog/` (datasets, models, reports)
- **Metrics** → `.run_log_store/` (performance numbers)

## Compare: Ad-hoc vs Structured Metrics

**Ad-hoc Performance Tracking (Chapters 1-5):**

- ❌ Metrics buried in print statements
- ❌ No systematic comparison across runs
- ❌ Hard to answer "which experiment was best?"
- ❌ Team can't easily compare approaches

**Structured Metrics Tracking (Chapter 6):**

- ✅ Metrics stored as searchable data
- ✅ Easy comparison across experiments
- ✅ Clear performance trends over time
- ✅ Team collaboration on model performance
- ✅ Data-driven experiment decisions

## Real-World Metrics Use Cases

### Hyperparameter Optimization

```python
# Track different hyperparameter combinations
returns=[
    metric("accuracy"),
    metric("n_estimators"),     # Track hyperparameter used
    metric("max_depth"),        # Track another hyperparameter
    metric("training_time")     # Track performance metrics
]
```

### A/B Testing

```python
# Compare two models
returns=[
    metric("model_a_accuracy"),
    metric("model_b_accuracy")
]
```

### Team Leaderboard

```python
# Everyone tracks the same metrics
returns=[
    metric("accuracy"),
    metric("f1_score"),
    metric("training_time"),
    metric("data_scientist")  # Track who ran the experiment
]

# Easy to see: "Who achieved the best accuracy?"
# "Which approach is fastest?"
```

## What's Next?

We have reproducible pipelines, flexible configuration, efficient data handling (Chapter 5), and structured metrics tracking (Chapter 6). But everything is running on your laptop. What about production?

**Next chapter:** We'll show how the same pipeline runs anywhere - your laptop, containers, or Kubernetes - without code changes.

---

**Next:** [Running Anywhere](07-running-anywhere.md) - Same code, different environments
