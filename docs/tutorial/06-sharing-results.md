# Sharing Results

You've trained a great model! But what happens next? Your teammate needs to use it, or you want to compare today's model with yesterday's. Let's make your results persistent and shareable.

## The Disappearing Results Problem

After running your pipeline, where are your results?

```python
# Train a model...
pipeline.execute()
# Great! But where is the model now?
# Can your teammate use it?
# Can you compare with yesterday's run?
```

**Problems:**

- Models only exist during execution
- No way to track metrics over time
- Can't share trained models with teammates
- Hard to compare different runs

## The Solution: Persistent Storage

Let's save models and metrics that persist beyond execution:

```python title="examples/tutorials/getting-started/06_sharing_results.py"
from runnable import Pipeline, PythonTask, Catalog, metric
import pickle

def save_model_artifact(model_data):
    """Save trained model to a file."""
    with open("trained_model.pkl", "wb") as f:
        pickle.dump(model_data, f)
    return model_data

PythonTask(
    function=save_model_artifact,
    name="save_model",
    catalog=Catalog(put=["trained_model.pkl"]),  # Store in catalog
    returns=[pickled("model_data")]
)
```

**Try it:**

```bash
uv run examples/tutorials/getting-started/06_sharing_results.py
```

## Storing Model Artifacts

Save your trained models so they can be reused:

```python
def save_model_artifact(model_data):
    """Save model to file for sharing."""
    with open("trained_model.pkl", "wb") as f:
        pickle.dump(model_data, f)

    print(f"Model saved: trained_model.pkl")
    return model_data

# Store the model file in catalog
PythonTask(
    function=save_model_artifact,
    name="save_model",
    catalog=Catalog(put=["trained_model.pkl"])
)
```

**What happens:**

1. Model is saved to `trained_model.pkl` in your working directory
2. Runnable copies it to `.catalog/` for permanent storage
3. File is now available for future runs or teammates

## Tracking Metrics Over Time

Use `metric()` to track performance metrics:

```python
def save_evaluation_metrics(evaluation_results):
    """Save metrics for tracking."""
    accuracy = evaluation_results['accuracy']
    report = evaluation_results['classification_report']

    # Save detailed report
    with open("evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # Return metrics for tracking
    return {
        "accuracy": accuracy,
        "precision": report['weighted avg']['precision'],
        "recall": report['weighted avg']['recall'],
        "f1_score": report['weighted avg']['f1-score']
    }

PythonTask(
    function=save_evaluation_metrics,
    name="save_metrics",
    catalog=Catalog(put=["evaluation_report.json"]),
    returns=[
        metric("accuracy"),      # Tracked as metrics
        metric("precision"),
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

## Loading Saved Models

Your teammate (or you in a future run) can load the saved model:

```python
def load_and_verify_model():
    """Load a previously saved model."""
    with open("trained_model.pkl", "rb") as f:
        model_data = pickle.load(f)

    print(f"Model loaded successfully!")
    return {"model_verified": True}

PythonTask(
    function=load_and_verify_model,
    name="verify_model",
    catalog=Catalog(get=["trained_model.pkl"])  # Get the saved model
)
```

## Complete Pipeline with Persistent Storage

Here's the full pipeline that saves and shares results:

```python title="examples/tutorials/getting-started/06_sharing_results.py"
--8<-- "examples/tutorials/getting-started/06_sharing_results.py:77:137"
```

## What You Get with Persistent Storage

### 📦 **Shareable Model Artifacts**

Models are stored in the catalog and can be shared:

```bash
# Your trained model is here
ls .catalog/*/trained_model.pkl

# Teammate can copy and use it
cp .catalog/run-id-123/trained_model.pkl shared-models/
```

### 📊 **Metrics Tracking**

All metrics are tracked in the run log:

```json
{
  "run_id": "happy-euler-0123",
  "metrics": {
    "accuracy": 0.9234,
    "precision": 0.9156,
    "recall": 0.9234,
    "f1_score": 0.9189
  }
}
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

### 🤝 **Team Collaboration**

- Share trained models without retraining
- Compare your model with teammate's models
- Reuse preprocessing results
- Track team's overall progress

## Storage Locations

Runnable keeps everything organized:

```bash
.catalog/                    # Persistent file storage
  ├── run-id-123/
  │   ├── trained_model.pkl       # Your model
  │   ├── evaluation_report.json  # Detailed metrics
  │   └── metrics_summary.json    # Quick summary
  └── run-id-124/
      └── trained_model.pkl       # Next run's model

.run_log_store/             # Execution metadata
  ├── run-id-123/
  │   └── run_log.json           # Includes metrics
  └── run-id-124/
      └── run_log.json
```

## Compare: Transient vs Persistent

**Transient Results (Chapters 1-5):**

- ❌ Results lost after execution
- ❌ Can't compare different runs
- ❌ Can't share with teammates
- ❌ Must retrain to reuse model

**Persistent Results (Chapter 6):**

- ✅ Models saved permanently
- ✅ Metrics tracked over time
- ✅ Easy to share with team
- ✅ Reuse without retraining
- ✅ Performance history available

## Real-World Use Cases

### Model Versioning

```python
# Save each model with version info
def save_model_v2(model_data):
    with open(f"model_v2.pkl", "wb") as f:
        pickle.dump(model_data, f)

catalog=Catalog(put=["model_v2.pkl"])
```

### A/B Testing

```python
# Compare two models
returns=[
    metric("model_a_accuracy"),
    metric("model_b_accuracy")
]
```

### Experiment Tracking

```python
# Track different hyperparameters
returns=[
    metric("accuracy"),
    metric("n_estimators"),  # Track hyperparameter used
    metric("max_depth")
]
```

## What's Next?

We have reproducible pipelines, flexible configuration, efficient data handling, and persistent results. But everything is running on your laptop. What about production?

**Next chapter:** We'll show how the same pipeline runs anywhere - your laptop, containers, or Kubernetes - without code changes.

---

**Next:** [Running Anywhere](07-running-anywhere.md) - Same code, different environments
