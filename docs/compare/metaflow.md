# 🆚 Runnable vs Metaflow: Capability Comparison

Both Runnable and Metaflow solve ML pipeline orchestration with different approaches. Here's a side-by-side comparison using a real ML workflow.

## The Example: Existing ML Functions

Let's start with typical Python functions you might already have:

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import joblib

def load_and_clean_data():
    """Your existing data loading function."""
    customers = pd.read_csv("s3://bucket/raw-data/customers.csv")
    transactions = pd.read_csv("s3://bucket/raw-data/transactions.csv")

    data = customers.merge(transactions, on="customer_id").dropna()
    X = data.drop(['target'], axis=1)
    y = data['target']

    X.to_csv("features.csv", index=False)
    y.to_csv("target.csv", index=False)
    return {"n_samples": len(X), "n_features": X.shape[1]}

def train_random_forest(n_samples, n_features, max_depth=10):
    """Your existing RF training function."""
    X = pd.read_csv("features.csv")
    y = pd.read_csv("target.csv").values.ravel()

    model = RandomForestClassifier(max_depth=max_depth, random_state=42)
    model.fit(X, y)
    joblib.dump(model, "rf_model.pkl")

    return {"model_type": "RandomForest", "accuracy": model.score(X, y)}

def train_xgboost(n_samples, n_features, max_depth=10):
    """Your existing XGBoost training function."""
    X = pd.read_csv("features.csv")
    y = pd.read_csv("target.csv").values.ravel()

    model = xgb.XGBClassifier(max_depth=max_depth, random_state=42)
    model.fit(X, y)
    joblib.dump(model, "xgb_model.pkl")

    return {"model_type": "XGBoost", "accuracy": model.score(X, y)}

def select_best_model(model_results):
    """Your existing model selection function."""
    best_model = max(model_results, key=lambda x: x['accuracy'])
    # Copy best model logic...
    return best_model
```

**Goal:** Create a pipeline that runs these functions with parallel model training.

---

## Making It Work with Runnable

**Work required:** Add pipeline wrapper (functions stay unchanged)

```python
from runnable import Pipeline, PythonTask, Parallel

# Import your existing functions (no changes needed)
from your_ml_code import load_and_clean_data, train_random_forest, train_xgboost, select_best_model

def main():
    pipeline = Pipeline(steps=[
        PythonTask(function=load_and_clean_data, returns=["n_samples", "n_features"]),
        Parallel(branches={
            "rf": PythonTask(function=train_random_forest, returns=["rf_results"]).as_pipeline(),
            "xgb": PythonTask(function=train_xgboost, returns=["xgb_results"]).as_pipeline()
        }),
        PythonTask(function=select_best_model, returns=["best_model"])
    ])
    pipeline.execute()
    return pipeline  # Required for Runnable

if __name__ == "__main__":
    main()
```

**That's it.** Functions unchanged, single wrapper file.

---

## Making It Work with Metaflow

**Work required:** Convert functions to FlowSpec class structure

**Functions Can Stay External:**

```python
# your_ml_code.py (functions unchanged)
def load_and_clean_data():
    # Your existing logic stays the same
    return {"n_samples": 1000, "n_features": 20}

def train_random_forest(n_samples, n_features, max_depth=10):
    # Your existing logic stays the same
    return {"model_type": "RandomForest", "accuracy": 0.95}
```

**Metaflow requires FlowSpec wrapper:**
```python
from metaflow import FlowSpec, step, Parameter
# Import your existing functions (no changes needed)
from your_ml_code import load_and_clean_data, train_random_forest, train_xgboost, select_best_model

class MLTrainingFlow(FlowSpec):
    max_depth = Parameter('max_depth', default=15)

    @step
    def start(self):
        # Call your existing function directly
        data_stats = load_and_clean_data()
        self.n_samples = data_stats['n_samples']
        self.n_features = data_stats['n_features']
        self.next(self.train_models, foreach=['RandomForest', 'XGBoost'])

    @step
    def train_models(self):
        # Call your existing functions directly
        if self.input == 'RandomForest':
            results = train_random_forest(self.n_samples, self.n_features, self.max_depth)
        else:
            results = train_xgboost(self.n_samples, self.n_features, self.max_depth)

        self.model_results = results
        self.next(self.select_best)

    @step
    def select_best(self, inputs):
        model_results = [input.model_results for input in inputs]
        self.best = select_best_model(model_results)  # Call your existing function
        self.next(self.end)

    @step
    def end(self):
        pass
```

**Running the Pipeline:**
```bash
python ml_metaflow.py run --max_depth 15
```

---

## Core Capabilities Comparison

### Workflow Features

| Feature | Runnable Approach | Metaflow Approach |
|---------|-------------------|-------------------|
| **Pipeline Definition** | Single Python file with minimal setup | FlowSpec class with decorators |
| **Task Types** | Python, Notebooks, Shell, Stubs | Python steps with flow state |
| **Parameter Configuration** | YAML/JSON config files via `parameters_file` | Config files and command-line parameters |
| **Parallel Execution** | `Parallel()` with explicit branching | `foreach` parameter for fan-out execution |
| **Conditional Logic** | Native `Conditional()` support | Manual implementation in step logic |
| **Map/Reduce** | Native `Map()` with custom reducers | `foreach` with join steps for result aggregation |

### Data Handling

| Feature | Runnable Approach | Metaflow Approach |
|---------|-------------------|-------------------|
| **File Management** | Automatic file sync via `Catalog(put/get)` | Manual file I/O - no catalog system |
| **Data Versioning** | Content-based hashing for change detection | Automatic versioning via Metaflow datastore (Python objects only) |
| **Storage Backends** | File, S3, Minio via plugins | Local, S3, Azure, GCP datastores |
| **Data Lineage** | Automatic via run logs | Rich lineage through Metaflow UI |

### Production Deployment

| Feature | Runnable Approach | Metaflow Approach |
|---------|-------------------|-------------------|
| **Environment Portability** | Same code runs local/container/K8s/Argo | Same FlowSpec runs local/AWS/K8s with --with flags |
| **AWS Integration** | Manual configuration required | Native AWS Batch, Step Functions integration |
| **Monitoring** | Basic run logs and timeline visualization | Rich Metaflow UI with execution graphs |
| **Extensibility** | Entry points auto-discovery for custom task types, executors, catalogs | Limited plugin system - primarily configuration-based extensions |


## When to Choose Each Tool

### **Choose Runnable When:**

- Working with existing Python functions without refactoring
- Need multi-environment portability (local → container → K8s → Argo)
- Require advanced workflow patterns (parallel, conditional, map-reduce)
- Want immediate productivity with minimal setup
- Working with mixed task types (Python + notebooks + shell)

### **Choose Metaflow When:**

- Need rich execution visualization and monitoring
- Heavy investment in AWS services and infrastructure
- Managing hundreds/thousands of concurrent workflows
- Want automatic Python object serialization between steps
- Already familiar with decorator-based patterns
- Need built-in experiment tracking and comparison

## Implementation Structure Comparison

**Runnable Approach:**

- **Minimal disruption**: Wrap existing functions directly without changes
- **Single file**: Complete pipeline in one Python file
- **No restructuring**: Keep your current code organization and patterns
- **Optional infrastructure**: Add AWS/K8s configs only when needed for specific environments

**Metaflow Approach:**

- **Function restructuring**: Convert existing functions to fit FlowSpec class patterns
- **Decorator-based**: Use `@step` and `@parallel` decorators for flow control
- **Flow state management**: Store data in `self` attributes between steps
- **Infrastructure integration**: Built-in AWS Batch, Step Functions, S3 datastore

---

**Next:** See how Runnable compares to [Kedro](kedro.md) and other orchestration tools.
