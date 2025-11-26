# 🆚 Runnable vs Kedro: Simplicity Wins

Both Runnable and Kedro solve pipeline orchestration, but with radically different philosophies. Here's a side-by-side comparison using a real ML workflow.

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
from runnable import Pipeline, PythonTask, Parallel, Catalog

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

## Making It Work with Kedro

**Work required:** Project restructuring + configuration files

### Required Project Structure

```
ml-kedro-project/
├── conf/base/
│   ├── catalog.yml          # Data source/destination definitions
│   ├── parameters.yml       # Pipeline parameters
│   └── logging.yml          # Logging configuration
├── src/ml_kedro_project/
│   ├── pipelines/
│   │   ├── data_engineering/
│   │   │   ├── nodes.py     # Data processing functions
│   │   │   └── pipeline.py  # Pipeline definition
│   │   └── data_science/
│   │       ├── nodes.py     # ML model functions
│   │       └── pipeline.py  # ML pipeline definition
│   └── pipeline_registry.py # Register all pipelines
└── pyproject.toml
```

### Configuration Files Required

**Data Catalog (`conf/base/catalog.yml`)**
```yaml
# Must define every data input/output with type and location
customers_raw:
  type: pandas.CSVDataSet
  filepath: data/01_raw/customers.csv

features:
  type: pandas.CSVDataSet
  filepath: data/03_primary/features.csv

rf_model:
  type: pickle.PickleDataSet
  filepath: data/06_models/rf_model.pkl
# ... repeat for all data assets
```

**Parameters (`conf/base/parameters.yml`)**
```yaml
model_options:
  max_depth: 15
  random_state: 42
```

### Functions Must Be Restructured

**Original function:**
```python
def train_random_forest(n_samples, n_features, max_depth=10):
    # Your existing logic
```

**Kedro requires changing to:**
```python
def train_random_forest(features: pd.DataFrame, target: pd.Series,
                       parameters: Dict[str, Any]) -> Dict[str, Any]:
    # Must accept data from catalog, parameters from config
    model = RandomForestClassifier(max_depth=parameters["model_options"]["max_depth"])
    # Restructured logic to fit Kedro patterns
    return {"model": model, "accuracy": accuracy}
```

**Pipeline Registration Required:**
```python
# src/ml_kedro_project/pipeline_registry.py
def register_pipelines() -> Dict[str, Pipeline]:
    return {
        "__default__": data_engineering.create_pipeline() + data_science.create_pipeline()
    }
```

**Running the Pipeline:**
```bash
kedro new --starter=pandas-iris ml-kedro-project
# Implement node functions, pipeline definitions, configurations
kedro run
```

---

## Core Capabilities Comparison

### Workflow Features

| Feature | Runnable Approach | Kedro Approach |
|---------|-------------------|----------------|
| **Pipeline Definition** | Single Python file with minimal setup | Structured project layout with enforced conventions |
| **Task Types** | Python, Notebooks, Shell, Stubs | Python nodes |
| **Parallel Execution** | `Parallel()` with explicit branching | Automatic dependency resolution |
| **Conditional Logic** | Native `Conditional()` support | Manual implementation in node logic |
| **Map/Reduce** | Native `Map()` with custom reducers | Manual implementation required |

### Data Handling

| Feature | Runnable Approach | Kedro Approach |
|---------|-------------------|----------------|
| **File Management** | Simple `Catalog(put/get)` with minimal config | Rich catalog.yml definitions with fine control |
| **Data Versioning** | Content-based hashing for change detection | Timestamp-based versioning |
| **Storage Backends** | File, S3, Minio via plugins | 20+ built-in dataset types with validation |
| **Data Lineage** | Automatic via run logs | kedro-viz visualization |

### Production Deployment

| Feature | Runnable Approach | Kedro Approach |
|---------|-------------------|----------------|
| **Environment Portability** | Same code runs local/container/K8s/Argo | Requires deployment-specific configurations |
| **Container Execution** | Same containerized code runs across environments | May require deployment-specific configurations |
| **Extensibility** | Entry points auto-discovery - custom executors, catalogs, secrets in your codebase | Plugin system - public kedro-* packages or custom internal plugins |
| **Monitoring** | Basic run logs | Rich hooks ecosystem |
| **MLOps Integration** | Tool-agnostic - choose your own MLOps stack | Plugin ecosystem (MLflow, Airflow via kedro-* packages) |


## When to Choose Each Tool

### **Choose Runnable When:**

- Working with existing Python functions without refactoring
- Need multi-environment portability (local → container → K8s → Argo)
- Require advanced workflow patterns (parallel, conditional, map-reduce)
- Want immediate productivity with minimal setup
- Working with mixed task types (Python + notebooks + shell)

### **Choose Kedro When:**

- Need standardized project structure across large teams
- Require rich data catalog features and validation
- Heavy ETL pipelines with extensive data governance needs
- Want established MLOps ecosystem integrations (MLflow, Airflow)
- Already invested in Kedro infrastructure and expertise


## Implementation Structure Comparison

**Runnable Approach:**

- **Minimal disruption**: Wrap existing functions directly without changes
- **Single file**: Complete pipeline in one Python file
- **No restructuring**: Keep your current code organization and patterns
- **Optional configuration**: Add YAML configs only when needed for specific environments

**Kedro Approach:**

- **Project restructuring**: Requires adopting Kedro's directory structure and conventions
- **Multi-file organization**: Separate files for nodes, pipelines, catalogs, and configurations
- **Function refactoring**: Convert existing functions to fit Kedro node patterns
- **Required configuration**: YAML files for catalog, parameters, and logging are essential

## 🚀 Try Both Yourself

**Test Runnable** (2 minutes):
```bash
pip install runnable
# Copy the Runnable example above
python ml_pipeline.py
```

**Test Kedro** (2+ hours):
```bash
pip install kedro
kedro new --starter=pandas-iris my-project
# Implement all the files shown above
kedro run
```

The productivity difference speaks for itself.

---

**Next:** See how Runnable compares to [Metaflow](metaflow.md) and other orchestration tools.
