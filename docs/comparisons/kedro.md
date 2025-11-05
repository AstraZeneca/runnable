# 🆚 Runnable vs Kedro: Simplicity Wins

Both Runnable and Kedro solve pipeline orchestration, but with radically different philosophies. Here's a side-by-side comparison using a real ML workflow.

## 📊 The Example: ML Training Pipeline

Let's build a machine learning pipeline that:

1. **Loads and cleans data** from multiple sources
2. **Trains two models** in parallel (Random Forest + XGBoost)
3. **Evaluates models** and selects the best one
4. **Generates a report** with results

This is a typical ML workflow with parallel execution, parameter passing, and file management.

---

## 🟢 The Runnable Way: Simple & Direct

**Single file implementation:**

```python title="ml_pipeline.py"
from runnable import Pipeline, PythonTask, Parallel, Catalog, pickled
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import joblib

def load_and_clean_data() -> dict:
    """Load multiple data sources and clean them."""
    # Load from S3 or central storage (not local files that don't exist yet)
    customers = pd.read_csv("s3://bucket/raw-data/customers.csv")
    transactions = pd.read_csv("s3://bucket/raw-data/transactions.csv")

    # Clean and merge
    data = customers.merge(transactions, on="customer_id")
    data = data.dropna()

    # Split features and target
    X = data.drop(['target'], axis=1)
    y = data['target']

    # Save processed data
    X.to_csv("features.csv", index=False)
    y.to_csv("target.csv", index=False)

    return {"n_samples": len(X), "n_features": X.shape[1]}

def train_random_forest(n_samples: int, n_features: int, max_depth: int = 10) -> dict:
    """Train Random Forest model."""
    X = pd.read_csv("features.csv")
    y = pd.read_csv("target.csv").values.ravel()

    model = RandomForestClassifier(max_depth=max_depth, random_state=42)
    model.fit(X, y)

    # Save model
    joblib.dump(model, "rf_model.pkl")

    accuracy = model.score(X, y)
    return {"model_type": "RandomForest", "accuracy": accuracy}

def train_xgboost(n_samples: int, n_features: int, max_depth: int = 10) -> dict:
    """Train XGBoost model."""
    X = pd.read_csv("features.csv")
    y = pd.read_csv("target.csv").values.ravel()

    model = xgb.XGBClassifier(max_depth=max_depth, random_state=42)
    model.fit(X, y)

    # Save model
    joblib.dump(model, "xgb_model.pkl")

    accuracy = model.score(X, y)
    return {"model_type": "XGBoost", "accuracy": accuracy}

def select_best_model(model_results: list) -> dict:
    """Compare models and select the best one."""
    # model_results is a list from parallel execution
    best_model = max(model_results, key=lambda x: x['accuracy'])

    print(f"Best model: {best_model['model_type']} with accuracy: {best_model['accuracy']:.3f}")

    # Copy best model to final location
    if best_model['model_type'] == 'RandomForest':
        import shutil
        shutil.copy("rf_model.pkl", "best_model.pkl")
    else:
        import shutil
        shutil.copy("xgb_model.pkl", "best_model.pkl")

    return best_model

def generate_report(best_model: dict, data_stats: dict) -> dict:
    """Generate final ML report."""
    report = f"""
    ML Pipeline Report
    ==================

    Data Statistics:
    - Samples: {data_stats['n_samples']}
    - Features: {data_stats['n_features']}

    Best Model: {best_model['model_type']}
    Accuracy: {best_model['accuracy']:.3f}

    Model saved as: best_model.pkl
    """

    with open("ml_report.txt", "w") as f:
        f.write(report)

    return {"report_path": "ml_report.txt"}

# Define the pipeline
def create_pipeline():
    # Step 1: Data preparation
    # Note: Domain code exists separately - we just wrap it here
    data_prep = PythonTask(
        function=load_and_clean_data,
        returns=["n_samples", "n_features"],  # Simple returns, no type annotations needed
        catalog=Catalog(
            # No 'get' - data comes from S3/central storage, not local catalog
            put=["features.csv", "target.csv"]
        ),
        name="data_preparation"
    )

    # Step 2: Parallel model training
    parallel_training = Parallel(
        name="train_models",
        branches={
            "random_forest": PythonTask(
                function=train_random_forest,
                returns=["rf_results"],
                catalog=Catalog(
                    get=["features.csv", "target.csv"],
                    put=["rf_model.pkl"]
                ),
                name="train_rf"
            ).as_pipeline(),

            "xgboost": PythonTask(
                function=train_xgboost,
                returns=["xgb_results"],
                catalog=Catalog(
                    get=["features.csv", "target.csv"],
                    put=["xgb_model.pkl"]
                ),
                name="train_xgb"
            ).as_pipeline()
        }
    )

    # Step 3: Model selection
    model_selection = PythonTask(
        function=select_best_model,
        returns=["best_model"],
        catalog=Catalog(
            get=["rf_model.pkl", "xgb_model.pkl"],
            put=["best_model.pkl"]
        ),
        name="select_best"
    )

    # Step 4: Report generation
    reporting = PythonTask(
        function=generate_report,
        returns=["report"],
        catalog=Catalog(put=["ml_report.txt"]),
        name="generate_report"
    )

    return Pipeline(steps=[
        data_prep,
        parallel_training,
        model_selection,
        reporting
    ])

if __name__ == "__main__":
    # Create parameters file for proper Runnable parameter handling
    import yaml
    with open("parameters.yaml", "w") as f:
        yaml.dump({"max_depth": 15}, f)

    pipeline = create_pipeline()
    # Parameters via file (recommended) or environment variables, not inline
    result = pipeline.execute(parameters_file="parameters.yaml")
    print(f"Pipeline completed! Run ID: {result.run_id}")
```

**Run it:**
```bash
uv run ml_pipeline.py
```

**That's it.** One file, runs immediately, works locally and in production.

!!! note "Domain Code Separation"

    Notice how your **business logic functions remain pure Python** - no framework dependencies. You could import these functions from completely separate modules:

    ```python
    # Import from your existing codebase
    from my_ml_library.preprocessing import load_and_clean_data
    from my_ml_library.models import train_random_forest, train_xgboost
    from my_ml_library.evaluation import select_best_model
    from my_ml_library.reporting import generate_report

    # Just wrap them with Runnable - no changes to your domain code needed
    ```

---

## 🔴 The Kedro Way: Configuration Heavy

The same pipeline in Kedro requires **multiple files and directories**:

### Project Structure (Kedro's Recommended Layout)
*Note: While this specific directory structure is Kedro's recommendation rather than a strict requirement, most teams follow this pattern for consistency:*
```
ml-kedro-project/
├── conf/
│   ├── base/
│   │   ├── catalog.yml          # Data definitions
│   │   ├── parameters.yml       # Parameters
│   │   └── logging.yml          # Logging config
│   └── local/
│       └── credentials.yml      # Local overrides
├── data/
│   ├── 01_raw/                  # Raw data
│   ├── 02_intermediate/         # Processed data
│   ├── 03_primary/             # Feature data
│   ├── 04_feature/             # Features
│   ├── 05_model_input/         # Model inputs
│   ├── 06_models/              # Saved models
│   ├── 07_model_output/        # Model outputs
│   └── 08_reporting/           # Reports
├── src/
│   └── ml_kedro_project/
│       ├── __init__.py
│       ├── pipeline_registry.py # Pipeline registration
│       ├── pipelines/
│       │   ├── __init__.py
│       │   ├── data_engineering/
│       │   │   ├── __init__.py
│       │   │   ├── nodes.py     # Data processing functions
│       │   │   └── pipeline.py  # Data pipeline definition
│       │   ├── data_science/
│       │   │   ├── __init__.py
│       │   │   ├── nodes.py     # ML functions
│       │   │   └── pipeline.py  # ML pipeline definition
│       │   └── reporting/
│       │       ├── __init__.py
│       │       ├── nodes.py     # Reporting functions
│       │       └── pipeline.py  # Reporting pipeline
│       └── settings.py          # Project settings
├── pyproject.toml              # Project configuration
└── README.md
```

### 1. Data Catalog Configuration (`conf/base/catalog.yml`)
```yaml
# Raw data
customers_raw:
  type: pandas.CSVDataSet
  filepath: data/01_raw/customers.csv

transactions_raw:
  type: pandas.CSVDataSet
  filepath: data/01_raw/transactions.csv

# Processed data
features:
  type: pandas.CSVDataSet
  filepath: data/03_primary/features.csv
  save_args:
    index: false

target:
  type: pandas.CSVDataSet
  filepath: data/03_primary/target.csv
  save_args:
    index: false

# Models
rf_model:
  type: pickle.PickleDataSet
  filepath: data/06_models/rf_model.pkl

xgb_model:
  type: pickle.PickleDataSet
  filepath: data/06_models/xgb_model.pkl

best_model:
  type: pickle.PickleDataSet
  filepath: data/06_models/best_model.pkl

# Reports
ml_report:
  type: text.TextDataSet
  filepath: data/08_reporting/ml_report.txt
```

### 2. Parameters Configuration (`conf/base/parameters.yml`)
```yaml
model_options:
  max_depth: 15
  random_state: 42

data_processing:
  test_size: 0.2
  random_state: 42
```

### 3. Data Engineering Pipeline (`src/ml_kedro_project/pipelines/data_engineering/nodes.py`)
```python
import pandas as pd
from typing import Dict, Any

def load_and_clean_data(customers: pd.DataFrame, transactions: pd.DataFrame) -> Dict[str, Any]:
    """Load multiple data sources and clean them."""
    # Clean and merge
    data = customers.merge(transactions, on="customer_id")
    data = data.dropna()

    # Split features and target
    X = data.drop(['target'], axis=1)
    y = data['target']

    return {
        "features": X,
        "target": y,
        "n_samples": len(X),
        "n_features": X.shape[1]
    }
```

### 4. Data Engineering Pipeline Definition (`src/ml_kedro_project/pipelines/data_engineering/pipeline.py`)
```python
from kedro.pipeline import Pipeline, node
from .nodes import load_and_clean_data

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=load_and_clean_data,
            inputs=["customers_raw", "transactions_raw"],
            outputs=["features", "target", "data_stats"],
            name="data_preparation_node"
        )
    ])
```

### 5. Data Science Nodes (`src/ml_kedro_project/pipelines/data_science/nodes.py`)
```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from typing import Dict, Any

def train_random_forest(features: pd.DataFrame, target: pd.Series, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Train Random Forest model."""
    model = RandomForestClassifier(
        max_depth=parameters["model_options"]["max_depth"],
        random_state=parameters["model_options"]["random_state"]
    )
    model.fit(features, target)

    accuracy = model.score(features, target)
    return {
        "model": model,
        "model_type": "RandomForest",
        "accuracy": accuracy
    }

def train_xgboost(features: pd.DataFrame, target: pd.Series, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Train XGBoost model."""
    model = xgb.XGBClassifier(
        max_depth=parameters["model_options"]["max_depth"],
        random_state=parameters["model_options"]["random_state"]
    )
    model.fit(features, target)

    accuracy = model.score(features, target)
    return {
        "model": model,
        "model_type": "XGBoost",
        "accuracy": accuracy
    }

def select_best_model(rf_results: Dict[str, Any], xgb_results: Dict[str, Any]) -> Dict[str, Any]:
    """Compare models and select the best one."""
    if rf_results["accuracy"] > xgb_results["accuracy"]:
        return {
            "best_model": rf_results["model"],
            "best_results": rf_results
        }
    else:
        return {
            "best_model": xgb_results["model"],
            "best_results": xgb_results
        }
```

### 6. Data Science Pipeline Definition (`src/ml_kedro_project/pipelines/data_science/pipeline.py`)
```python
from kedro.pipeline import Pipeline, node
from .nodes import train_random_forest, train_xgboost, select_best_model

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        # Parallel model training (Kedro handles this automatically)
        node(
            func=train_random_forest,
            inputs=["features", "target", "parameters"],
            outputs="rf_results",
            name="train_rf_node"
        ),
        node(
            func=train_xgboost,
            inputs=["features", "target", "parameters"],
            outputs="xgb_results",
            name="train_xgb_node"
        ),

        # Model selection
        node(
            func=select_best_model,
            inputs=["rf_results", "xgb_results"],
            outputs=["best_model", "model_selection_results"],
            name="select_best_model_node"
        )
    ])
```

### 7. Pipeline Registry (`src/ml_kedro_project/pipeline_registry.py`)
```python
from typing import Dict
from kedro.pipeline import Pipeline
from ml_kedro_project.pipelines import data_engineering, data_science, reporting

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines."""
    de_pipeline = data_engineering.create_pipeline()
    ds_pipeline = data_science.create_pipeline()
    reporting_pipeline = reporting.create_pipeline()

    return {
        "__default__": de_pipeline + ds_pipeline + reporting_pipeline,
        "de": de_pipeline,
        "ds": ds_pipeline,
        "reporting": reporting_pipeline
    }
```

### 8. Running the Pipeline
```bash
# First, create project structure
kedro new --starter=pandas-iris ml-kedro-project

# Then implement all the above files

# Finally, run
kedro run
```

---

## 📊 Side-by-Side Comparison

| Aspect | Runnable | Kedro |
|--------|----------|-------|
| **Files needed** | 1 Python file | 15+ files across directories |
| **Configuration** | Optional (inline parameters) | Mandatory YAML files |
| **Learning curve** | 5 minutes | 2-3 days |
| **Time to first run** | Immediate | Hours (setup + learning) |
| **Your existing code** | Zero changes needed | Must refactor to fit patterns |
| **Parallel execution** | `Parallel()` - explicit and clear | Automatic but less obvious |
| **Local to production** | Same code, different config | Requires external orchestrators |
| **Debugging** | Standard Python debugging | Kedro-specific tooling needed |

## 💡 The Numbers Don't Lie

**Runnable Implementation:**
- ✅ **1 file** (130 lines including logic)
- ✅ **0 YAML files**
- ✅ **5 minutes** to working pipeline
- ✅ **Zero refactoring** of existing functions

**Kedro Implementation:**
- ❌ **15+ files** across complex directory structure
- ❌ **3 YAML configuration files**
- ❌ **Hours to days** for setup and implementation
- ❌ **Complete refactoring** required

## 🎯 When to Choose Kedro

**Kedro excels when you need:**

- **Large team standardization** - Enforces consistent project structure across many data scientists
- **Complex data governance** - Rich catalog with data lineage, versioning, and validation
- **Enterprise compliance** - Established patterns for regulated industries
- **Data engineering focus** - Heavy ETL with many data transformation steps
- **Kedro ecosystem** - Specific plugins (kedro-airflow, kedro-mlflow, etc.)

!!! tip "Honest assessment"

    Kedro is powerful but comes with significant overhead. If you're a team of 1-5 people building ML pipelines, Runnable will make you dramatically more productive. If you're a 50-person data science organization needing strict governance, Kedro's structure might be worth the complexity cost.

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

**Next:** See how Runnable compares to other orchestration tools like [Airflow](airflow.md) and [Prefect](prefect.md).
