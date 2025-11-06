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

## 📊 Comprehensive Feature Comparison

### Core Workflow Features

| Feature | Runnable Strength | Kedro Strength |
|---------|-------------------|----------------|
| **Pipeline Definition** | Single Python file - minimal setup | Structured project layout - team consistency |
| **Task Types** | Python, Notebooks, Shell, Stubs - mixed execution | Python nodes with rich abstractions |
| **Parameter Passing** | Automatic flow between tasks | Explicit catalog + parameters - full control |
| **Parallel Execution** | `Parallel()` - explicit branching control | Automatic dependency resolution |
| **Conditional Logic** | Native `Conditional()` with branches | *Manual implementation required* |
| **Map/Reduce** | Native `Map()` with custom reducers | *Manual implementation required* |
| **Failure Handling** | `on_failure` pipeline redirection | Hook-based custom handling |

### Data Management

| Feature | Runnable Strength | Kedro Strength |
|---------|-------------------|----------------|
| **File Handling** | Simple `Catalog(put/get)` - minimal config | Rich catalog.yml definitions - fine control |
| **Data Versioning** | Content-based MD5 hashing - true change detection | *Timestamp-based - can miss identical content* |
| **Data Validation** | *Manual in your functions* | Built-in dataset validation |
| **Storage Backends** | File, S3, Minio via plugins | 20+ built-in dataset types |
| **Data Lineage** | Automatic via run logs | Kedro-viz visualization |
| **Schema Evolution** | *Manual handling required* | Built-in schema validation |

### Development Experience

| Feature | Runnable Strength | Kedro Strength |
|---------|-------------------|----------------|
| **Learning Curve** | 5-10 minutes - immediate productivity | *2-3 days setup and learning* |
| **Code Changes** | Zero refactoring - use existing functions | *Must fit Kedro patterns* |
| **IDE Support** | Standard Python tooling | Kedro-specific extensions |
| **Debugging** | Native Python debugging | *Framework-specific debugging needed* |
| **Testing** | Standard pytest + Stubs | Rich testing utilities |
| **Hot Reloading** | Standard Python imports | *Kedro session management required* |

### Production & Operations

| Feature | Runnable Strength | Kedro Strength |
|---------|-------------------|----------------|
| **Local Development** | Instant - just run Python | *Kedro CLI required* |
| **Environment Portability** | Same code everywhere | *Requires deployment strategy* |
| **Container Support** | Built-in via config | *Manual Dockerization needed* |
| **Kubernetes** | Native executor | *Requires external tools* |
| **Argo Workflows** | Native executor | *Requires kedro-argo plugin* |
| **Monitoring** | *Basic run logs only* | Rich hooks ecosystem |
| **Experiment Tracking** | *Basic via run IDs* | MLflow integration |

### Reproducibility & Governance

| Feature | Runnable Strength | Kedro Strength |
|---------|-------------------|----------------|
| **Run Tracking** | Automatic unique IDs + metadata | *Manual experiment setup required* |
| **Code Versioning** | Automatic git commit tracking | *Manual version management* |
| **Parameter Tracking** | Full parameter flow capture | *Explicit parameter logging required* |
| **Audit Trails** | Complete JSON logs - zero setup | *Custom hook implementation required* |
| **Data Governance** | *Basic file tracking only* | Rich catalog governance features |
| **Compliance** | *Basic logging - custom implementation needed* | *No built-in regulatory compliance* |

### Ecosystem & Integration

| Feature | Runnable Strength | Kedro Strength |
|---------|-------------------|----------------|
| **Plugin System** | Entry points architecture | Mature plugin ecosystem |
| **MLOps Integration** | *Basic via shell tasks* | Native MLflow, Airflow integrations |
| **Visualization** | *Basic via logs only* | Advanced kedro-viz pipeline visualization |
| **Community** | *Smaller, growing community* | Large, established community |
| **Documentation** | *Growing documentation* | Comprehensive documentation |
| **Enterprise Support** | *Limited enterprise features* | AstraZeneca backing and enterprise focus |

## 🟢 What Runnable Excels At

### **Unique Advantages**

1. **Zero Framework Lock-in**
   - Your functions remain pure Python
   - No refactoring of existing code required
   - Easy to extract logic if you change frameworks

2. **Environment Portability**
   ```python
   # Same exact code
   pipeline.execute()                     # Local
   pipeline.execute(config="k8s.yaml")   # Kubernetes
   pipeline.execute(config="argo.yaml")   # Argo Workflows
   ```

3. **Advanced Workflow Patterns**
   - Native `Parallel()`, `Map()`, `Conditional()`
   - Complex branching and merging logic
   - Custom reducers and failure handling

4. **Instant Productivity**
   - 5-minute setup vs days
   - No learning curve for existing Python developers
   - Immediate execution without CLI tools

5. **Mixed Task Types**
   ```python
   Pipeline(steps=[
       PythonTask(function=preprocess),      # Python
       NotebookTask(notebook="explore.ipynb"), # Jupyter
       ShellTask(command="./deploy.sh")      # Shell
   ])
   ```

6. **Automatic Reproducibility**
   - Every run gets unique ID and complete metadata
   - Git commit tracking without setup
   - Parameter flow capture built-in

7. **Superior Data Versioning**
   ```json
   {
     "name": "model_data.csv",
     "data_hash": "8650858600ce25b35e978ecb162414d9",
     "catalog_relative_path": "run-123/model_data.csv"
   }
   ```
   - **Content-based hashing**: Detects actual data changes, not just timestamps
   - **Smart large file handling**: Uses last 5MB for performance on big datasets
   - **True deduplication**: Identical content = same hash, regardless of when created

### **Why Content Hashing Beats Timestamps**

**Runnable's Approach:**
```python
# Same data = same hash, always
file_v1 = "customers.csv"  # Hash: abc123
file_v2 = "customers_copy.csv"  # Same content → Hash: abc123
file_v3 = "customers_modified.csv"  # Different content → Hash: def456
```

**Problems with Timestamp Versioning:**
- ❌ **False changes**: File touched but content unchanged → new "version"
- ❌ **Missed duplicates**: Same content, different timestamps → separate versions
- ❌ **Clock skew issues**: Different machines = timestamp inconsistencies
- ❌ **No content verification**: Can't detect corruption or partial writes

**Runnable's Content Hash Advantages:**
- ✅ **True change detection**: Only creates new version when content actually changes
- ✅ **Automatic deduplication**: Identical files share same hash regardless of name/time
- ✅ **Data integrity**: Hash mismatch immediately reveals corruption
- ✅ **Performance optimized**: Last 5MB sampling for large files (TBs)

## 🎯 Compliance & Governance Reality Check

### **What Both Tools Actually Provide:**

**Runnable's Compliance Foundation:**
- ✅ **Automatic audit trails**: Complete JSON logs with unique run IDs
- ✅ **Data integrity**: Content-based hashing detects any changes
- ✅ **Reproducibility**: Git commit tracking and parameter capture
- ❌ **Regulatory frameworks**: No SOX/GDPR/HIPAA built-in
- ❌ **Access control**: No user authentication or permissions
- ❌ **Data classification**: No PII or sensitive data labeling

**Kedro's Compliance Foundation:**
- ✅ **Data governance patterns**: Rich metadata and catalog structure
- ✅ **Extensibility**: Hooks where compliance can be built
- ✅ **Visualization**: kedro-viz helps explain data flows to auditors
- ❌ **Regulatory frameworks**: No SOX/GDPR/HIPAA built-in
- ❌ **Access control**: No user authentication or permissions
- ❌ **Compliance automation**: Requires custom implementation

### **The Truth About Enterprise Compliance:**

Neither tool provides **turnkey regulatory compliance**. For actual compliance, organizations need:
- External governance platforms (Collibra, Alation)
- Identity management (Active Directory, LDAP)
- Specialized compliance software
- Custom implementation on top of the ML framework

**Kedro's advantage:** More enterprises have built compliance systems on top of it
**Runnable's advantage:** Provides better audit trail foundations automatically

## 🔄 What Runnable Lacks (Kedro's Strengths)

### **Where Kedro is Superior**

1. **Data Catalog Sophistication**
   - 20+ built-in dataset types with validation
   - Rich dataset abstractions and transformations
   - Schema evolution and data quality checks
   - *Note: Timestamp versioning is less robust than content hashing*

2. **Enterprise Data Governance**
   - Rich catalog metadata capabilities
   - Data ownership and classification patterns
   - Enterprise-grade data lineage via kedro-viz
   - *Note: No built-in regulatory compliance - requires custom implementation*

3. **MLOps Ecosystem**
   - Native MLflow integration
   - Airflow deployment patterns
   - Established ML workflow conventions

4. **Visualization & Monitoring**
   - Advanced pipeline visualization (kedro-viz)
   - Rich experiment tracking
   - Sophisticated monitoring hooks

5. **Team Collaboration**
   - Opinionated project structure enforces consistency
   - Standardized patterns across large teams
   - Better for 50+ person data science organizations

6. **Plugin Ecosystem**
   - Mature plugin architecture
   - Many contributed integrations
   - Active community contributions

## 🎯 Honest Trade-off Analysis

### **Choose Runnable When:**
- **Philosophy**: Prefer autonomy and flexibility over imposed structure
- **Existing codebase**: Want to reuse functions without refactoring
- **Onboarding priority**: Need new team members productive in minutes, not days
- **Deployment needs**: Require multi-environment portability (local → cloud)
- **Workflow complexity**: Need advanced patterns (parallel, conditional, map-reduce)
- **Development speed**: Value immediate productivity over upfront structure

### **Choose Kedro When:**
- **Philosophy**: Prefer standardized patterns and central governance
- **Data complexity**: Heavy ETL pipelines with extensive validation needs
- **Existing investment**: Already using Kedro or have Kedro expertise
- **MLOps ecosystem**: Need established integrations (MLflow, Airflow)
- **Data governance**: Require rich metadata management and catalog features
- **Long-term structure**: Willing to invest setup time for enforced consistency

### **The Bottom Line**

**The choice isn't about team size - it's about philosophy and priorities.**

**Runnable optimizes for:**
- Developer productivity and immediate value
- Flexibility and autonomy over imposed structure
- Leveraging existing code and skills
- Seamless multi-environment deployment

**Kedro optimizes for:**
- Structured patterns and enforced consistency
- Rich data governance and metadata management
- Established MLOps ecosystem integrations
- Central coordination across complex data workflows

Both can work at any scale - the question is what trade-offs align with your organization's values and constraints.

## 📊 Decision Matrix: Focus on What Actually Matters

| Your Priority | Recommended Choice | Why |
|--------------|-------------------|-----|
| **Speed to market** | 🟢 **Runnable** | Immediate productivity, no setup overhead |
| **Existing Python codebase** | 🟢 **Runnable** | Zero refactoring, works with current functions |
| **Multi-environment deployment** | 🟢 **Runnable** | Same code runs local → container → K8s → Argo |
| **Team onboarding efficiency** | 🟢 **Runnable** | 5 minutes vs days to productivity |
| **Advanced workflow patterns** | 🟢 **Runnable** | Native parallel, conditional, map-reduce |
| **Heavy ETL with validation** | 🟡 **Kedro** | Rich dataset types and validation features |
| **Established MLOps ecosystem** | 🟡 **Kedro** | Native MLflow, Airflow, kedro-viz integrations |
| **Central governance requirements** | 🟡 **Kedro** | Structured project patterns and rich metadata |
| **Existing Kedro investment** | 🟡 **Kedro** | Leverage current expertise and infrastructure |

### **Scaling Considerations Revisited**

**Large Team Scenarios:**

**Runnable might scale better when:**
- **Onboarding efficiency matters**: 100 developers × 5 minutes vs 100 × 3 days
- **Team autonomy preferred**: Each team uses existing functions without central coordination
- **Diverse skill levels**: Standard Python is universally understood
- **Rapid iteration needed**: No framework constraints slow development

**Kedro might scale better when:**
- **Standardization critical**: Need enforced consistency across many teams
- **Complex data governance**: Rich metadata management becomes essential
- **Established processes**: Teams already invested in Kedro patterns
- **Central control preferred**: Want coordinated approach across organization

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
