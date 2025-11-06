# 🆚 Runnable vs Metaflow: Simplicity vs Scale

Both Runnable and Metaflow solve ML pipeline orchestration, but with different philosophies. Metaflow (created by Netflix) focuses on data science at massive scale, while Runnable prioritizes simplicity and immediate productivity.

## 📊 The Example: ML Training Pipeline

Let's build the same machine learning pipeline that:

1. **Loads and cleans data** from multiple sources
2. **Trains two models** in parallel (Random Forest + XGBoost)
3. **Evaluates models** and selects the best one
4. **Generates a report** with results

This will show how each tool approaches the same real-world ML workflow.

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
    # Load from S3 or central storage
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
    data_prep = PythonTask(
        function=load_and_clean_data,
        returns=["n_samples", "n_features"],
        catalog=Catalog(put=["features.csv", "target.csv"]),
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
    result = pipeline.execute(parameters_file="parameters.yaml")
    print(f"Pipeline completed! Run ID: {result.run_id}")
```

**Run it:**
```bash
uv run ml_pipeline.py
```

**That's it.** One file, runs immediately, works locally and in production.

!!! note "Domain Code Separation"

    Your **business logic functions remain pure Python** - no framework dependencies. You could import these from completely separate modules:

    ```python
    # Import from your existing codebase
    from my_ml_library.preprocessing import load_and_clean_data
    from my_ml_library.models import train_random_forest, train_xgboost
    from my_ml_library.evaluation import select_best_model
    from my_ml_library.reporting import generate_report

    # Just wrap them with Runnable - no changes to your domain code needed
    ```

---

## 🔵 The Metaflow Way: Netflix-Scale Architecture

Metaflow requires decorator-based approach and class structure:

```python title="ml_metaflow.py"
from metaflow import FlowSpec, step, Parameter, parallel, catch
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import joblib

class MLTrainingFlow(FlowSpec):
    """ML Training Pipeline using Metaflow"""

    max_depth = Parameter('max_depth', help='Max depth for models', default=15)

    @step
    def start(self):
        """Initialize the flow"""
        print("Starting ML Training Pipeline")
        self.next(self.load_data)

    @step
    def load_data(self):
        """Load and clean data"""
        # Load from S3 or central storage
        customers = pd.read_csv("s3://bucket/raw-data/customers.csv")
        transactions = pd.read_csv("s3://bucket/raw-data/transactions.csv")

        # Clean and merge
        data = customers.merge(transactions, on="customer_id")
        data = data.dropna()

        # Split features and target
        X = data.drop(['target'], axis=1)
        y = data['target']

        # Store as flow artifacts
        self.X = X
        self.y = y
        self.n_samples = len(X)
        self.n_features = X.shape[1]

        print(f"Loaded {self.n_samples} samples with {self.n_features} features")
        self.next(self.train_models)

    @parallel
    @step
    def train_models(self):
        """Train models in parallel"""
        self.model_type = self.input

        if self.model_type == 'RandomForest':
            model = RandomForestClassifier(max_depth=self.max_depth, random_state=42)
        elif self.model_type == 'XGBoost':
            model = xgb.XGBClassifier(max_depth=self.max_depth, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        # Train model
        model.fit(self.X, self.y)

        # Calculate accuracy
        self.accuracy = model.score(self.X, self.y)
        self.trained_model = model

        print(f"{self.model_type} accuracy: {self.accuracy:.3f}")
        self.next(self.select_best_model)

    @step
    def select_best_model(self, inputs):
        """Select the best model from parallel training"""
        # Merge results from parallel branches
        best_model = None
        best_accuracy = 0
        best_type = None

        for input_flow in inputs:
            if input_flow.accuracy > best_accuracy:
                best_accuracy = input_flow.accuracy
                best_model = input_flow.trained_model
                best_type = input_flow.model_type

        # Store best model
        self.best_model = best_model
        self.best_model_type = best_type
        self.best_accuracy = best_accuracy

        # Save model to file
        joblib.dump(self.best_model, "best_model.pkl")

        print(f"Best model: {self.best_model_type} with accuracy: {self.best_accuracy:.3f}")

        # Pass data stats for reporting
        self.n_samples = inputs[0].n_samples
        self.n_features = inputs[0].n_features

        self.next(self.generate_report)

    @step
    def generate_report(self):
        """Generate final report"""
        report = f"""
        ML Pipeline Report
        ==================

        Data Statistics:
        - Samples: {self.n_samples}
        - Features: {self.n_features}

        Best Model: {self.best_model_type}
        Accuracy: {self.best_accuracy:.3f}

        Model saved as: best_model.pkl
        """

        with open("ml_report.txt", "w") as f:
            f.write(report)

        self.report_path = "ml_report.txt"
        print("Report generated successfully")
        self.next(self.end)

    @step
    def end(self):
        """End the flow"""
        print("ML Training Pipeline completed successfully!")
        print(f"Best model: {self.best_model_type}")
        print(f"Accuracy: {self.best_accuracy:.3f}")

if __name__ == '__main__':
    MLTrainingFlow()
```

**Run it:**
```bash
python ml_metaflow.py run --max_depth 15
```

**But wait, there's more setup needed:**

### **Required Infrastructure Setup:**
```bash
# Configure Metaflow for production
pip install metaflow

# Configure AWS integration (for production)
metaflow configure aws

# Set up S3 datastore
export METAFLOW_DATASTORE_SYSROOT_S3=s3://my-metaflow-bucket/

# Configure compute environment
export METAFLOW_BATCH_JOB_QUEUE=my-batch-queue
```

### **Production Execution:**
```bash
# Run on AWS Batch
python ml_metaflow.py run --with batch --max_depth 15

# Run on Kubernetes
python ml_metaflow.py run --with kubernetes --max_depth 15

# Schedule with Argo Workflows
python ml_metaflow.py argo-workflows create
```

---

## 📊 Comprehensive Feature Comparison

### Core Workflow Features

| Feature | Runnable Strength | Metaflow Strength |
|---------|-------------------|-------------------|
| **Pipeline Definition** | Single Python file - minimal setup | FlowSpec class with decorators - structured approach |
| **Task Types** | Python, Notebooks, Shell, Stubs - mixed execution | Python steps with rich metadata tracking |
| **Parameter Passing** | Automatic flow between tasks | Flow artifacts with automatic serialization |
| **Parallel Execution** | `Parallel()` - explicit branching control | `@parallel` decorator - automatic fan-out/join |
| **Conditional Logic** | Native `Conditional()` with branches | *Manual implementation in step logic* |
| **Map/Reduce** | Native `Map()` with custom reducers | `@parallel` with custom merge logic |
| **Failure Handling** | `on_failure` pipeline redirection | `@catch` decorator with retry logic |

### Data Management

| Feature | Runnable Strength | Metaflow Strength |
|---------|-------------------|-------------------|
| **File Handling** | Simple `Catalog(put/get)` - automatic file management | *Manual file I/O - no automatic file management* |
| **Data Versioning** | Content-based MD5 hashing - true change detection | Automatic versioning with Metaflow datastore |
| **Data Validation** | *Manual in your functions* | *Manual in step functions* |
| **Storage Backends** | File, S3, Minio via plugins | S3 datastore with automatic serialization |
| **Data Lineage** | Automatic via run logs | Rich lineage through Metaflow UI |
| **Large Data Handling** | *Manual optimization needed* | *Same Python/pandas memory limits - no magic scaling* |

### Development Experience

| Feature | Runnable Strength | Metaflow Strength |
|---------|-------------------|-------------------|
| **Learning Curve** | 5-10 minutes - immediate productivity | *1-2 hours - decorator patterns to learn* |
| **Code Changes** | Zero refactoring - use existing functions | *Must convert to FlowSpec class structure* |
| **IDE Support** | Standard Python tooling | Standard Python + Metaflow-specific features |
| **Debugging** | Native Python debugging | Metaflow client debugging tools |
| **Testing** | Standard pytest + Stubs | Metaflow testing utilities |
| **Hot Reloading** | Standard Python imports | *Flow restart required for changes* |

### Production & Operations

| Feature | Runnable Strength | Metaflow Strength |
|---------|-------------------|-------------------|
| **Local Development** | Instant - just run Python | Local execution with same decorator syntax |
| **Environment Portability** | Same code everywhere | Same FlowSpec runs local/AWS/K8s |
| **Container Support** | Built-in via config | *Requires container setup and configuration* |
| **Kubernetes** | Native executor | `--with kubernetes` execution |
| **AWS Integration** | *Manual S3 configuration* | Native AWS Batch, Step Functions integration |
| **Monitoring** | *Basic run logs only* | Rich Metaflow UI with execution visualization |
| **Experiment Tracking** | *Basic via run IDs* | Built-in experiment tracking and comparison |

### Reproducibility & Governance

| Feature | Runnable Strength | Metaflow Strength |
|---------|-------------------|-------------------|
| **Run Tracking** | Automatic unique IDs + metadata | Comprehensive run tracking with Metaflow client |
| **Code Versioning** | Automatic git commit tracking | Automatic code snapshots per execution |
| **Parameter Tracking** | Full parameter flow capture | Parameter and artifact tracking |
| **Audit Trails** | Complete JSON logs - zero setup | Rich execution history via Metaflow UI |
| **Data Governance** | *Basic file tracking only* | Enterprise-grade artifact management |
| **Compliance** | *Basic logging - custom implementation needed* | *No built-in regulatory compliance* |

### Ecosystem & Integration

| Feature | Runnable Strength | Metaflow Strength |
|---------|-------------------|-------------------|
| **Plugin System** | Entry points architecture | *Limited extensibility* |
| **MLOps Integration** | *Basic via shell tasks* | Native AWS ecosystem integration |
| **Visualization** | *Basic via logs only* | Rich Metaflow UI with DAG visualization |
| **Community** | *Smaller, growing community* | Strong Netflix backing and community |
| **Documentation** | *Growing documentation* | Comprehensive documentation and tutorials |
| **Enterprise Support** | *Limited enterprise features* | Netflix-scale proven, enterprise-ready |

## 🟢 What Runnable Excels At

### **Unique Advantages**

1. **Zero Framework Lock-in**
   - Your functions remain pure Python
   - No refactoring of existing code required
   - Easy to extract logic if you change frameworks

2. **Immediate Productivity**
   ```python
   # Existing function
   def my_analysis(data):
       return processed_data

   # Runnable wrapper - that's it!
   PythonTask(function=my_analysis)
   ```

3. **Advanced Workflow Patterns**
   - Native `Parallel()`, `Map()`, `Conditional()`
   - Complex branching and merging logic
   - Custom reducers and failure handling

4. **Mixed Task Types**
   ```python
   Pipeline(steps=[
       PythonTask(function=preprocess),           # Python
       NotebookTask(notebook="explore.ipynb"),    # Jupyter
       ShellTask(command="./deploy.sh")          # Shell
   ])
   ```

5. **Environment Portability**
   ```python
   # Same exact code
   pipeline.execute()                     # Local
   pipeline.execute(config="k8s.yaml")   # Kubernetes
   pipeline.execute(config="argo.yaml")  # Argo Workflows
   ```

## 🔵 What Runnable Lacks (Metaflow's Strengths)

### **Where Metaflow is Superior**

1. **Operational Scale Architecture**
   - Proven for managing thousands of concurrent workflows
   - Rich UI and monitoring for organizational-scale deployment
   - *Note: "Netflix scale" refers to workflow management, not data processing magic*

2. **Rich Execution Visualization**
   - Metaflow UI shows detailed execution graphs
   - Real-time monitoring of running workflows
   - Historical comparison of runs and experiments

3. **AWS Ecosystem Integration**
   - Native AWS Batch for compute scaling
   - Step Functions for serverless orchestration
   - S3 datastore with automatic optimization

4. **Flow Variable Management**
   - Automatic serialization of Python objects between steps
   - S3 datastore for storing flow artifacts
   - *Note: Does not manage external files - still need manual file I/O*

5. **Enterprise Experiment Tracking**
   - Rich metadata capture for every execution
   - Built-in A/B testing and experiment comparison
   - Production-grade monitoring and alerting

## 🔍 The "Netflix Scale" Reality Check

### **Marketing Claims vs Technical Reality**

**Common Misconception:** "Metaflow handles petabyte data because Netflix uses it"

**Technical Reality:** Metaflow doesn't overcome fundamental Python limitations:

```python
# This is still the same in Metaflow
@step
def process_data(self):
    df = pd.read_csv("huge_file.csv")  # Same pandas memory limits
    model = train_model(df)            # Same compute constraints
    self.result = model                # Still limited by Python memory
```

### **What "Netflix Scale" Actually Means**

| Scale Claim | Reality | Any Framework Can Do This |
|-------------|---------|---------------------------|
| **"Petabyte data processing"** | Many workflows processing chunks | ✅ Kubernetes parallel jobs |
| **"Thousands of data scientists"** | Operational management and UI | ✅ Any orchestrator + monitoring |
| **"Massive datasets"** | AWS Batch with big instances | ✅ Any framework + cloud compute |
| **"Enterprise grade"** | Rich UI and AWS integrations | ✅ Custom dashboards + cloud services |

### **Netflix's Real Architecture (Likely):**

```python
# Netflix runs thousands of these manageable workflows
@step
def analyze_user_segment(self):
    # Process 1M users (not all 200M at once)
    segment_users = get_segment(self.segment_id)  # Manageable chunk
    recommendations = calculate_recs(segment_users)
    self.results = recommendations
```

**The "scale" comes from:**
- ✅ **Operational management** of many concurrent workflows
- ✅ **Infrastructure automation** with AWS services
- ✅ **Workflow monitoring** across organization
- ❌ **NOT** magical data processing capabilities

### **Same Scale with Any Framework:**

```python
# Runnable achieving identical "Netflix scale"
for segment_id in all_user_segments:
    pipeline = Pipeline(steps=[
        PythonTask(function=process_user_segment,
                  parameters={"segment_id": segment_id})
    ])
    pipeline.execute(config="kubernetes.yaml")  # Same horizontal scaling
```

### **The File Management Reality:**

```python
# Metaflow still requires manual file handling
@step
def save_model(self):
    joblib.dump(self.model, "model.pkl")        # Manual file save
    with open("report.txt", "w") as f:          # Manual file write
        f.write(self.report)
    # No automatic file management like Runnable's Catalog
```

**Runnable provides better file management:**
```python
# Automatic file handling
PythonTask(
    function=save_model,
    catalog=Catalog(put=["model.pkl", "report.txt"])  # Automatic management
)
```

## 🎯 Honest Trade-off Analysis

### **Choose Runnable When:**
- **Philosophy**: Prefer simplicity and immediate productivity
- **Existing codebase**: Want to reuse functions without refactoring
- **Mixed workflows**: Need Python + notebooks + shell integration
- **Environment flexibility**: Require local → container → K8s → Argo portability
- **Advanced patterns**: Need conditional logic, map-reduce, complex branching
- **Learning curve**: Want team productive in minutes, not hours

### **Choose Metaflow When:**
- **AWS ecosystem**: Heavy investment in AWS services and infrastructure
- **Organizational scale**: Managing hundreds/thousands of concurrent workflows
- **Rich UI requirements**: Need detailed execution visualization and monitoring
- **Flow variable management**: Want automatic Python object serialization between steps
- **Established patterns**: Team already familiar with Metaflow decorator approach
- **AWS Batch integration**: Need seamless scaling with AWS compute services

### **The Bottom Line**

**The choice is about operational complexity and infrastructure investment.**

**Runnable optimizes for:**
- Developer productivity and immediate value
- Minimal infrastructure and setup requirements
- Multi-environment portability (not AWS-locked)
- Leveraging existing Python skills and code
- Automatic file management with simple catalog system

**Metaflow optimizes for:**
- Organizational workflow management at scale
- Rich execution monitoring and visualization
- AWS-native infrastructure integration
- Structured experiment tracking and comparison
- *Note: Doesn't provide magical data processing scale*

## 📊 Decision Matrix: Choose Based on Your Priorities

| Your Priority | Recommended Choice | Why |
|--------------|-------------------|-----|
| **Immediate productivity** | 🟢 **Runnable** | 5 minutes vs hours to first working pipeline |
| **Existing Python codebase** | 🟢 **Runnable** | Zero refactoring, works with current functions |
| **Multi-cloud deployment** | 🟢 **Runnable** | Same code runs anywhere vs AWS-specific |
| **Advanced workflow patterns** | 🟢 **Runnable** | Native conditional, map-reduce vs manual implementation |
| **Mixed task types** | 🟢 **Runnable** | Python + notebooks + shell vs Python only |
| **Organizational workflow management** | 🔵 **Metaflow** | Rich UI for managing many concurrent workflows |
| **AWS-native infrastructure** | 🔵 **Metaflow** | Native Batch, Step Functions, S3 integration |
| **Rich execution visualization** | 🔵 **Metaflow** | Detailed UI vs basic logs |
| **Enterprise experiment tracking** | 🔵 **Metaflow** | Built-in experiment comparison and monitoring |
| **Python object serialization** | 🔵 **Metaflow** | Automatic flow variable management between steps |

## 💡 The Scale Reality Check

**Runnable Implementation:**
- ✅ **1 file** (160 lines including logic)
- ✅ **0 infrastructure setup**
- ✅ **5 minutes** to working pipeline
- ✅ **Zero refactoring** of existing functions

**Metaflow Implementation:**
- ❌ **Class-based structure** (must refactor existing functions)
- ❌ **Infrastructure setup** required for production features
- ❌ **Learning curve** for decorator patterns and FlowSpec
- ❌ **AWS dependency** for full feature set

## 🚀 Try Both Yourself

**Test Runnable** (2 minutes):
```bash
pip install runnable
# Copy the Runnable example above
python ml_pipeline.py
```

**Test Metaflow** (30+ minutes):
```bash
pip install metaflow
# Setup AWS configuration
metaflow configure aws
# Copy the Metaflow example above
python ml_metaflow.py run
```

The complexity difference for getting started is significant, but Metaflow's Netflix-scale capabilities may justify the investment for the right use cases.

---

**Next:** See how Runnable compares to [Kedro](kedro.md) and other orchestration tools.
