# Adding Flexibility

Now let's solve another major problem: hardcoded parameters. We'll make our ML function configurable so you can run different experiments without touching any code.

## The Problem with Hardcoded Parameters

In Chapter 2, our function still had hardcoded values:

```python
# Fixed values - need code changes for experiments
preprocessed = preprocess_data(df, test_size=0.2, random_state=42)
model_data = train_model(preprocessed, n_estimators=100, random_state=42)
```

Want to try `n_estimators=200`? Edit the code. Different train/test split? Edit the code. This doesn't scale for experimentation.

## The Solution: Parameterized Functions

Let's create a flexible version that accepts parameters:

```python title="examples/tutorials/getting-started/functions_parameterized.py"
def train_ml_model_flexible(
    data_path="data.csv",
    test_size=0.2,
    n_estimators=100,
    random_state=42,
    model_path="model.pkl",
    results_path="results.json"
):
    """Same ML logic, now configurable!"""
    print("Loading data...")
    df = load_data(data_path)

    print("Preprocessing...")
    preprocessed = preprocess_data(df, test_size=test_size, random_state=random_state)

    print(f"Training model with {n_estimators} estimators...")
    model_data = train_model(preprocessed, n_estimators=n_estimators, random_state=random_state)

    # ... rest unchanged but uses parameters
```

## Running with Parameters

Now you can run different experiments without changing code:

### 🌍 **Environment Variables**

```bash
# Default parameters
uv run examples/tutorials/getting-started/03_adding_flexibility.py

# Large forest experiment
RUNNABLE_PRM_n_estimators=200 uv run examples/tutorials/getting-started/03_adding_flexibility.py

# Different train/test split
RUNNABLE_PRM_test_size=0.3 RUNNABLE_PRM_n_estimators=150 uv run examples/tutorials/getting-started/03_adding_flexibility.py
```

### 📁 **Configuration Files**

Create experiment configurations:

```yaml title="examples/tutorials/getting-started/experiment_configs/basic.yaml"
test_size: 0.2
n_estimators: 50
random_state: 42
model_path: "models/basic_model.pkl"
results_path: "results/basic_results.json"
```

```yaml title="examples/tutorials/getting-started/experiment_configs/large_forest.yaml"
test_size: 0.25
n_estimators: 200
random_state: 123
model_path: "models/large_forest.pkl"
results_path: "results/large_forest_results.json"
```

Run different experiments:

```bash
# Basic experiment
uv run examples/tutorials/getting-started/03_adding_flexibility.py --parameters-file experiment_configs/basic.yaml

# Large forest experiment
uv run examples/tutorials/getting-started/03_adding_flexibility.py --parameters-file experiment_configs/large_forest.yaml
```

## Parameter Precedence

Runnable handles parameter conflicts intelligently:

1. **Environment variables** (highest priority): `RUNNABLE_PRM_n_estimators=300`
2. **Command line config**: `--parameters-file config.yaml`
3. **Function defaults** (lowest priority): What you defined in the function signature

This means you can have a base configuration file but override specific values with environment variables.

## What You Get Now

### 🧪 **Easy Experimentation**

- Test different hyperparameters instantly
- Compare multiple approaches without code changes
- Save each experiment configuration for reproducibility

### 📊 **Automatic Experiment Tracking**

Every run gets logged with the exact parameters used:

```bash
ls .runnable/run-log-store/
# Each timestamped directory contains the parameters for that run
```

### 🔄 **Reproducible Experiments**

Want to recreate that great result from last week? Just rerun with the same config file.

### 🎯 **Clean Separation**

- **Your ML logic**: Stays in the function, unchanged
- **Experiment configuration**: Lives in config files or environment variables
- **Execution tracking**: Handled automatically by Runnable

## Try It Yourself

Run these experiments and watch how each gets tracked separately:

```bash
cd examples/tutorials/getting-started

# Experiment 1: Default
uv run 03_adding_flexibility.py

# Experiment 2: Large forest
RUNNABLE_PRM_n_estimators=200 uv run 03_adding_flexibility.py

# Experiment 3: From config file
uv run 03_adding_flexibility.py --parameters-file experiment_configs/large_forest.yaml

# Check the logs - each run preserved with its parameters
ls .run_log_store/
```

## Compare: Before vs After

**Before:**

- ❌ Parameters hardcoded in functions
- ❌ Code changes needed for experiments
- ❌ Hard to track which parameters produced which results

**After:**

- ✅ Functions accept parameters with sensible defaults
- ✅ Experiments configurable via environment or config files
- ✅ Every run logged with exact parameters used
- ✅ Easy to reproduce any experiment

**Next:** We'll break our monolithic function into a proper multi-step ML pipeline.

---

**Next:** [Connecting the Workflow](04-connecting-workflow.md) - Multi-step ML pipeline with automatic data flow
