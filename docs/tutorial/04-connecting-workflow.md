# Connecting the Workflow

So far we've been treating ML training as one big function. In reality, ML workflows have distinct steps: data loading, preprocessing, training, and evaluation. Let's break our monolithic function into a proper pipeline.

## Why Break It Up?

Our current approach has limitations:

```python
def train_ml_model_flexible():
    # All steps in one function
    df = load_data()           # Step 1
    preprocessed = preprocess_data()  # Step 2
    model = train_model()      # Step 3
    results = evaluate_model() # Step 4
    return results
```

**Problems:**

- If training fails, you lose preprocessing work
- Hard to debug specific steps
- Can't reuse preprocessing for different models
- No visibility into step-by-step progress

## The Solution: Pipeline with Tasks

Let's use the individual functions we already have and connect them as a pipeline:

```python title="examples/tutorials/getting-started/04_connecting_workflow.py"
from runnable import Pipeline, PythonTask, pickled
from functions import load_data, preprocess_data, train_model, evaluate_model

def main():
    pipeline = Pipeline(steps=[
        PythonTask(
            function=load_data,
            name="load_data",
            returns=[pickled("df")]
        ),
        PythonTask(
            function=preprocess_data,
            name="preprocess",
            returns=[pickled("preprocessed_data")]
        ),
        PythonTask(
            function=train_model,
            name="train",
            returns=[pickled("model_data")]
        ),
        PythonTask(
            function=evaluate_model,
            name="evaluate",
            returns=[pickled("evaluation_results")]
        )
    ])

    pipeline.execute()
    return pipeline
```

**Try it:**

```bash
uv run examples/tutorials/getting-started/04_connecting_workflow.py
```

## How Data Flow Works Through Run Log

Runnable connects pipeline steps through **run log parameter management**:

1. **`returns=[pickled("df")]`** → Run log stores parameter "df" (binary data in catalog)
2. **`preprocess_data(df, ...)`** → Run log provides "df" parameter (fetches from catalog if pickled)
3. **`train_model(preprocessed_data)`** → Run log provides "preprocessed_data" parameter
4. **`evaluate_model(model_data, preprocessed_data)`** → Run log provides both parameters

**The pattern:** Function parameter names must match the names in previous tasks' `returns` declarations because that's how the run log maps parameters.

## What You Get with Pipelines

### ⚡ **Step-by-Step Execution**

Each step runs individually and you can see progress:

```
load_data: ✅ Completed in 0.1s
preprocess: ✅ Completed in 0.3s
train: ✅ Completed in 2.4s
evaluate: ✅ Completed in 0.2s
```

### 🔍 **Intermediate Results Preserved**

Each step's output is saved. You can inspect intermediate results without rerunning expensive steps:

```bash
# Check run log parameter tracking
ls .run_log_store/

# Check pickled data storage
ls .catalog/
```

### 🛠️ **Better Debugging**

If training fails, you don't lose your preprocessing work. You can debug just the training step.

### 📊 **Individual Step Tracking**

See timing and resource usage for each step, helping identify bottlenecks.

## 🔗 Understanding Parameter Naming

For data flow to work correctly, follow this naming pattern:

```python
# Step 1: Function returns something, run log tracks as "df"
PythonTask(function=load_data, returns=[pickled("df")])

# Step 2: Function parameter "df" matches run log parameter "df"
def preprocess_data(df, test_size=0.2):  # Gets "df" from run log
    return preprocessed_data

# Step 3: Save as "preprocessed_data" in run log
PythonTask(function=preprocess_data, returns=[pickled("preprocessed_data")])

# Step 4: Parameter names match run log parameter names
def train_model(preprocessed_data, n_estimators=100):  # Gets from run log
def evaluate_model(model_data, preprocessed_data):    # Gets both from run log
```

**Key Rule:** Parameter names in later functions must exactly match the names in earlier `returns` declarations.

## 🚨 Common Parameter Issues

**Problem**: Parameter name doesn't match returns name
```python
# Won't work - name mismatch!
PythonTask(function=load_data, returns=[pickled("dataframe")])
def preprocess_data(df, test_size=0.2):  # Run log has "dataframe", expects "df"
```

**Solution**: Make parameter names match returns names
```python
# Works - run log has "df", function expects "df"
PythonTask(function=load_data, returns=[pickled("df")])
def preprocess_data(df, test_size=0.2):  # Gets "df" from run log
```

**Debug Tip**: Check run log files in `.run_log_store/` to see actual parameter names stored.

## Advanced: Parameters in Pipelines

You can still use parameters, but now at the step level:

```python
# Add parameters to specific steps
pipeline = Pipeline(steps=[
    PythonTask(function=load_data, name="load_data", returns=[pickled("df")]),
    PythonTask(function=preprocess_data, name="preprocess", returns=[pickled("preprocessed_data")]),
    PythonTask(function=train_model, name="train", returns=[pickled("model_data")]),
    PythonTask(function=evaluate_model, name="evaluate", returns=[pickled("evaluation_results")])
])

# Parameters still work the same way
# RUNNABLE_PRM_test_size=0.3 uv run 04_connecting_workflow.py
```

Parameters get passed to the appropriate functions based on their parameter names.

## Compare: Monolithic vs Pipeline

**Monolithic Function (Chapters 1-3):**

- ❌ All-or-nothing execution
- ❌ Hard to debug failed steps
- ❌ Expensive to rerun everything
- ❌ No intermediate result visibility

**Pipeline (Chapter 4):**

- ✅ Step-by-step execution with progress
- ✅ Intermediate results preserved
- ✅ Resume from failed steps
- ✅ Better debugging and development
- ✅ Parameter-based data flow between steps

## Your Functions Didn't Change

Notice that we're using the exact same functions from earlier:

- `load_data()`
- `preprocess_data()`
- `train_model()`
- `evaluate_model()`

**No refactoring required.** Runnable works with your existing functions - you just organize them into steps.

## What's Next?

We have a great pipeline, but we're still dealing with everything in memory. What about large datasets that don't fit in RAM? Or sharing intermediate results with teammates?

**Next chapter:** We'll add efficient data management for large-scale ML workflows.

---

**Next:** [Handling Large Datasets](05-handling-datasets.md) - Efficient storage and retrieval of data artifacts
