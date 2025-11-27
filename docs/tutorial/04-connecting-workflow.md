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

## How Data Flows Automatically

Notice something magical: we didn't write any glue code! Runnable automatically connects the steps:

1. **`load_data()`** returns a DataFrame
2. **`preprocess_data(df)`** - gets the DataFrame automatically (parameter name matches!)
3. **`train_model(preprocessed_data)`** - gets preprocessing results automatically
4. **`evaluate_model(model_data, preprocessed_data)`** - gets both model and data automatically

**The secret:** Parameter names in your functions determine data flow. If `train_model()` expects a parameter called `preprocessed_data`, and a previous step returns something called `preprocessed_data`, they get connected automatically.

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
# Check what the preprocessing step produced
ls .runnable/
```

### 🛠️ **Better Debugging**

If training fails, you don't lose your preprocessing work. You can debug just the training step.

### 📊 **Individual Step Tracking**

See timing and resource usage for each step, helping identify bottlenecks.

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
- ✅ Automatic data flow between steps

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
