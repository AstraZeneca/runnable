# Runnable

<figure markdown>
  ![Image title](assets/sport.png){ width="200" height="100"}
  <figcaption>Orchestrate your functions, notebooks, scripts anywhere!!</figcaption>
</figure>

<span style="font-size:0.75em;">
<a href="https://www.flaticon.com/free-icons/runner" title="runner icons">Runner icons created by Leremy - Flaticon</a>
</span>

Transform any Python function into a portable, trackable pipeline in seconds.

<hr style="border:2px dotted orange">

## Step 1: Install (10 seconds)

```bash
pip install runnable
```

## Step 2: Your Function (unchanged!)

```python
# Your existing function - zero changes needed
def analyze_sales():
    total_revenue = 50000
    best_product = "widgets"
    return total_revenue, best_product
```

## Step 3: Make It Runnable (2 lines)

```python
# Add 2 lines → Make it runnable everywhere
from runnable import PythonJob
PythonJob(function=analyze_sales).execute()
```

## 🎉 Success!

You just made your first function runnable and got:
- ✅ **Automatic tracking**: execution logs, timestamps, results saved
- ✅ **Reproducible runs**: full execution history and metadata
- ✅ **Environment portability**: runs the same on laptop, containers, Kubernetes

**Your code now runs anywhere without changes!**

---

## Want to See More?

### 🔧 Same Code, Different Parameters (2 minutes)

Change parameters without touching your code:

```python
# Function accepts parameters
def forecast_growth(revenue, growth_rate):
    return revenue * (1 + growth_rate) ** 3

from runnable import PythonJob
PythonJob(function=forecast_growth).execute()

# Run different scenarios anywhere:
# Local: RUNNABLE_PRM_revenue=100000 RUNNABLE_PRM_growth_rate=0.05 python forecast.py
# Container: same command, same results
# Kubernetes: same command, same results

# ✨ Every run tracked with parameters - reproducible everywhere
```

??? example "See complete parameter example"
    ```python title="examples/03-parameters/passing_parameters_python.py"
    --8<-- "examples/03-parameters/passing_parameters_python.py"
    ```

    **Try it:** `uv run examples/03-parameters/passing_parameters_python.py`

**Why bother?** No more "what parameters gave us those good results?" - tracked automatically across all environments.

---

### 🔗 Chain Functions, No Glue Code (3 minutes)

Build workflows that run anywhere unchanged:

```python
# Your existing functions
def load_customer_data():
    customers = {"count": 1500, "segments": ["premium", "standard"]}
    return customers

def analyze_segments(customer_data):  # Name matches = automatic connection
    analysis = {"premium_pct": 30, "growth_potential": "high"}
    return analysis

# What you used to write (glue code):
# customer_data = load_customer_data()
# analysis = analyze_segments(customer_data)

# What Runnable needs (same logic, no glue):
from runnable import Pipeline, PythonTask
Pipeline(steps=[
    PythonTask(function=load_customer_data, returns=["customer_data"]),
    PythonTask(function=analyze_segments, returns=["analysis"])
]).execute()

# Same pipeline runs unchanged on:
# • Your laptop (development)
# • Docker containers (testing)
# • Kubernetes (production)

# ✨ Write once, run anywhere - zero deployment rewrites
```

??? example "See complete pipeline example"
    ```python title="examples/02-sequential/simple.py"
    --8<-- "examples/02-sequential/simple.py"
    ```

    **Try it:** `uv run examples/02-sequential/simple.py`

**Why bother?** No more "it works locally but breaks in production" - same code, guaranteed same behavior.

---

### 🚀 Mix Python + Notebooks (5 minutes)

Different tools, portable workflows:

```python
# Python prepares data, notebook analyzes - works everywhere
def prepare_dataset():
    clean_data = {"sales": [100, 200, 300], "regions": ["north", "south"]}
    return clean_data

from runnable import Pipeline, PythonTask, NotebookTask
Pipeline(steps=[
    PythonTask(function=prepare_dataset, returns=["dataset"]),
    NotebookTask(notebook_path="deep_analysis.ipynb", returns=["insights"])
]).execute()

# This exact pipeline runs unchanged on:
# • Local Jupyter setup
# • Containerized environments
# • Cloud Kubernetes clusters

# ✨ No more environment setup headaches or "works on my machine"
```

??? example "See complete mixed workflow"
    ```python title="examples/02-sequential/traversal.py"
    --8<-- "examples/02-sequential/traversal.py"
    ```

    **Try it:** `uv run examples/02-sequential/traversal.py`

**Why bother?** Your entire data science workflow becomes truly portable - no environment-specific rewrites.

---

## What's Next?

You've seen how Runnable transforms your code for portability and tracking. Ready to go deeper?

**🎯 Master the Concepts** → [Jobs vs Pipelines](concepts/building-blocks/jobs-vs-pipelines.md)
Learn when to use single jobs vs multi-step pipelines

**📊 Handle Your Data** → [Task Types](concepts/building-blocks/task-types.md)
Work with returns, parameters, and different data types

**⚡ See Real Examples** → [Usage Examples](usage.md)
Browse practical patterns and real-world scenarios

**🚀 Deploy Anywhere** → [Production Guide](configurations/overview.md)
Scale from laptop to containers to Kubernetes

**🔍 Compare Alternatives** → [Compare Tools](comparisons/kedro.md)
See how Runnable compares to Kedro, Metaflow, and other orchestration tools

---

## Why Choose Runnable?

<div class="grid cards" markdown>

-   :material-clock-fast:{ .lg .middle } __Easy to adopt, its mostly your code__

    ---

    Your application code remains as it is. Runnable exists outside of it.

    - No API's or decorators or any imposed structure.

    [:octicons-arrow-right-24: Getting started](#step-1-install-10-seconds)

-    :building_construction:{ .lg .middle } __Bring your infrastructure__

    ---

    ```runnable``` is not a platform. It works with your platforms.

    - ```runnable``` composes pipeline definitions suited to your infrastructure.

    [:octicons-arrow-right-24: Infrastructure](configurations/overview.md)

-   :memo:{ .lg .middle } __Reproducibility__

    ---

    Runnable tracks key information to reproduce the execution. All this happens without
    any additional code.

    [:octicons-arrow-right-24: Run Log](concepts/run-log.md)

-   :repeat:{ .lg .middle } __Retry failures__

    ---

    Debug any failure in your local development environment.

    [:octicons-arrow-right-24: Advanced Patterns](concepts/advanced-patterns/failure-handling.md)

-   :microscope:{ .lg .middle } __Testing__

    ---

    Unit test your code and pipelines.

    - mock/patch the steps of the pipeline
    - test your functions as you normally do.

    [:octicons-arrow-right-24: Testing Guide](concepts/advanced-patterns/mocking-testing.md)

-   :broken_heart:{ .lg .middle } __Move on__

    ---

    Moving away from runnable is as simple as deleting relevant files.

    - Your application code remains as it is.

</div>

<hr style="border:2px dotted orange">
