# Runnable

<img style="float: right;" alt="Runnable" src="docs/assets/sport.png" width="100" height="100">

**Transform any Python function into a portable, trackable pipeline in seconds.**

<p align="center">
<a href="https://pypi.org/project/runnable/"><img alt="python:" src="https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue.svg"></a>
<a href="https://pypi.org/project/runnable/"><img alt="Pypi" src="https://badge.fury.io/py/runnable.svg"></a>
<a href="https://github.com/AstraZeneca/runnable/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache%202.0-blue.svg"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
<a href="https://github.com/python/mypy"><img alt="MyPy Checked" src="https://www.mypy-lang.org/static/mypy_badge.svg"></a>
<a href="https://github.com/AstraZeneca/runnable/actions/workflows/release.yaml"><img alt="Tests:" src="https://github.com/AstraZeneca/runnable/actions/workflows/release.yaml/badge.svg">
</p>

---

## 🚀 30-Second Transformation

**Your existing function (unchanged!):**

```python
def analyze_sales():
    total_revenue = 50000
    best_product = "widgets"
    return total_revenue, best_product
```

**Make it runnable everywhere (2 lines):**

```python
from runnable import PythonJob
PythonJob(function=analyze_sales).execute()
```

**🎉 Success!** Your function now runs the same on laptop, containers, and Kubernetes with automatic tracking and reproducibility.

## 🔗 Chain Functions Without Glue Code

```python
def load_customer_data():
    return {"count": 1500, "segments": ["premium", "standard"]}

def analyze_segments(customer_data):  # Name matches = automatic connection
    return {"premium_pct": 30, "growth_potential": "high"}

# What Runnable needs (same logic, no glue):
from runnable import Pipeline, PythonTask
Pipeline(steps=[
    PythonTask(function=load_customer_data, returns=["customer_data"]),
    PythonTask(function=analyze_segments, returns=["analysis"])
]).execute()
```

**Same pipeline runs unchanged on laptop, containers, and Kubernetes.**

## ⚡ Installation

```bash
pip install runnable
```

## 📊 Why Choose Runnable?

- **🎯 Easy to adopt**: Your code remains as-is, no decorators or imposed structure
- **🏗️ Bring your infrastructure**: Works with your platforms, not a replacement
- **📝 Reproducibility**: Automatic tracking without additional code
- **🔁 Retry failures**: Debug anywhere, retry from failure points
- **🧪 Testing**: Mock/patch pipeline steps, test functions normally
- **💔 Move on**: Easy removal - just delete runnable files, your code stays

## 📖 Documentation

**[Complete Documentation →](https://astrazeneca.github.io/runnable/)**

## 🔀 Pipeline Types

### Linear Pipelines
Simple sequential execution of Python functions, notebooks, or shell scripts.

### Parallel Branches
Execute multiple branches simultaneously for improved performance.

### Map Patterns
Execute pipelines over iterable parameters for batch processing.

### Arbitrary Nesting
Combine parallel, map, and sequential patterns as needed.

---

**Ready to get started?** Check out our [30-second demo](https://astrazeneca.github.io/runnable/) for immediate results!
