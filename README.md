# Runnable

<img style="float: right;" alt="Runnable" src="docs/assets/sport.png" width="100" height="100">

**Transform any Python function into a portable, trackable pipeline in seconds.**

<p align="center">
<a href="https://pypi.org/project/runnable/"><img alt="python:" src="https://img.shields.io/badge/python-3.10+-blue.svg"></a>
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

**Make it runnable everywhere:**

```python
from runnable import PythonJob

def main():
    PythonJob(function=analyze_sales).execute()

if __name__ == "__main__":
    main()
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

def main():
    Pipeline(steps=[
        PythonTask(function=load_customer_data, returns=["customer_data"]),
        PythonTask(function=analyze_segments, returns=["analysis"])
    ]).execute()

if __name__ == "__main__":
    main()
```

**Same pipeline runs unchanged on laptop, containers, and Kubernetes.**

## ⚡ Installation

```bash
pip install runnable
```

**For development:**
```bash
uv sync --all-extras --dev
```

**Run examples:**
```bash
uv run examples/01-tasks/python_tasks.py
```

## 📊 Why Choose Runnable?

- **🎯 Easy to adopt**: Your code remains as-is, no decorators or imposed structure
- **🏗️ Bring your infrastructure**: Works with your platforms, not a replacement
- **📝 Reproducibility**: Automatic tracking without additional code
- **🔁 Retry failures**: Debug anywhere, retry from failure points
- **🧪 Testing**: Mock/patch pipeline steps, test functions normally
- **💔 Move on**: Easy removal - just delete runnable files, your code stays

## 🎯 Core Strengths & Extensibility

**Runnable excels at data pipeline orchestration:**

- Data processing pipelines
- ML model training workflows
- ETL operations
- Batch job orchestration
- Scientific computing reproducibility

**🧪 Extensible Architecture & Experimental Features:**

Runnable's plugin-based architecture demonstrates its extensibility through experimental features like:

- **Async streaming capabilities** - Proof of concept for real-time processing (local execution)
- **Loop workflows** - Dynamic iteration patterns for complex data processing
- **Custom executors** - Extensible to any infrastructure via plugins

> **For production agentic frameworks**, consider specialized tools like [Pydantic AI](https://github.com/pydantic/pydantic-ai), [LangChain](https://python.langchain.com/), or [CrewAI](https://github.com/joaomdmoura/crewAI), which are purpose-built for complex LLM applications.

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

## 🆚 Why Choose Runnable?

| | **Runnable** | **Kedro** | **Metaflow** | **Airflow** |
|---|---|---|---|---|
| **Zero Code Changes** | ✅ Wrap existing functions | ❌ Restructure to nodes | ❌ Convert to FlowSpec | ❌ Rewrite as DAG tasks |
| **Environment Portability** | Same code: laptop→container→K8s→Argo | Deployment-specific configs | AWS-focused with --with flags | Platform-specific operators |
| **Mixed Task Types** | Python + Notebooks + Shell | Python nodes only | Python steps only | Requires separate operators |
| **Plugin Extensibility** | Auto-discovery via entry points | kedro-* packages | Limited extensions | Complex plugin development |
| **Parameter Passing** | Automatic by name matching | Manual catalog definitions | Flow state management | Manual XCom operations |

## 🚀 Time to Value

**Runnable**:
```bash
pip install runnable
# Wrap your existing function - done!
```

**Others**:
- Learn framework conventions
- Restructure existing code
- Configure project structure
- Set up deployment configs

---

**Ready to get started?** Check out our [30-second demo](https://astrazeneca.github.io/runnable/) for immediate results!
