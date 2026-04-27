# Runnable

<figure markdown>
  ![Image title](assets/sport.png){ width="200" height="100"}
  <figcaption>Orchestrate your functions, notebooks, scripts anywhere!!</figcaption>
</figure>

<span style="font-size:0.75em;">
<a href="https://www.flaticon.com/free-icons/runner" title="runner icons">Runner icons created by Leremy - Flaticon</a>
</span>

Transform Python functions into portable, trackable pipelines.

<hr style="border:2px dotted orange">

## Step 1: Install

```bash
pip install runnable
```

!!! tip "Optional Features"

    Install optional features as needed:
    ```bash
    pip install runnable[notebook]    # Jupyter notebook execution
    pip install runnable[docker]     # Container execution
    pip install runnable[k8s]        # Kubernetes job executors
    pip install runnable[s3]         # S3 storage backend
    pip install runnable[examples]   # Example dependencies
    ```

## Step 2: Your Function (unchanged!)

```python
# Your existing function - zero changes needed
def analyze_sales():
    total_revenue = 50000
    best_product = "widgets"
    return total_revenue, best_product
```

## Step 3: Make It Runnable

```python
# Add main function → Make it runnable everywhere
from runnable import PythonJob

def main():
    job = PythonJob(
        function=analyze_sales,
        # returns=["total_revenue", "best_product"]  # Optional: track return values
    )
    job.execute()
    return job  # REQUIRED: Always return the job object

if __name__ == "__main__":
    main()
```

## 🎉 Success!

You just made your first function runnable and got:

- ✅ **Automatic tracking**: execution logs, timestamps, results saved
- ✅ **Reproducible runs**: full execution history and metadata
- ✅ **Environment portability**: runs the same on laptop, containers, Kubernetes

**Your code runs consistently across different environments!**

---

## Want to See More?

### 🔧 Same Code, Different Parameters

Change parameters without touching your code:

```python
# Function accepts parameters
def forecast_growth(revenue, growth_rate):
    return revenue * (1 + growth_rate) ** 3

from runnable import PythonJob

def main():
    job = PythonJob(
        function=forecast_growth,
        # returns=["forecast"]  # Optional: track return values
    )
    job.execute()
    return job  # REQUIRED: Always return the job object

if __name__ == "__main__":
    main()

# Run different scenarios anywhere:
# RUNNABLE_PRM_revenue=100000 RUNNABLE_PRM_growth_rate=0.05 python forecast.py
# ✨ Every run tracked with parameters - reproducible everywhere
```

??? example "See complete parameter example"
    ```python title="examples/11-jobs/passing_parameters_python.py"
    --8<-- "examples/11-jobs/passing_parameters_python.py"
    ```

    **Try it:** `uv run examples/11-jobs/passing_parameters_python.py`

**Why bother?** No more "what parameters gave us those good results?" - tracked automatically across all environments.

---

### 🔗 Chain Functions, No Glue Code

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

def main():
    pipeline = Pipeline(steps=[
        PythonTask(name="load_data", function=load_customer_data, returns=["customer_data"]),
        PythonTask(name="analyze", function=analyze_segments, returns=["analysis"])
    ])
    pipeline.execute()
    return pipeline  # REQUIRED: Always return the pipeline object

if __name__ == "__main__":
    main()

# Same pipeline runs unchanged on:
# • Your laptop (development)
# • Docker containers (testing)
# • Kubernetes (production)

# ✨ Write once, run anywhere - zero deployment rewrites
```

??? example "See complete pipeline example"
    ```python title="examples/02-sequential/traversal.py"
    --8<-- "examples/02-sequential/traversal.py"
    ```

    **Try it:** `uv run examples/02-sequential/traversal.py`

**Why bother?** No more "it works locally but breaks in production" - same code, guaranteed same behavior.

---

### 🚀 Mix Python + Notebooks

Different tools, portable workflows:

```python
# Python prepares data, notebook analyzes - works everywhere
def prepare_dataset():
    clean_data = {"sales": [100, 200, 300], "regions": ["north", "south"]}
    return clean_data

from runnable import Pipeline, PythonTask, NotebookTask

def main():
    pipeline = Pipeline(steps=[
        PythonTask(function=prepare_dataset, returns=["dataset"]),
        NotebookTask(notebook="deep_analysis.ipynb", returns=["insights"])
    ])
    pipeline.execute()
    return pipeline  # REQUIRED: Always return the pipeline object

if __name__ == "__main__":
    main()

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

## 🔍 Complete Working Examples

**All examples in this documentation are fully working code!** Every code snippet comes from the `examples/` directory with complete, tested implementations.

!!! example "Repository Examples"

    **📁 [Browse All Examples](https://github.com/AstraZeneca/runnable/tree/main/examples)**

    Complete, tested examples organized by topic:

    - **`examples/01-tasks/`** - Basic task types (Python, notebooks, shell scripts)
    - **`examples/02-sequential/`** - Multi-step workflows and conditional logic
    - **`examples/03-parameters/`** - Configuration and parameter passing
    - **`examples/04-catalog/`** - File storage and data management
    - **`examples/06-parallel/`** - Parallel execution patterns
    - **`examples/07-map/`** - Iterative processing over data
    - **`examples/10-loop/`** - Loop workflows with dynamic iteration
    - **`examples/11-jobs/`** - Single job execution examples
    - **`examples/configs/`** - Configuration files for different environments

    **📋 All examples include:**

    - ✅ Complete Python code following the correct patterns
    - ✅ Configuration files for different execution environments
    - ✅ Instructions on how to run them with `uv run`
    - ✅ Tested in CI to ensure they always work

**🚀 Quick Start**: Pick any example and run it immediately:
```bash
git clone https://github.com/AstraZeneca/runnable.git
cd runnable
uv run examples/01-tasks/python_tasks.py
```

---

## 🔄 Batch Processing + 🧪 Experimental Async (Proof of Concept)

**Runnable provides robust batch processing with experimental async streaming capabilities as a proof of concept.**

!!! warning "Not Recommended for Production Agentic Frameworks"

    While Runnable includes **experimental async capabilities**, these are primarily a **proof of possibility** rather than production-ready features for agentic frameworks.

    **For production agentic applications, we recommend using mature, purpose-built frameworks like:**

    - **[Pydantic AI](https://github.com/pydantic/pydantic-ai)** - Production-ready async agent framework
    - **[LangChain](https://python.langchain.com/)** - Comprehensive LLM application framework
    - **[CrewAI](https://github.com/joaomdmoura/crewAI)** - Multi-agent orchestration framework

    Runnable's async features are **experimental, local-only, and subject to change** without notice.

Runnable's strength lies in **data pipeline orchestration**:

**🔄 Batch Processing** - Production-ready data pipelines with full reproducibility
```python
# Regular batch pipeline - Runnable's core strength
pipeline = Pipeline(steps=[
    PythonTask(function=process_data, name="process"),
    PythonTask(function=train_model, name="train")
])
pipeline.execute()  # Runs to completion
```

**🧪 Experimental Async** - Proof of concept for streaming (local execution only)
```python
# Experimental async - NOT recommended for production agents
pipeline = AsyncPipeline(steps=[
    AsyncPythonTask(function=stream_llm_response, name="llm")
])
async for event in pipeline.execute_streaming():
    print(event)  # {"type": "chunk", "text": "Hello"}
```

**🎯 Runnable is designed for**:

- Data processing pipelines
- ML model training workflows
- ETL operations
- Batch job orchestration
- Scientific computing reproducibility

**❌ Not recommended for**:

- Production agentic frameworks
- Complex multi-agent systems
- Real-time conversational AI
- Advanced LLM orchestration patterns

[:octicons-arrow-right-24: Learn Async Limitations](advanced-patterns/async-streaming.md)

---

## What's Next?

You've seen how Runnable transforms your code for portability and tracking. Ready to go deeper?

**🎯 Master the Concepts** → [Jobs vs Pipelines](pipelines/jobs-vs-pipelines.md)
Learn when to use single jobs vs multi-step pipelines

**⚡ Async & Streaming** → [Async & Streaming Execution](advanced-patterns/async-streaming.md)
Real-time streaming workflows for LLMs, APIs, and live data processing

**📊 Handle Your Data** → [Task Types](pipelines/task-types.md)
Work with returns, parameters, and different data types

**👁️ Visualize Execution** → [Pipeline Visualization](pipelines/visualization.md)
Interactive timelines showing execution flow and timing

**⚡ See Real Examples** → [Browse Repository Examples](https://github.com/AstraZeneca/runnable/tree/main/examples)
All working examples with full code in the `examples/` directory

**🚀 Deploy Anywhere** → [Production Guide](production/deploy-anywhere.md)
Scale from laptop to containers to Kubernetes

**🔍 Compare Alternatives** → [Compare Tools](compare/kedro.md)
See how Runnable compares to Kedro, Metaflow, and other orchestration tools

---

## Why Choose Runnable?

<div class="grid cards" markdown>

-   :material-clock-fast:{ .lg .middle } __Easy to adopt, its mostly your code__

    ---

    Your application code remains as it is. Runnable exists outside of it.

    - No API's or decorators or any imposed structure.

    [:octicons-arrow-right-24: Getting started](jobs/index.md)

-    :building_construction:{ .lg .middle } __Bring your infrastructure__

    ---

    ```runnable``` is not a platform. It works with your platforms.

    - ```runnable``` composes pipeline definitions suited to your infrastructure.
    - **Extensible plugin architecture**: Build custom executors, storage backends, and task types for any platform.

    [:octicons-arrow-right-24: Infrastructure](production/deploy-anywhere.md)

-   :memo:{ .lg .middle } __Reproducibility__

    ---

    Runnable tracks key information to reproduce the execution. All this happens without
    any additional code.

    [:octicons-arrow-right-24: Run Log](production/run-log.md)

-   :repeat:{ .lg .middle } __Retry failures__

    ---

    Debug any failure in your local development environment.

    [:octicons-arrow-right-24: Advanced Patterns](advanced-patterns/failure-handling.md)

-   :microscope:{ .lg .middle } __Testing__

    ---

    Unit test your code and pipelines.

    - mock/patch the steps of the pipeline
    - test your functions as you normally do.

    [:octicons-arrow-right-24: Testing Guide](advanced-patterns/mocking-testing.md)

-   :broken_heart:{ .lg .middle } __Move on__

    ---

    Moving away from runnable is as simple as deleting relevant files.

    - Your application code remains as it is.

</div>

<hr style="border:2px dotted orange">
