# Task Types

Execute different types of tasks in pipelines through runnable's extensible task type system.

## The Core Insight

**All task types follow the same pattern**: They create pipeline steps that wrap a `TaskType` for actual execution, with runnable providing orchestration, parameter passing, and data flow.

## Built-in Task Types

### Python Tasks 🐍
Execute Python functions as pipeline steps:

```python
from runnable import Pipeline, PythonTask
from examples.common.functions import hello

def main():
    task = PythonTask(function=hello, name="say_hello")
    pipeline = Pipeline(steps=[task])
    pipeline.execute()
    return pipeline  # REQUIRED: Always return the pipeline object

if __name__ == "__main__":
    main()
```

**Perfect for**: Data processing, ML models, business logic

!!! success "IDE Debugging Just Works"

    Python tasks are plain functions - **set breakpoints and debug with any IDE**. No special configuration required.

    ```python
    def process_data(input_file: str) -> dict:
        data = load_file(input_file)  # Set breakpoint here
        result = transform(data)       # Step through code
        return {"output": result}      # Inspect variables
    ```

    VSCode, PyCharm, or any Python debugger works out of the box. Runnable calls your functions directly during local execution - no subprocess isolation or remote calls to complicate debugging.

### Notebook Tasks 📓
Execute Jupyter notebooks as pipeline steps:

```python
from runnable import Pipeline, NotebookTask

def main():
    task = NotebookTask(
        name="analyze",
        notebook="examples/common/simple_notebook.ipynb"
    )
    pipeline = Pipeline(steps=[task])
    pipeline.execute()
    return pipeline  # REQUIRED: Always return the pipeline object

if __name__ == "__main__":
    main()
```

**Perfect for**: Exploration, visualization, reporting

### Shell Tasks 🔧
Execute shell commands as pipeline steps:

```python
from runnable import Pipeline, ShellTask

def main():
    task = ShellTask(
        name="greet",
        command="echo 'Hello World!'"
    )
    pipeline = Pipeline(steps=[task])
    pipeline.execute()
    return pipeline  # REQUIRED: Always return the pipeline object

if __name__ == "__main__":
    main()
```

**Perfect for**: System commands, external tools, legacy scripts

### Stub Tasks 🎭
Placeholder tasks for testing and workflow structure:

```python
from runnable import Pipeline, Stub

def main():
    pipeline = Pipeline(steps=[
        Stub(name="extract_data"),
        Stub(name="process_data"),
        Stub(name="save_results")
    ])
    pipeline.execute()
    return pipeline  # REQUIRED: Always return the pipeline object

if __name__ == "__main__":
    main()
```

**Perfect for**: Testing pipeline structure, placeholder steps

### Async Python Tasks ⚡
Execute async functions with streaming support in async pipelines:

!!! warning "Local Execution Only"

    AsyncPythonTask and AsyncPipeline are currently **only supported for local execution**. They cannot be used with containerized or Kubernetes pipeline executors.

```python
from runnable import AsyncPipeline, AsyncPythonTask
import asyncio

async def async_data_fetch(url: str):
    """Simple async function."""
    await asyncio.sleep(1)  # Simulate API call
    return {"data": f"fetched from {url}", "status": "success"}

def main():
    task = AsyncPythonTask(
        name="fetch_data",
        function=async_data_fetch,
        returns=["result"]
    )

    pipeline = AsyncPipeline(steps=[task])

    # In async context: await pipeline.execute()
    return pipeline

if __name__ == "__main__":
    main()
```

**Perfect for**: LLM inference, real-time streaming, async APIs, long-running operations

!!! info "Streaming Support"

    AsyncPythonTask supports **AsyncGenerator functions** for real-time streaming:

    ```python
    from typing import AsyncGenerator

    async def stream_llm_response(prompt: str) -> AsyncGenerator[dict, None]:
        # Stream events in real-time
        yield {"type": "status", "message": "Processing"}
        await asyncio.sleep(0.5)

        # Stream incremental results
        words = ["Hello", "from", "the", "LLM"]
        for word in words:
            yield {"type": "chunk", "text": word}
            await asyncio.sleep(0.1)

        # Final result for pipeline
        yield {"type": "done", "response": " ".join(words)}

    def main():
        task = AsyncPythonTask(
            name="llm_stream",
            function=stream_llm_response,
            returns=["response"],
            stream_end_type="done"  # Extract values from "done" events
        )

        return AsyncPipeline(steps=[task])
    ```

    See [Async & Streaming](../advanced-patterns/async-streaming.md) for complete streaming patterns and FastAPI integration.

## Pipeline Task Execution Context

All task types share the same rich pipeline execution features:

### Common Features Across All Task Types

- **Parameter flow**: Tasks receive parameters from previous steps and configuration
- **Return values**: Tasks can return data to subsequent steps
- **Cross-step data passing**: Use the catalog system for file-based data sharing
- **Mixed execution**: Combine different task types in the same pipeline
- **Environment agnostic**: Run on local, container, or Kubernetes environments

### Example: Mixed Task Pipeline
```python
from runnable import Pipeline, PythonTask, NotebookTask, ShellTask

def main():
    pipeline = Pipeline(steps=[
        PythonTask(function=extract_data, name="extract", returns=["raw_df"]),
        NotebookTask(name="clean", notebook="clean.ipynb", returns=["clean_df"]),
        ShellTask(name="analyze", command="./analyze.sh", returns=["report_path"])
    ])
    pipeline.execute()
    return pipeline  # REQUIRED: Always return the pipeline object
```

Each task type provides the same capabilities:

- **Parameter access**: `{previous_step_return}` interpolation
- **Configuration**: Same YAML/environment variable system
- **Catalog integration**: File storage and retrieval
- **Execution tracking**: Complete run logs and metadata

## The Plugin System

**Task types are pluggable** - runnable automatically discovers and loads custom task types via entry points.

### How Pipeline Tasks Work Internally

**Every task type follows the same pattern**:

1. **Task class**: Provides the pipeline API (`PythonTask`, `ShellTask`, etc.)
2. **Task type**: Handles the actual execution (`PythonTaskType`, `ShellTaskType`, etc.)
3. **Entry point registration**: Makes it discoverable

```python
# Built-in task types are registered like this:
[project.entry-points.'tasks']
"python" = "runnable.tasks:PythonTaskType"
"shell" = "runnable.tasks:ShellTaskType"
"notebook" = "runnable.tasks:NotebookTaskType"
```

## Building Custom Task Types for Pipelines

Create new task types for your specific pipeline needs:

### 1. Implement the Task Type (same as Jobs)
```python
# my_package/tasks.py
from runnable.tasks import BaseTaskType

class RTaskType(BaseTaskType):
    """Execute R scripts with full runnable integration"""
    task_type: str = "r"
    script_path: str = Field(...)

    # Any pydantic validators

    def execute_command(
        self,
        map_variable: MapVariableType = None,
    ) -> StepAttempt:
        # Your R execution logic
        command = f"Rscript {self.script_path}"
        # Run command and return StepAttempt
        pass
```

### 2. Create the Pipeline Task Wrapper
```python
# my_package/tasks.py
from runnable.sdk import BaseTask

class RTask(BaseTask):
    """R script execution in pipelines"""
    # The fields should match the fields of the corresponding task
    script_path: str = Field(...)

    # Should match to the key used in the plugin
    command_type: str = Field(default="r")
```

### 3. Register the Task Type
```toml
# pyproject.toml
[project.entry-points.'tasks']
"r" = "my_package.tasks:RTaskType"
```

### 4. Use Your Custom Task in Pipelines
```python
from my_package.tasks import RTask
from runnable import Pipeline

def main():
    pipeline = Pipeline(steps=[
        RTask(name="analysis", script_path="analysis.R"),
        PythonTask(name="postprocess", function=process_r_results)
    ])
    pipeline.execute()
    return pipeline  # REQUIRED: Always return the pipeline object
```

## Integration Advantage

**🔑 Key Benefit**: Custom task types live entirely in **your codebase**, enabling domain-specific pipeline steps.

### Complete Control & Customization

```python
# In your private repository
# company-analytics/tasks/proprietary_tasks.py

class CompanyAnalyticsTask(BaseTask):
    """Execute proprietary analytics in pipelines"""
    dataset_id: str = Field(...)
    compliance_level: str = Field(default="confidential")

    def create_job(self) -> CompanyAnalyticsTaskType:
        # Your proprietary task implementation
        pass
```

**Integration benefits:**

- **🔒 Proprietary Tools**: Connect pipelines to internal platforms, databases, and tools
- **🏢 Domain-Specific**: Create task types for your specific business processes
- **💼 Compliance**: Implement organization-specific governance and audit requirements
- **🔧 Standardization**: Reusable task types across teams and projects

### Reusable Task Libraries

```python
# Internal package: company-runnable-tasks
from company_runnable_tasks import (
    DataValidationTask,       # Company data quality checks
    ComplianceReportTask,     # Regulatory reporting
    MLModelTrainingTask,      # Your ML platform integration
    CustomerSegmentationTask, # CRM analytics integration
)

# Teams build standardized pipelines
pipeline = Pipeline(steps=[
    DataValidationTask(name="validate", dataset="customer_data"),
    CustomerSegmentationTask(name="segment", model_type="rfm"),
    ComplianceReportTask(name="report", format="sox_compliance")
])
```

**This makes runnable a platform for building your company's custom pipeline ecosystem** - standardized, compliant, and tailored to your business logic.

## Need Help?

**Custom task types involve understanding both the task execution model and your target tool's integration requirements.**

!!! question "Get Support"

    **We're here to help you succeed!** Building custom task types involves:

    - Understanding runnable's task execution lifecycle and pipeline integration
    - Integrating with external tools and platforms
    - Proper parameter flow and data passing between pipeline steps
    - Plugin registration and discovery

    **Don't hesitate to reach out:**

    - 📧 **Contact the team** for architecture guidance and integration support
    - 🤝 **Collaboration opportunities** - we're interested in supporting domain-specific integrations
    - 📖 **Documentation feedback** - help us improve these guides based on your implementation experience

Your success with custom task types helps the entire runnable community!
