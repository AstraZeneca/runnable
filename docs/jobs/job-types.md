# Job Types

Execute different types of tasks through runnable's extensible job type system.

## The Core Insight

**All job types follow the same pattern**: They wrap a `TaskType` that handles the actual execution, with runnable providing the context, logging, and infrastructure.

## Built-in Job Types

### Python Jobs 🐍
Execute Python functions with full context tracking:

```python
from runnable import PythonJob

def my_analysis():
    return {"result": "success"}

def main():
    job = PythonJob(function=my_analysis, returns=["result"])
    job.execute()
    return job

if __name__ == "__main__":
    main()
```

### Shell Jobs 🔧
Execute shell commands with full context tracking:

```python
from runnable import ShellJob

def main():
    job = ShellJob(command="echo 'Hello World!'")
    job.execute()
    return job

if __name__ == "__main__":
    main()
```

### Notebook Jobs 📓
Execute Jupyter notebooks with complete output preservation:

!!! info "Installation Required"

    Notebook execution requires the optional notebook dependency:
    ```bash
    pip install runnable[notebook]
    ```

```python
from runnable import NotebookJob

def main():
    job = NotebookJob(notebook="examples/common/simple_notebook.ipynb")
    job.execute()
    return job

if __name__ == "__main__":
    main()
```

## Job Execution Context

All Job types share the same rich execution context and features:

### Common Features Across All Job Types

- **Parameters**: All jobs support the same parameter system (files, environment variables)
- **Catalog**: File storage and retrieval using glob patterns
- **Return values**: Specify what data to capture and store
- **Secrets**: Access to environment variables and secret management
- **Run logs**: Complete execution tracking and metadata

### Execution Context Example
```
JobContext(
    execution_mode='python',           # or 'shell', 'notebook'
    run_id='feasible-booth-0628',      # Unique execution ID
    catalog=FileSystemCatalog(         # File storage
        catalog_location='.catalog'
    ),
    job_executor=LocalJobExecutor(),   # Execution environment
    secrets=EnvSecretsManager(),       # Secret management
    run_log_store=FileSystemRunLogstore()  # Metadata storage
)
```

### Choosing the Right Job Type

| Use Case | Best Job Type | Why |
|----------|---------------|-----|
| Custom data analysis | **PythonJob** | Full control, return values, type safety |
| System operations | **ShellJob** | Leverage existing scripts and tools |
| Interactive analysis | **NotebookJob** | Visual outputs, step-by-step exploration |
| File processing | **ShellJob** | Use command-line tools directly |
| Model training | **PythonJob** or **NotebookJob** | Python for code, Notebook for exploration |
| Report generation | **NotebookJob** | Rich outputs with plots and formatting |

## The Plugin System

**Job types are pluggable** - runnable automatically discovers and loads custom job types via entry points.

### How Jobs Work Internally

**Every job type follows the same pattern**:

1. **Job wrapper**: Provides the user API (`PythonJob`, `ShellJob`, etc.)
2. **Task type**: Handles the actual execution (`PythonTaskType`, `ShellTaskType`, etc.)
3. **Entry point registration**: Makes it discoverable

```python
[project.entry-points.'tasks']
"python" = "runnable.tasks:PythonTaskType"
"shell" = "runnable.tasks:ShellTaskType"
"notebook" = "runnable.tasks:NotebookTaskType"
```

## Building Custom Job Types

Create new job types for your specific execution needs:

### 1. Implement the Task Type
```python
# my_package/tasks.py
from runnable.tasks import BaseTaskType

class RTaskType(BaseTaskType):
    """Execute R scripts with full runnable integration"""
    task_type: str = "r"
    script_path: str = Field(...)

    def execute_command(
        self,
        map_variable: MapVariableType = None,
    ) -> StepAttempt:
        # Your R execution logic
        command = f"Rscript {self.script_path}"
        # Run command and return StepAttempt
        pass

```

### 2. Create the Job Wrapper
```python
# my_package/jobs.py
from runnable.sdk import BaseJob

class RJob(BaseJob):
    # The name of the plugin of Task
    command_type: str = Field(default="r")

    # The fields should be the same as the corresponding task definition
    script_path: str = Field(...)

```

### 3. Register Task Entry Point
```toml
# pyproject.toml
[project.entry-points.'tasks']
"r" = "my_package.tasks:RTaskType"
```

### 4. Use Your Custom Job
```python
from my_package.jobs import RJob

def main():
    job = RJob(script_path="analysis.R")
    job.execute()
    return job
```

## Integration Advantage

**🔑 Key Benefit**: Custom job types live entirely in **your codebase**, enabling domain-specific execution models.

### Complete Control & Customization

```python
# In your private repository
# company-analytics/jobs/proprietary_jobs.py

class CompanyAnalyticsJob(BaseJob):
    """Execute proprietary analytics with company-specific integrations"""
    dataset_id: str = Field(...)
    compliance_level: str = Field(default="confidential")

    def get_task(self) -> CompanyAnalyticsTaskType:
        # Your proprietary task implementation
        pass
```

**Integration benefits:**

- **🔒 Proprietary Tools**: Integrate with internal analytics platforms, databases, or custom tools
- **🏢 Domain-Specific**: Create job types for your specific business logic (financial modeling, scientific computing, etc.)
- **💼 Compliance**: Implement organization-specific security, audit, and governance requirements
- **🔧 Standardization**: Create reusable job types across teams and projects

### Reusable Job Libraries

```python
# Internal package: company-runnable-jobs
from company_runnable_jobs import (
    FinancialModelingJob,     # Company financial calculations
    ComplianceReportJob,      # SOX/regulatory reporting
    DataScienceJob,          # Your ML platform integration
    CustomerAnalyticsJob,    # CRM and analytics integration
)

# Teams use your standardized job types
job = FinancialModelingJob(
    model_type="monte_carlo",
    compliance_required=True
)
job.execute()
```

**This makes runnable a platform for creating your company's custom execution ecosystem** - from simple scripts to complex domain-specific workflows.

## Need Help?

**Custom job types involve understanding both the task execution model and your target tool's integration requirements.**

!!! question "Get Support"

    **We're here to help you succeed!** Building custom job types involves:

    - Understanding runnable's task execution lifecycle
    - Integrating with external tools and platforms
    - Proper error handling and result management
    - Plugin registration and discovery

    **Don't hesitate to reach out:**

    - 📧 **Contact the team** for architecture guidance and integration support
    - 🤝 **Collaboration opportunities** - we're interested in supporting domain-specific integrations
    - 📖 **Documentation feedback** - help us improve these guides based on your implementation experience

Your success with custom job types helps the entire runnable community!
