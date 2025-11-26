# Runnable Examples 🚀

**Complete, tested examples showing how to use Runnable for jobs and pipelines.**

All examples are fully working Python code that you can run immediately with `uv run`. Every example follows the correct patterns and includes proper configuration files.

## 🎯 What's Provided

**📋 Every example includes:**

- ✅ **Complete Python code** following the correct `main()` return patterns
- ✅ **Configuration files** for different execution environments
- ✅ **Clear instructions** on how to run with `uv run`
- ✅ **CI tested** to ensure they always work

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/AstraZeneca/runnable.git
cd runnable

# Run any example immediately
uv run examples/01-tasks/python_tasks.py
uv run examples/11-jobs/python_tasks.py
uv run examples/02-sequential/traversal.py
```

## 📁 Examples Directory Structure

### 🔧 **Basic Task Types** (`01-tasks/`)
Learn the fundamental building blocks of Runnable pipelines:

- **[python_tasks.py](01-tasks/python_tasks.py)** - Execute Python functions as pipeline tasks
- **[notebook.py](01-tasks/notebook.py)** - Execute Jupyter notebooks as tasks
- **[scripts.py](01-tasks/scripts.py)** - Execute shell commands as tasks
- **[stub.py](01-tasks/stub.py)** - Placeholder tasks for testing workflow structure

### 🔗 **Sequential Workflows** (`02-sequential/`)
Build multi-step pipelines with proper flow control:

- **[traversal.py](02-sequential/traversal.py)** - Mixed pipeline: Python + notebooks + shell scripts
- **[conditional.py](02-sequential/conditional.py)** - Conditional branching based on parameters
- **[default_fail.py](02-sequential/default_fail.py)** - Default failure handling behavior
- **[on_failure_fail.py](02-sequential/on_failure_fail.py)** - Custom failure handling
- **[on_failure_succeed.py](02-sequential/on_failure_succeed.py)** - Recovery from failures

### ⚙️ **Configuration & Parameters** (`03-parameters/`)
Pass data and configuration between tasks:

- **[passing_parameters_python.py](03-parameters/passing_parameters_python.py)** - Parameter passing between Python tasks
- **[passing_parameters_notebook.py](03-parameters/passing_parameters_notebook.py)** - Parameters with Jupyter notebooks
- **[passing_parameters_shell.py](03-parameters/passing_parameters_shell.py)** - Parameters with shell scripts
- **[static_parameters_python.py](03-parameters/static_parameters_python.py)** - Static configuration parameters

### 📁 **File Management** (`04-catalog/`)
Handle file storage and sharing between tasks:

- **[catalog.py](04-catalog/catalog.py)** - File sharing between different task types
- **[catalog_python.py](04-catalog/catalog_python.py)** - Python-specific file handling
- **[catalog_no_copy.py](04-catalog/catalog_no_copy.py)** - Hash-only tracking for large files
- **[catalog_on_fail.py](04-catalog/catalog_on_fail.py)** - File handling during failures

### ⚡ **Parallel Execution** (`06-parallel/`)
Run multiple tasks simultaneously:

- **[parallel.py](06-parallel/parallel.py)** - Basic parallel execution
- **[parallel_branch_fail.py](06-parallel/parallel_branch_fail.py)** - Handling failures in parallel branches
- **[nesting.py](06-parallel/nesting.py)** - Nested parallel and sequential workflows

### 🔁 **Iterative Processing** (`07-map/`)
Process data collections with map operations:

- **[map.py](07-map/map.py)** - Basic map operation over data
- **[map_fail.py](07-map/map_fail.py)** - Error handling in map operations
- **[custom_reducer.py](07-map/custom_reducer.py)** - Custom result aggregation

### 🎯 **Single Job Execution** (`11-jobs/`)
Execute standalone jobs (not pipelines):

- **[python_tasks.py](11-jobs/python_tasks.py)** - Execute Python functions as jobs
- **[notebooks.py](11-jobs/notebooks.py)** - Execute Jupyter notebooks as jobs
- **[scripts.py](11-jobs/scripts.py)** - Execute shell commands as jobs
- **[catalog.py](11-jobs/catalog.py)** - File management in jobs
- **[catalog_no_copy.py](11-jobs/catalog_no_copy.py)** - Hash-only file tracking
- **[passing_parameters_python.py](11-jobs/passing_parameters_python.py)** - Parameter handling in jobs

### ⚙️ **Configuration Files** (`configs/`)
Ready-to-use configuration for different environments:

- **[default.yaml](configs/default.yaml)** - Basic local execution
- **[local-container.yaml](configs/local-container.yaml)** - Docker container execution
- **[parallel_enabled.yaml](configs/parallel_enabled.yaml)** - Parallel execution settings
- **[argo-config.yaml](configs/argo-config.yaml)** - Argo Workflows production setup
- **[k8s-job.yaml](11-jobs/k8s-job.yaml)** - Kubernetes job execution
- **[minio.yaml](configs/minio.yaml)** - MinIO object storage
- **[in-memory.yaml](configs/in-memory.yaml)** - In-memory testing configuration

### 🧩 **Common Utilities** (`common/`)
Shared functions, notebooks, and scripts used across examples:

- **[functions.py](common/functions.py)** - Common Python functions
- **[simple_notebook.ipynb](common/simple_notebook.ipynb)** - Example Jupyter notebook

## 📚 How to Use Examples

### 🏃 **Run Examples Immediately**
All examples work out of the box:

```bash
# Basic task execution
uv run examples/01-tasks/python_tasks.py

# Sequential workflow
uv run examples/02-sequential/traversal.py

# Parallel execution
uv run examples/06-parallel/parallel.py

# Single job execution
uv run examples/11-jobs/python_tasks.py
```

### 🔧 **Use Different Configurations**
Run the same code in different environments:

```bash
# Local execution (default)
uv run examples/01-tasks/python_tasks.py

# Container execution
RUNNABLE_CONFIGURATION_FILE=examples/configs/local-container.yaml uv run examples/01-tasks/python_tasks.py

# Parallel execution
RUNNABLE_CONFIGURATION_FILE=examples/configs/parallel_enabled.yaml uv run examples/06-parallel/parallel.py
```

### 📊 **Parameter Examples**
Pass parameters from environment or files:

```bash
# Environment parameters
RUNNABLE_PRM_my_parameter="test_value" uv run examples/03-parameters/static_parameters_python.py

# Parameter files
RUNNABLE_PARAMETERS_FILE=examples/03-parameters/parameters.yaml uv run examples/03-parameters/passing_parameters_python.py
```

## 🎓 Learning Path

**New to Runnable?** Follow this progression:

1. **Start Simple**: `examples/11-jobs/python_tasks.py` - Single job execution
2. **Add Workflows**: `examples/02-sequential/traversal.py` - Multi-step pipelines
3. **Handle Data**: `examples/03-parameters/passing_parameters_python.py` - Parameter passing
4. **Manage Files**: `examples/04-catalog/catalog.py` - File storage and sharing
5. **Scale Up**: `examples/06-parallel/parallel.py` - Parallel execution
6. **Go Advanced**: `examples/07-map/map.py` - Iterative processing

## 💡 Key Patterns

**All examples demonstrate:**

- ✅ **Correct main() pattern** - Always return job/pipeline objects
- ✅ **Environment configuration** - Use `RUNNABLE_CONFIGURATION_FILE`
- ✅ **Parameter management** - Both static and dynamic parameters
- ✅ **Error handling** - Proper failure management
- ✅ **File management** - Catalog integration for data persistence
- ✅ **Multi-environment** - Same code runs locally, in containers, on Kubernetes

## 🔍 Need Help?

- 📖 **[Full Documentation](https://astrazeneca.github.io/runnable/)** - Complete guides and API reference
- 🎯 **[Jobs vs Pipelines](https://astrazeneca.github.io/runnable/pipelines/jobs-vs-pipelines/)** - Choose the right approach
- 🚀 **[Production Guide](https://astrazeneca.github.io/runnable/production/deploy-anywhere/)** - Deploy to any environment

**Every example is tested and documented - pick one and start building!** 🚀
