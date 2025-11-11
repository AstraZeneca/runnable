# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Runnable is a Python-based pipeline orchestration framework for data science and machine learning workflows. It provides a plugin-based architecture for creating reproducible, configurable pipelines that can execute Python functions, Jupyter notebooks, and shell scripts across different environments.

When trying to match examples to documentation, always use the python based examples.
DO NOT use yaml examples unless I ask you to do so.

When running Python code in this repository, ALWAYS use `uv run` instead of `python` directly.
This ensures proper dependency management and virtual environment isolation.

## Development Commands

### Environment Setup
```bash
# Install with development dependencies and docs
uv sync --all-extras --dev --group docs

# Install pre-commit hooks
pre-commit install
```

### Testing
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/runnable/test_sdk.py

# Run tests matching pattern
pytest -k "test_function_name"
```

### Code Quality
```bash
# Format code (pre-commit will run this automatically)
ruff format .

# Run linting
ruff check . --fix

# Pre-commit
Ensure that the checks in pre-commit config are respected

# Type checking
mypy runnable/ extensions/
```

### CLI Usage
```bash
# Execute a pipeline from YAML definition
runnable execute pipeline.yaml --config config.yaml --parameters params.yaml

# Get help for CLI commands
runnable --help
runnable execute --help
```

## Architecture Overview

### Core Components
- **Pipeline Definition**: Declarative YAML or programmatic Python API for defining workflows
- **Task Types**: Support for Python functions, Jupyter notebooks, shell scripts, and PyTorch models
- **Node Types**: Linear, parallel, map (iterative), conditional, and nested pipeline execution
- **Plugin System**: Extensible architecture with entry points for executors, catalogs, secrets, and storage

### Workspace Structure
This is a UV workspace with multiple extensions:
- `runnable/`: Core framework code
- `extensions/`: Plugin implementations
  - `catalog/`: Data storage backends (file system, S3, Minio)
  - `job_executor/`: Job execution backends (local, Kubernetes)
  - `nodes/`: Node type implementations
  - `pipeline_executor/`: Pipeline execution backends (local, container, Argo)
  - `run_log_store/`: Execution metadata storage
  - `secrets/`: Secret management backends
- `visualization/`: Pipeline visualization tools

### Key Design Patterns
- **Plugin Architecture**: Uses Python entry points for extensibility
- **Separation of Concerns**: Domain code remains independent of orchestration
- **Reproducibility**: Built-in metadata tracking and execution history
- **Environment Agnostic**: Same pipeline can run locally, in containers, or on Kubernetes

## Common Development Workflows

### Adding New Task Types
1. Create task implementation in `runnable/tasks.py`
2. Register via entry point in `pyproject.toml` under `[project.entry-points.'tasks']`
3. Add tests in `tests/runnable/test_tasks.py`

### Adding New Executors
1. Implement executor in appropriate `extensions/` subdirectory
2. Register via entry point in main `pyproject.toml`
3. Add configuration examples in `examples/configs/`
4. Add integration tests

### Running yaml Examples, a legacy which might go away.
```bash
# Execute example pipeline
runnable execute examples/01-tasks/python_tasks.yaml

# Execute with custom config
runnable execute examples/01-tasks/python_tasks.yaml --config examples/configs/local-container.yaml

# Execute with parameters
runnable execute examples/03-parameters/passing_parameters_python.yaml --parameters examples/03-parameters/parameters.yaml
```

### Running python based examples

```bash
uv run examples/01-tasks/python_tasks.py
```

### Extension Development
Each extension is a separate package in the workspace:
- Has its own `pyproject.toml` with dependencies
- Registers plugins via entry points in the main `pyproject.toml`
- Can be developed and tested independently
- Uses workspace dependencies for shared code

## Pipeline Definition Patterns

### Python API
```python
from runnable import Pipeline, PythonTask, pickled

task = PythonTask(
    function=my_function,
    name="process_data",
    returns=[pickled("result")]
)

pipeline = Pipeline(steps=[task])
pipeline.execute()
```

### YAML Definition which is legacy and might go away
```yaml
dag:
  start_at: process_data

  process_data:
    type: task
    task_type: python
    command: my_module.my_function
    returns:
      - name: result
        type: pickled
```

## Release Process

- Uses semantic-release for automated versioning
- Conventional commit format required
- GitHub Actions handle CI/CD
- Supports alpha releases from `alpha-*` branches
- Main release branch is `main`
- Never EVER put a ! in the prefix of commit as it triggers a major release.

## Testing Strategy

- Unit tests for core functionality in `tests/runnable/`
- Extension tests in `tests/extensions/`
- Integration tests for pipeline examples in `tests/test_pipeline_examples.py`
- Mock executor for testing without external dependencies
- Assertion helpers in `tests/assertions.py`

# Documentation
The docs are based on mkdocs and the base configuration file is in mkdocs.yaml while the content is present in docs folder.

The docs use a lot of code snippets based on examples and you should keep that pattern where ever possible.
The docs explain the contextual example first and then show a detailed working example from the examples folder

When writing docs always use code from examples directory and always use code snippets to avoid duplication

Remember that when writing lists in md, there should be an empty line between the list - and the preceding line


I prefer to give prompts in a visual editor and I have my prompts in a file called prompt.md.
I will refer to the particular section that I want to talk about and use that as my interactions.
