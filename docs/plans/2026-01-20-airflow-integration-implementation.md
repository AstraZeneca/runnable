# Airflow DAG Factory Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create an AirflowDagFactory that converts Runnable pipelines to native Airflow DAGs at import time.

**Architecture:** The factory uses `runpy.run_path()` to load pipeline files and extract the Pipeline object, then builds Airflow DAG objects with DockerOperator tasks executing `runnable` CLI commands (`execute-single-node`, `fan`).

**Tech Stack:** Python, Pydantic, Apache Airflow 2.8+ (< 3.0), Docker

---

## Implementation Learnings

### Pipeline Loading Strategy

**Problem:** `get_pipeline_spec_from_python()` expects a `module:function` format (e.g., `examples.pipeline:main`), not a file path.

**Solution:** Use `runpy.run_path()` to execute the pipeline file and extract the Pipeline object:

```python
import runpy
from runnable import context
from runnable.sdk import Pipeline, AsyncPipeline

# Load without triggering if __name__ == "__main__" block
result = runpy.run_path(pipeline_file, run_name="__airflow_dag_builder__")

# Set temporary context so pipeline.execute() returns early
context.set_run_context(object())  # Any non-None value works

try:
    if "main" in result and callable(result["main"]):
        pipeline = result["main"]()
    else:
        # Look for Pipeline in module namespace
        for value in result.values():
            if isinstance(value, (Pipeline, AsyncPipeline)):
                pipeline = value
                break
finally:
    context.set_run_context(None)  # Clear context

runnable_graph = pipeline.return_dag()
```

**Key insights:**
1. Use `run_name="__airflow_dag_builder__"` to avoid triggering `if __name__ == "__main__"` block
2. Set a temporary context before calling `main()` so `pipeline.execute()` returns early (it checks `_is_called_for_definition()`)
3. Always clear context in `finally` block to avoid test pollution

### Airflow Version Constraint

**Problem:** Airflow 3.x has a dependency conflict with structlog:
```
ImportError: cannot import name 'Styles' from 'structlog.dev'
```

**Solution:** Pin to Airflow 2.x in `pyproject.toml`:
```toml
airflow = [
    "apache-airflow>=2.8.0,<3.0.0",
    "apache-airflow-providers-docker>=3.0.0",
]
```

### Test Context Isolation

**Problem:** The `test_context_isolation_sync` test was leaving context set, causing subsequent tests to fail because `pipeline.execute()` would return early.

**Solution:** Always clean up context in tests:
```python
def test_context_isolation_sync():
    try:
        set_run_context(context1)
        # ... test code ...
    finally:
        set_run_context(None)  # Clean up!
```

### Config vs Auto-Discovery

**Decision:** Use explicit `config_file` and `parameters_file` fields in `AirflowDagFactory` instead of auto-discovery. This provides:
1. Clarity about which config is used
2. Flexibility to use configs in different locations
3. No magic path conventions to remember

---

## User Workflow

### Step 1: Create Airflow Config Alongside Pipeline

File naming conventions in the same directory as the pipeline:
- Config: `{pipeline_name}.airflow.yaml` (required)
- Parameters: `{pipeline_name}.parameters.yaml` (optional)

```
examples/02-sequential/
├── traversal.py              # Runnable pipeline
├── traversal.airflow.yaml    # Airflow-specific config (auto-discovered)
└── traversal.parameters.yaml # Static parameters (optional, auto-discovered)
```

The config file serves two purposes:
1. **At DAG creation time**: Read Docker settings (image, volumes) and per-step overrides
2. **At runtime**: Provide catalog, run-log-store, secrets configuration

**Config file structure:**
```yaml
# Runtime configuration (used by runnable CLI in containers)
run-log-store:
  type: file-system
  config:
    log_folder: /tmp/run_logs

catalog:
  type: file-system
  config:
    catalog_location: /tmp/catalog

secrets:
  type: do-nothing

# Airflow/Docker configuration (used by factory at DAG creation time)
executor:
  type: airflow
  config:
    image: my-runnable-image:latest
    docker_url: unix://var/run/docker.sock
    network_mode: bridge
    auto_remove: success
    mount_tmp_dir: false
    volumes:
      - /host/run_logs:/tmp/run_logs    # Share run log across containers
      - /host/catalog:/tmp/catalog      # Share catalog across containers
    environment:
      KEY: value
    # Per-step image overrides (optional)
    overrides:
      step_name:
        image: custom-image:latest
```

**Important:** Volume mounts must map host paths to the same paths used in `run-log-store` and `catalog` configs to share data across containers.

### Step 2: Create DAG Loader in Airflow's DAGs Folder

```python
# dags/my_dags.py
from extensions.pipeline_executor.airflow import AirflowDagFactory

factory = AirflowDagFactory(
    # Only DAG-level defaults - Docker config comes from airflow.yaml
    default_args={"owner": "runnable", "retries": 1},
    catchup=False,
    tags=["runnable"],
)

# Config auto-discovered: examples/02-sequential/traversal.airflow.yaml
# All Docker settings (image, volumes) come from the config file
traversal_dag = factory.create_dag(
    pipeline_file="examples/02-sequential/traversal.py",
    dag_id="traversal",
)
```

### Step 3: Docker Image Must Contain

- Runnable installed
- Pipeline files at same paths as host
- Config files at same paths as host
- User's actual code (the functions being called)

### How It Works

1. **Airflow scheduler imports** `dags/my_dags.py`
2. **At import time**, `create_dag()` runs:
   - Loads pipeline graph via `get_pipeline_spec_from_python()`
   - Auto-discovers config file (`traversal.airflow.yaml`)
   - Creates Airflow DAG with DockerOperator per task
3. **When DAG runs**, each DockerOperator:
   - Starts container with specified image
   - Executes `runnable execute-single-node ...` with `--config` pointing to config file

---

## Key Reference Files

- **Pipeline loading:** `runnable/context.py:52-62` - `get_pipeline_spec_from_python()`
- **Command formats:** `runnable/context.py:331-416` - `get_node_callable_command()`, `get_fan_command()`
- **CLI definitions:** `runnable/cli.py:98-226` - `execute-single-node`, `fan` commands
- **Existing executor reference:** `extensions/pipeline_executor/argo.py` - ArgoExecutor pattern

## CLI Command Formats

**execute-single-node:**
```
runnable execute-single-node {run_id} {pipeline_file} {step_name} --mode python [--config {config}] [--iter-variable '{json}'] [--init-run-log] [--parameters-file {params}]
```

**fan:**
```
runnable fan {run_id} {step_name} {pipeline_file} {in|out} --mode python [--config-file {config}] [--iter-variable '{json}'] [--init-run-log] [--parameters-file {params}]
```

## Parameters and Run Log Initialization

### Problem
- The first step must initialize the run log with parameters from `parameters_file`
- Argo uses `error_on_existing_run_id=true` env var for first container only (in `_set_up_initial_container`)
- This approach is Argo-specific and doesn't translate well to Airflow

### Solution
Add `--init-run-log` CLI flag to `execute-single-node` with env var fallback for backward compatibility:

1. **CLI change**: Add `--init-run-log` flag that also reads from `error_on_existing_run_id` env var
2. **Backward compatible**: Argo continues using env var, Airflow uses CLI flag
3. **Factory change**: For `graph.start_at` node only, add `--init-run-log --parameters-file {params}`
4. **Minimal damage**: Existing Argo code continues to work unchanged

**Airflow (CLI flag):**
```
runnable execute-single-node {run_id} {pipeline} {step} --mode python --config {config} \
    --init-run-log --parameters-file {params}
```

**Argo (env var - continues to work):**
```
error_on_existing_run_id=true runnable execute-single-node ...
```

### Parameters File Convention
Similar to config: `{pipeline_name}.parameters.yaml` alongside pipeline (optional).

---

### Task 0: Add --init-run-log CLI Flag (Backward Compatible)

**Files:**
- Modify: `runnable/cli.py`

**Step 1: Add --init-run-log option with envvar fallback**

In `runnable/cli.py`, add to `execute_single_node` function:

```python
@click.option(
    "--init-run-log",
    is_flag=True,
    default=False,
    envvar="error_on_existing_run_id",  # Backward compatible with Argo
    help="Initialize run log (first step only). Also reads from error_on_existing_run_id env var.",
)
@click.option(
    "--parameters-file",
    default=None,
    help="Path to parameters YAML file (used with --init-run-log).",
)
```

**Step 2: Set env var when flag is present (for downstream code)**

In the `execute_single_node` function body, before any run log setup:

```python
if init_run_log:
    os.environ["error_on_existing_run_id"] = "true"
```

This ensures existing code that checks the env var continues to work.

**Step 3: Pass parameters_file to _set_up_run_log**

Ensure parameters_file is passed to `_set_up_run_log` when `--init-run-log` is set.

**Step 4: Test CLI flag**

Run: `uv run runnable execute-single-node --help`

Expected: Shows `--init-run-log` and `--parameters-file` options

**Step 5: Test env var backward compatibility**

Run: `error_on_existing_run_id=true uv run runnable execute-single-node --help`

Expected: The flag should be recognized from env var

**Step 6: Commit**

```bash
git add runnable/cli.py
git commit -m "feat(cli): add --init-run-log flag with envvar fallback for backward compatibility"
```

---

### Task 1: Delete Existing Implementation and Create Test File

**Files:**
- Delete: `extensions/pipeline_executor/airflow.py`
- Create: `tests/extensions/pipeline_executor/test_airflow.py`

**Step 1: Delete the broken implementation**

Run: `rm extensions/pipeline_executor/airflow.py`

**Step 2: Create empty test file with imports**

Create `tests/extensions/pipeline_executor/test_airflow.py`:

```python
"""Tests for AirflowDagFactory."""

import pytest
from unittest.mock import MagicMock, patch
```

**Step 3: Verify deletion**

Run: `ls extensions/pipeline_executor/airflow.py 2>&1 || echo "File deleted successfully"`

Expected: "File deleted successfully"

**Step 4: Commit**

```bash
git add -A
git commit -m "chore(airflow): remove broken implementation, add test file"
```

---

### Task 2: Create Base Factory Class with Import Guard

**Files:**
- Create: `extensions/pipeline_executor/airflow.py`
- Modify: `tests/extensions/pipeline_executor/test_airflow.py`

**Step 1: Write failing test for import guard**

Add to `tests/extensions/pipeline_executor/test_airflow.py`:

```python
def test_airflow_unavailable_raises_import_error():
    """Test that ImportError is raised when Airflow is not installed."""
    with patch.dict("sys.modules", {"airflow": None}):
        # Force reimport
        import importlib
        import extensions.pipeline_executor.airflow as airflow_module

        with pytest.raises(ImportError, match="Airflow is not installed"):
            importlib.reload(airflow_module)
            airflow_module.AirflowDagFactory(image="test:latest")
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/extensions/pipeline_executor/test_airflow.py::test_airflow_unavailable_raises_import_error -v`

Expected: FAIL (module doesn't exist)

**Step 3: Create airflow.py with import guard and base class**

Create `extensions/pipeline_executor/airflow.py`:

```python
"""
Airflow DAG Factory for Runnable pipelines.

Creates native Airflow DAGs from Runnable pipeline definitions at import time.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

from pydantic import BaseModel, Field

from runnable import defaults

logger = logging.getLogger(defaults.LOGGER_NAME)

# Guard imports for optional Airflow dependency
try:
    from airflow import DAG
    from airflow.operators.empty import EmptyOperator
    from airflow.operators.python import BranchPythonOperator
    from airflow.providers.docker.operators.docker import DockerOperator
    from airflow.utils.task_group import TaskGroup
    from airflow.utils.trigger_rule import TriggerRule

    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False
    if TYPE_CHECKING:
        from airflow import DAG
        from airflow.operators.empty import EmptyOperator
        from airflow.operators.python import BranchPythonOperator
        from airflow.providers.docker.operators.docker import DockerOperator
        from airflow.utils.task_group import TaskGroup
        from airflow.utils.trigger_rule import TriggerRule


def _check_airflow_available():
    """Raise ImportError if Airflow is not available."""
    if not AIRFLOW_AVAILABLE:
        raise ImportError(
            "Airflow is not installed. Install with: "
            "pip install 'runnable[airflow]' or "
            "pip install apache-airflow apache-airflow-providers-docker"
        )


class AirflowDagFactory(BaseModel):
    """
    Factory for creating Airflow DAGs from Runnable pipelines.

    Config files are auto-discovered using the convention:
    {pipeline_name}.airflow.yaml in the same directory as the pipeline.

    Docker configuration (image, volumes, etc.) is read from the config file,
    not from the factory constructor. This keeps per-pipeline config in one place.

    Example:
        factory = AirflowDagFactory()
        dag = factory.create_dag("examples/pipeline.py", dag_id="my-dag")
        # Auto-discovers: examples/pipeline.airflow.yaml
        # Docker settings come from executor.config in the YAML
    """

    # Airflow DAG defaults only - Docker config comes from per-pipeline YAML
    default_args: dict[str, Any] = Field(default_factory=dict)
    schedule: Optional[str] = Field(default=None)
    catchup: bool = Field(default=False)
    tags: list[str] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, **data):
        _check_airflow_available()
        super().__init__(**data)

    @staticmethod
    def _get_config_file(pipeline_file: str) -> Optional[str]:
        """
        Auto-discover config file for a pipeline.

        Convention: {pipeline_name}.airflow.yaml in same directory.
        Example: examples/traversal.py -> examples/traversal.airflow.yaml
        """
        import os

        base = pipeline_file.rsplit(".", 1)[0]  # Remove .py extension
        config_path = f"{base}.airflow.yaml"

        if os.path.exists(config_path):
            return config_path

        logger.warning(f"No config file found at {config_path}")
        return None

    @staticmethod
    def _get_parameters_file(pipeline_file: str) -> Optional[str]:
        """
        Auto-discover parameters file for a pipeline.

        Convention: {pipeline_name}.parameters.yaml in same directory.
        Example: examples/traversal.py -> examples/traversal.parameters.yaml
        """
        import os

        base = pipeline_file.rsplit(".", 1)[0]  # Remove .py extension
        params_path = f"{base}.parameters.yaml"

        if os.path.exists(params_path):
            return params_path

        return None  # Parameters file is optional, no warning

    @staticmethod
    def _load_docker_config(config_file: str) -> dict[str, Any]:
        """
        Load Docker configuration from the airflow config file.

        Reads executor.config section from the YAML.
        """
        import yaml

        if not config_file:
            raise ValueError("Config file is required for Docker settings")

        with open(config_file, "r") as f:
            config = yaml.safe_load(f)

        executor_config = config.get("executor", {}).get("config", {})

        if not executor_config.get("image"):
            raise ValueError(f"executor.config.image is required in {config_file}")

        # Return with defaults
        return {
            "image": executor_config["image"],
            "docker_url": executor_config.get("docker_url", "unix://var/run/docker.sock"),
            "network_mode": executor_config.get("network_mode", "bridge"),
            "auto_remove": executor_config.get("auto_remove", "success"),
            "mount_tmp_dir": executor_config.get("mount_tmp_dir", False),
            "volumes": executor_config.get("volumes", []),
            "environment": executor_config.get("environment", {}),
            "overrides": executor_config.get("overrides", {}),
        }
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/extensions/pipeline_executor/test_airflow.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add extensions/pipeline_executor/airflow.py tests/extensions/pipeline_executor/test_airflow.py
git commit -m "feat(airflow): add AirflowDagFactory base class with import guard"
```

---

### Task 3: Implement create_dag Method

**Files:**
- Modify: `extensions/pipeline_executor/airflow.py`
- Modify: `tests/extensions/pipeline_executor/test_airflow.py`

**Step 1: Write failing test for create_dag**

Add to `tests/extensions/pipeline_executor/test_airflow.py`:

```python
def test_create_dag_loads_graph_and_creates_dag():
    """Test create_dag loads pipeline graph and returns Airflow DAG."""
    with patch("extensions.pipeline_executor.airflow.AIRFLOW_AVAILABLE", True):
        with patch("extensions.pipeline_executor.airflow.DAG") as mock_dag_class:
            with patch(
                "extensions.pipeline_executor.airflow.get_pipeline_spec_from_python"
            ) as mock_load:
                # Setup mock graph with simple success node
                mock_graph = MagicMock()
                mock_graph.start_at = "success"
                mock_node = MagicMock()
                mock_node.node_type = "success"
                mock_node.internal_name = "success"
                mock_graph.get_node_by_name.return_value = mock_node
                mock_load.return_value = mock_graph

                mock_dag = MagicMock()
                mock_dag_class.return_value.__enter__ = MagicMock(
                    return_value=mock_dag
                )
                mock_dag_class.return_value.__exit__ = MagicMock(return_value=False)

                from extensions.pipeline_executor.airflow import AirflowDagFactory

                factory = AirflowDagFactory(image="test:latest")
                result = factory.create_dag(
                    pipeline_file="examples/test.py",
                    dag_id="test-dag",
                )

                mock_load.assert_called_once_with("examples/test.py")
                mock_dag_class.assert_called_once()
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/extensions/pipeline_executor/test_airflow.py::test_create_dag_loads_graph_and_creates_dag -v`

Expected: FAIL (create_dag not implemented)

**Step 3: Implement create_dag method**

Add to `AirflowDagFactory` class in `extensions/pipeline_executor/airflow.py`:

```python
    def create_dag(
        self,
        pipeline_file: str,
        dag_id: str,
        image: Optional[str] = None,
        schedule: Optional[str] = None,
        **dag_kwargs,
    ) -> "DAG":
        """
        Create an Airflow DAG from a Runnable pipeline file.

        Config file is auto-discovered: {pipeline_name}.airflow.yaml

        Args:
            pipeline_file: Path to the Runnable pipeline Python file
            dag_id: Unique identifier for the DAG
            image: Override Docker image for this DAG
            schedule: Override schedule for this DAG
            **dag_kwargs: Additional DAG constructor arguments

        Returns:
            Airflow DAG object
        """
        from runnable.context import get_pipeline_spec_from_python

        # Load pipeline graph
        graph = get_pipeline_spec_from_python(pipeline_file)

        # Auto-discover config and parameters files
        config_file = self._get_config_file(pipeline_file)
        parameters_file = self._get_parameters_file(pipeline_file)

        # Load Docker config from YAML
        docker_config = self._load_docker_config(config_file)

        # Resolve configuration (allow per-DAG image override)
        effective_image = image or docker_config["image"]
        effective_schedule = schedule if schedule is not None else self.schedule

        # Build DAG configuration
        dag_config = {
            "dag_id": dag_id,
            "default_args": {**self.default_args, **dag_kwargs.pop("default_args", {})},
            "schedule": effective_schedule,
            "catchup": dag_kwargs.pop("catchup", self.catchup),
            "tags": dag_kwargs.pop("tags", self.tags),
            **dag_kwargs,
        }

        # Create DAG and build tasks
        with DAG(**dag_config) as dag:
            self._build_dag_from_graph(
                dag=dag,
                graph=graph,
                pipeline_file=pipeline_file,
                docker_config=docker_config,
                config_file=config_file,
                parameters_file=parameters_file,
                start_node_name=graph.start_at,  # Track first node
            )

        return dag

    def _build_dag_from_graph(
        self,
        dag: "DAG",
        graph: Any,
        pipeline_file: str,
        image: str,
        config_file: Optional[str] = None,
    ) -> None:
        """Build Airflow tasks from Runnable graph. Placeholder for now."""
        pass
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/extensions/pipeline_executor/test_airflow.py::test_create_dag_loads_graph_and_creates_dag -v`

Expected: PASS

**Step 5: Commit**

```bash
git add extensions/pipeline_executor/airflow.py tests/extensions/pipeline_executor/test_airflow.py
git commit -m "feat(airflow): implement create_dag method with graph loading"
```

---

### Task 4: Implement Linear Graph Traversal

**Files:**
- Modify: `extensions/pipeline_executor/airflow.py`
- Modify: `tests/extensions/pipeline_executor/test_airflow.py`

**Step 1: Write failing test for linear graph traversal**

Add to `tests/extensions/pipeline_executor/test_airflow.py`:

```python
def test_build_dag_creates_tasks_for_linear_graph():
    """Test that _build_dag_from_graph creates tasks for each node."""
    with patch("extensions.pipeline_executor.airflow.AIRFLOW_AVAILABLE", True):
        with patch("extensions.pipeline_executor.airflow.EmptyOperator") as mock_empty:
            with patch(
                "extensions.pipeline_executor.airflow.DockerOperator"
            ) as mock_docker:
                from extensions.pipeline_executor.airflow import AirflowDagFactory

                factory = AirflowDagFactory(image="test:latest")

                # Create mock graph: task1 -> success
                mock_graph = MagicMock()
                mock_graph.start_at = "task1"

                task_node = MagicMock()
                task_node.node_type = "task"
                task_node.internal_name = "task1"
                task_node._command_friendly_name.return_value = "task1"
                task_node._get_next_node.return_value = "success"

                success_node = MagicMock()
                success_node.node_type = "success"
                success_node.internal_name = "success"

                def get_node(name):
                    return task_node if name == "task1" else success_node

                mock_graph.get_node_by_name.side_effect = get_node

                mock_dag = MagicMock()

                factory._build_dag_from_graph(
                    dag=mock_dag,
                    graph=mock_graph,
                    pipeline_file="test.py",
                    image="test:latest",
                )

                # Should have created DockerOperator for task and EmptyOperator for success
                assert mock_docker.called
                assert mock_empty.called
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/extensions/pipeline_executor/test_airflow.py::test_build_dag_creates_tasks_for_linear_graph -v`

Expected: FAIL

**Step 3: Implement _build_dag_from_graph**

Replace the placeholder in `extensions/pipeline_executor/airflow.py`:

```python
    def _build_dag_from_graph(
        self,
        dag: "DAG",
        graph: Any,
        pipeline_file: str,
        docker_config: dict[str, Any],
        config_file: Optional[str] = None,
        parameters_file: Optional[str] = None,
        start_node_name: Optional[str] = None,
        iter_variable: Optional[dict] = None,
    ) -> tuple[Any, Any]:
        """
        Build Airflow tasks from Runnable graph.

        Args:
            docker_config: Docker settings from executor.config in YAML
            start_node_name: The first node of the top-level graph (for --init-run-log)

        Returns:
            Tuple of (first_task, last_task) for dependency chaining
        """
        current_node_name = graph.start_at
        first_task = None
        previous_task = None

        while current_node_name:
            node = graph.get_node_by_name(current_node_name)
            task_id = self._sanitize_task_id(node.internal_name)

            # Check if this is the very first node (needs --init-run-log)
            is_first_node = (current_node_name == start_node_name)

            # Get per-step image override if configured
            step_overrides = docker_config.get("overrides", {}).get(node.internal_name, {})
            step_image = step_overrides.get("image", docker_config["image"])

            match node.node_type:
                case "task" | "stub":
                    task = self._create_docker_task(
                        node=node,
                        task_id=task_id,
                        pipeline_file=pipeline_file,
                        docker_config=docker_config,
                        step_image=step_image,
                        config_file=config_file,
                        parameters_file=parameters_file if is_first_node else None,
                        init_run_log=is_first_node,
                        iter_variable=iter_variable,
                    )

                case "success":
                    task = EmptyOperator(
                        task_id=task_id,
                        trigger_rule=TriggerRule.ALL_SUCCESS,
                    )

                case "fail":
                    task = EmptyOperator(
                        task_id=task_id,
                        trigger_rule=TriggerRule.ONE_FAILED,
                    )

                case _:
                    raise ValueError(f"Unsupported node type: {node.node_type}")

            if first_task is None:
                first_task = task

            if previous_task is not None:
                previous_task >> task

            previous_task = task

            # Terminal nodes end traversal
            if node.node_type in ("success", "fail"):
                break

            try:
                current_node_name = node._get_next_node()
            except Exception:
                break

        return first_task, previous_task

    def _sanitize_task_id(self, name: str) -> str:
        """Sanitize name for valid Airflow task ID."""
        sanitized = name.replace(" ", "_").replace(".", "_")
        return "".join(c for c in sanitized if c.isalnum() or c in "_-")

    def _create_docker_task(
        self,
        node: Any,
        task_id: str,
        pipeline_file: str,
        docker_config: dict[str, Any],
        step_image: str,
        config_file: Optional[str] = None,
        parameters_file: Optional[str] = None,
        init_run_log: bool = False,
        iter_variable: Optional[dict] = None,
    ) -> "DockerOperator":
        """Create DockerOperator for a task node."""
        # Build command: runnable execute-single-node {run_id} {file} {step} --mode python
        command = self._build_execute_command(
            node=node,
            pipeline_file=pipeline_file,
            config_file=config_file,
            parameters_file=parameters_file,
            init_run_log=init_run_log,
            iter_variable=iter_variable,
        )

        return DockerOperator(
            task_id=task_id,
            image=step_image,
            command=command,
            docker_url=docker_config["docker_url"],
            network_mode=docker_config["network_mode"],
            auto_remove=docker_config["auto_remove"],
            mount_tmp_dir=docker_config["mount_tmp_dir"],
            mounts=docker_config["volumes"],
            environment=docker_config["environment"],
        )

    def _build_execute_command(
        self,
        node: Any,
        pipeline_file: str,
        config_file: Optional[str] = None,
        parameters_file: Optional[str] = None,
        init_run_log: bool = False,
        iter_variable: Optional[dict] = None,
    ) -> str:
        """Build execute-single-node command string."""
        import json

        cmd = (
            f"runnable execute-single-node "
            f"{{{{ run_id }}}} "
            f"{pipeline_file} "
            f"{node._command_friendly_name()} "
            f"--mode python"
        )

        if config_file:
            cmd += f" --config {config_file}"

        # First node only: initialize run log with parameters
        if init_run_log:
            cmd += " --init-run-log"
            if parameters_file:
                cmd += f" --parameters-file {parameters_file}"

        if iter_variable:
            iter_json = json.dumps({"map_variable": iter_variable})
            cmd += f" --iter-variable '{iter_json}'"

        return cmd
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/extensions/pipeline_executor/test_airflow.py::test_build_dag_creates_tasks_for_linear_graph -v`

Expected: PASS

**Step 5: Commit**

```bash
git add extensions/pipeline_executor/airflow.py tests/extensions/pipeline_executor/test_airflow.py
git commit -m "feat(airflow): implement linear graph traversal with DockerOperator"
```

---

### Task 5: Implement Parallel Node Support

**Files:**
- Modify: `extensions/pipeline_executor/airflow.py`
- Modify: `tests/extensions/pipeline_executor/test_airflow.py`

**Step 1: Write failing test for parallel node**

Add to `tests/extensions/pipeline_executor/test_airflow.py`:

```python
def test_parallel_node_creates_task_group_with_branches():
    """Test parallel node creates TaskGroup with fan-out, branches, fan-in."""
    with patch("extensions.pipeline_executor.airflow.AIRFLOW_AVAILABLE", True):
        with patch("extensions.pipeline_executor.airflow.TaskGroup") as mock_tg:
            with patch(
                "extensions.pipeline_executor.airflow.DockerOperator"
            ) as mock_docker:
                with patch(
                    "extensions.pipeline_executor.airflow.EmptyOperator"
                ):
                    from extensions.pipeline_executor.airflow import AirflowDagFactory
                    from extensions.nodes.parallel import ParallelNode

                    factory = AirflowDagFactory(image="test:latest")

                    # Create mock parallel node with two branches
                    parallel_node = MagicMock(spec=ParallelNode)
                    parallel_node.node_type = "parallel"
                    parallel_node.internal_name = "parallel_step"
                    parallel_node._command_friendly_name.return_value = "parallel_step"
                    parallel_node._get_next_node.return_value = "success"

                    # Each branch is a mini-graph
                    branch_a = MagicMock()
                    branch_a.start_at = "task_a"
                    task_a = MagicMock()
                    task_a.node_type = "success"
                    task_a.internal_name = "success"
                    branch_a.get_node_by_name.return_value = task_a

                    branch_b = MagicMock()
                    branch_b.start_at = "task_b"
                    task_b = MagicMock()
                    task_b.node_type = "success"
                    task_b.internal_name = "success"
                    branch_b.get_node_by_name.return_value = task_b

                    parallel_node.branches = {"branch_a": branch_a, "branch_b": branch_b}

                    mock_graph = MagicMock()
                    mock_graph.start_at = "parallel_step"

                    success_node = MagicMock()
                    success_node.node_type = "success"
                    success_node.internal_name = "success"

                    def get_node(name):
                        if name == "parallel_step":
                            return parallel_node
                        return success_node

                    mock_graph.get_node_by_name.side_effect = get_node

                    # Mock TaskGroup context manager
                    mock_tg.return_value.__enter__ = MagicMock()
                    mock_tg.return_value.__exit__ = MagicMock(return_value=False)

                    factory._build_dag_from_graph(
                        dag=MagicMock(),
                        graph=mock_graph,
                        pipeline_file="test.py",
                        image="test:latest",
                    )

                    # TaskGroup should be created for parallel node
                    mock_tg.assert_called()
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/extensions/pipeline_executor/test_airflow.py::test_parallel_node_creates_task_group_with_branches -v`

Expected: FAIL (parallel not handled in match statement)

**Step 3: Add parallel node handling**

Add to the match statement in `_build_dag_from_graph`:

```python
                case "parallel":
                    task = self._create_parallel_group(
                        dag=dag,
                        node=node,
                        task_id=task_id,
                        pipeline_file=pipeline_file,
                        image=image,
                        config_file=config_file,
                        parameters_file=parameters_file if is_first_node else None,
                        init_run_log=is_first_node,
                        iter_variable=iter_variable,
                    )
```

Add the helper method:

```python
    def _create_parallel_group(
        self,
        dag: "DAG",
        node: Any,
        task_id: str,
        pipeline_file: str,
        image: str,
        config_file: Optional[str] = None,
        parameters_file: Optional[str] = None,
        init_run_log: bool = False,
        iter_variable: Optional[dict] = None,
    ) -> "TaskGroup":
        """Create TaskGroup for parallel node with fan-out/fan-in."""
        with TaskGroup(group_id=task_id) as group:
            # Fan-out
            fan_out = self._create_fan_task(
                node=node,
                task_id=f"{task_id}_fan_out",
                mode="out",
                pipeline_file=pipeline_file,
                image=image,
                config_file=config_file,
                parameters_file=parameters_file,
                init_run_log=init_run_log,
                iter_variable=iter_variable,
            )

            # Build each branch
            branch_ends = []
            for branch_name, branch_graph in node.branches.items():
                branch_id = self._sanitize_task_id(branch_name)
                with TaskGroup(group_id=branch_id):
                    first, last = self._build_dag_from_graph(
                        dag=dag,
                        graph=branch_graph,
                        pipeline_file=pipeline_file,
                        image=image,
                        config_file=config_file,
                        iter_variable=iter_variable,
                    )
                    if first:
                        fan_out >> first
                    if last:
                        branch_ends.append(last)

            # Fan-in
            fan_in = self._create_fan_task(
                node=node,
                task_id=f"{task_id}_fan_in",
                mode="in",
                pipeline_file=pipeline_file,
                image=image,
                config_file=config_file,
                iter_variable=iter_variable,
                trigger_rule=TriggerRule.ALL_DONE,
            )

            for end in branch_ends:
                end >> fan_in

        return group

    def _create_fan_task(
        self,
        node: Any,
        task_id: str,
        mode: str,
        pipeline_file: str,
        image: str,
        config_file: Optional[str] = None,
        parameters_file: Optional[str] = None,
        init_run_log: bool = False,
        iter_variable: Optional[dict] = None,
        trigger_rule: Optional["TriggerRule"] = None,
    ) -> "DockerOperator":
        """Create fan-out or fan-in DockerOperator."""
        command = self._build_fan_command(
            node=node,
            mode=mode,
            pipeline_file=pipeline_file,
            config_file=config_file,
            parameters_file=parameters_file,
            init_run_log=init_run_log,
            iter_variable=iter_variable,
        )

        kwargs = {
            "task_id": task_id,
            "image": image,
            "command": command,
            "docker_url": self.docker_url,
            "network_mode": self.network_mode,
            "auto_remove": self.auto_remove,
            "mount_tmp_dir": self.mount_tmp_dir,
            "environment": self.environment,
        }

        if trigger_rule:
            kwargs["trigger_rule"] = trigger_rule

        return DockerOperator(**kwargs)

    def _build_fan_command(
        self,
        node: Any,
        mode: str,
        pipeline_file: str,
        config_file: Optional[str] = None,
        parameters_file: Optional[str] = None,
        init_run_log: bool = False,
        iter_variable: Optional[dict] = None,
    ) -> str:
        """Build fan in/out command string."""
        import json

        cmd = (
            f"runnable fan "
            f"{{{{ run_id }}}} "
            f"{node._command_friendly_name()} "
            f"{pipeline_file} "
            f"{mode} "
            f"--mode python"
        )

        if config_file:
            cmd += f" --config-file {config_file}"

        # First step only: initialize run log with parameters
        if init_run_log:
            cmd += " --init-run-log"
            if parameters_file:
                cmd += f" --parameters-file {parameters_file}"

        if iter_variable:
            iter_json = json.dumps({"map_variable": iter_variable})
            cmd += f" --iter-variable '{iter_json}'"

        return cmd
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/extensions/pipeline_executor/test_airflow.py::test_parallel_node_creates_task_group_with_branches -v`

Expected: PASS

**Step 5: Commit**

```bash
git add extensions/pipeline_executor/airflow.py tests/extensions/pipeline_executor/test_airflow.py
git commit -m "feat(airflow): add parallel node support with TaskGroup"
```

---

### Task 6: Implement Map Node Support

**Files:**
- Modify: `extensions/pipeline_executor/airflow.py`
- Modify: `tests/extensions/pipeline_executor/test_airflow.py`

**Step 1: Write failing test for map node**

Add to `tests/extensions/pipeline_executor/test_airflow.py`:

```python
def test_map_node_creates_dynamic_task_mapping():
    """Test map node uses dynamic task mapping with XCom."""
    with patch("extensions.pipeline_executor.airflow.AIRFLOW_AVAILABLE", True):
        with patch("extensions.pipeline_executor.airflow.TaskGroup") as mock_tg:
            with patch(
                "extensions.pipeline_executor.airflow.DockerOperator"
            ) as mock_docker:
                from extensions.pipeline_executor.airflow import AirflowDagFactory
                from extensions.nodes.map import MapNode

                factory = AirflowDagFactory(image="test:latest")

                # Create mock map node
                map_node = MagicMock(spec=MapNode)
                map_node.node_type = "map"
                map_node.internal_name = "map_step"
                map_node._command_friendly_name.return_value = "map_step"
                map_node._get_next_node.return_value = "success"
                map_node.iterate_on = "items"
                map_node.iterate_as = "item"

                # Branch graph
                branch = MagicMock()
                branch.start_at = "process"
                process_node = MagicMock()
                process_node.node_type = "success"
                process_node.internal_name = "success"
                branch.get_node_by_name.return_value = process_node
                map_node.branch = branch

                mock_graph = MagicMock()
                mock_graph.start_at = "map_step"

                success_node = MagicMock()
                success_node.node_type = "success"
                success_node.internal_name = "success"

                def get_node(name):
                    if name == "map_step":
                        return map_node
                    return success_node

                mock_graph.get_node_by_name.side_effect = get_node

                mock_tg.return_value.__enter__ = MagicMock()
                mock_tg.return_value.__exit__ = MagicMock(return_value=False)

                factory._build_dag_from_graph(
                    dag=MagicMock(),
                    graph=mock_graph,
                    pipeline_file="test.py",
                    image="test:latest",
                )

                # Should create TaskGroup for map
                mock_tg.assert_called()
                # Should create DockerOperator with do_xcom_push for fan-out
                calls = mock_docker.call_args_list
                fan_out_calls = [c for c in calls if "fan_out" in str(c)]
                assert len(fan_out_calls) > 0
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/extensions/pipeline_executor/test_airflow.py::test_map_node_creates_dynamic_task_mapping -v`

Expected: FAIL

**Step 3: Add map node handling**

Add to the match statement in `_build_dag_from_graph`:

```python
                case "map":
                    task = self._create_map_group(
                        dag=dag,
                        node=node,
                        task_id=task_id,
                        pipeline_file=pipeline_file,
                        image=image,
                        config_file=config_file,
                        parameters_file=parameters_file if is_first_node else None,
                        init_run_log=is_first_node,
                        iter_variable=iter_variable,
                    )
```

Add the helper method:

```python
    def _create_map_group(
        self,
        dag: "DAG",
        node: Any,
        task_id: str,
        pipeline_file: str,
        image: str,
        config_file: Optional[str] = None,
        parameters_file: Optional[str] = None,
        init_run_log: bool = False,
        iter_variable: Optional[dict] = None,
    ) -> "TaskGroup":
        """Create TaskGroup for map node with dynamic task mapping."""
        iterate_as = node.iterate_as

        with TaskGroup(group_id=task_id) as group:
            # Fan-out with XCom push
            fan_out = self._create_fan_task(
                node=node,
                task_id=f"{task_id}_fan_out",
                mode="out",
                pipeline_file=pipeline_file,
                image=image,
                config_file=config_file,
                parameters_file=parameters_file,
                init_run_log=init_run_log,
                iter_variable=iter_variable,
            )
            # Enable XCom for iteration values
            fan_out.do_xcom_push = True

            # Build branch with map variable
            branch_iter = iter_variable.copy() if iter_variable else {}
            # Template for dynamic iteration value
            branch_iter[iterate_as] = {
                "value": f"{{{{ ti.xcom_pull(task_ids='{task_id}_fan_out')[ti.map_index] }}}}"
            }

            # For map nodes, we recursively build the branch
            # The branch tasks will be expanded dynamically by Airflow
            first, last = self._build_dag_from_graph(
                dag=dag,
                graph=node.branch,
                pipeline_file=pipeline_file,
                image=image,
                config_file=config_file,
                iter_variable=branch_iter,
            )

            if first:
                fan_out >> first

            # Fan-in
            fan_in = self._create_fan_task(
                node=node,
                task_id=f"{task_id}_fan_in",
                mode="in",
                pipeline_file=pipeline_file,
                image=image,
                config_file=config_file,
                iter_variable=iter_variable,
                trigger_rule=TriggerRule.ALL_DONE,
            )

            if last:
                last >> fan_in

        return group
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/extensions/pipeline_executor/test_airflow.py::test_map_node_creates_dynamic_task_mapping -v`

Expected: PASS

**Step 5: Commit**

```bash
git add extensions/pipeline_executor/airflow.py tests/extensions/pipeline_executor/test_airflow.py
git commit -m "feat(airflow): add map node support with dynamic task mapping"
```

---

### Task 7: Implement Conditional Node Support

**Files:**
- Modify: `extensions/pipeline_executor/airflow.py`
- Modify: `tests/extensions/pipeline_executor/test_airflow.py`

**Step 1: Write failing test for conditional node**

Add to `tests/extensions/pipeline_executor/test_airflow.py`:

```python
def test_conditional_node_creates_branch_operator():
    """Test conditional node creates BranchPythonOperator."""
    with patch("extensions.pipeline_executor.airflow.AIRFLOW_AVAILABLE", True):
        with patch("extensions.pipeline_executor.airflow.TaskGroup") as mock_tg:
            with patch(
                "extensions.pipeline_executor.airflow.BranchPythonOperator"
            ) as mock_branch:
                with patch("extensions.pipeline_executor.airflow.DockerOperator"):
                    from extensions.pipeline_executor.airflow import AirflowDagFactory
                    from extensions.nodes.conditional import ConditionalNode

                    factory = AirflowDagFactory(image="test:latest")

                    cond_node = MagicMock(spec=ConditionalNode)
                    cond_node.node_type = "conditional"
                    cond_node.internal_name = "cond_step"
                    cond_node._command_friendly_name.return_value = "cond_step"
                    cond_node._get_next_node.return_value = "success"

                    branch_a = MagicMock()
                    branch_a.start_at = "a"
                    node_a = MagicMock()
                    node_a.node_type = "success"
                    node_a.internal_name = "success"
                    branch_a.get_node_by_name.return_value = node_a

                    cond_node.branches = {"a": branch_a}

                    mock_graph = MagicMock()
                    mock_graph.start_at = "cond_step"

                    success_node = MagicMock()
                    success_node.node_type = "success"
                    success_node.internal_name = "success"

                    def get_node(name):
                        if name == "cond_step":
                            return cond_node
                        return success_node

                    mock_graph.get_node_by_name.side_effect = get_node

                    mock_tg.return_value.__enter__ = MagicMock()
                    mock_tg.return_value.__exit__ = MagicMock(return_value=False)

                    factory._build_dag_from_graph(
                        dag=MagicMock(),
                        graph=mock_graph,
                        pipeline_file="test.py",
                        image="test:latest",
                    )

                    mock_branch.assert_called()
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/extensions/pipeline_executor/test_airflow.py::test_conditional_node_creates_branch_operator -v`

Expected: FAIL

**Step 3: Add conditional node handling**

Add to the match statement:

```python
                case "conditional":
                    task = self._create_conditional_group(
                        dag=dag,
                        node=node,
                        task_id=task_id,
                        pipeline_file=pipeline_file,
                        image=image,
                        config_file=config_file,
                        parameters_file=parameters_file if is_first_node else None,
                        init_run_log=is_first_node,
                        iter_variable=iter_variable,
                    )
```

Add the helper:

```python
    def _create_conditional_group(
        self,
        dag: "DAG",
        node: Any,
        task_id: str,
        pipeline_file: str,
        image: str,
        config_file: Optional[str] = None,
        parameters_file: Optional[str] = None,
        init_run_log: bool = False,
        iter_variable: Optional[dict] = None,
    ) -> "TaskGroup":
        """Create TaskGroup for conditional node with BranchPythonOperator."""
        with TaskGroup(group_id=task_id) as group:
            # Fan-out determines which branch
            fan_out = self._create_fan_task(
                node=node,
                task_id=f"{task_id}_fan_out",
                mode="out",
                pipeline_file=pipeline_file,
                image=image,
                config_file=config_file,
                parameters_file=parameters_file,
                init_run_log=init_run_log,
                iter_variable=iter_variable,
            )
            fan_out.do_xcom_push = True

            # Branch selector
            def make_branch_selector(task_id_prefix: str):
                def branch_selector(**context):
                    ti = context["ti"]
                    selected = ti.xcom_pull(task_ids=f"{task_id_prefix}_fan_out")
                    return f"{task_id_prefix}.{selected}"

                return branch_selector

            branch_op = BranchPythonOperator(
                task_id=f"{task_id}_selector",
                python_callable=make_branch_selector(task_id),
            )

            fan_out >> branch_op

            # Build branches
            branch_ends = []
            for branch_name, branch_graph in node.branches.items():
                branch_id = self._sanitize_task_id(branch_name)
                with TaskGroup(group_id=branch_id):
                    first, last = self._build_dag_from_graph(
                        dag=dag,
                        graph=branch_graph,
                        pipeline_file=pipeline_file,
                        image=image,
                        config_file=config_file,
                        iter_variable=iter_variable,
                    )
                    if first:
                        branch_op >> first
                    if last:
                        branch_ends.append(last)

            # Fan-in
            fan_in = self._create_fan_task(
                node=node,
                task_id=f"{task_id}_fan_in",
                mode="in",
                pipeline_file=pipeline_file,
                image=image,
                config_file=config_file,
                iter_variable=iter_variable,
                trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
            )

            for end in branch_ends:
                end >> fan_in

        return group
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/extensions/pipeline_executor/test_airflow.py::test_conditional_node_creates_branch_operator -v`

Expected: PASS

**Step 5: Commit**

```bash
git add extensions/pipeline_executor/airflow.py tests/extensions/pipeline_executor/test_airflow.py
git commit -m "feat(airflow): add conditional node support with BranchPythonOperator"
```

---

### Task 8: Add pyproject.toml Dependency

**Files:**
- Modify: `pyproject.toml`

**Step 1: Add airflow dependency group**

In `pyproject.toml`, add after the `telemetry` dependency (around line 43):

```toml
airflow = [
    "apache-airflow>=2.7.0",
    "apache-airflow-providers-docker>=3.0.0",
]
```

**Step 2: Verify the addition**

Run: `grep -A3 'airflow = \[' pyproject.toml`

Expected: Shows airflow dependencies

**Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "feat(airflow): add optional airflow dependency group"
```

---

### Task 9: Create Example Files

**Files:**
- Create: `examples/airflow/dag_loader.py`
- Create: `examples/02-sequential/traversal.airflow.yaml`

**Step 1: Create directory and example DAG loader**

Run: `mkdir -p examples/airflow`

Create `examples/airflow/dag_loader.py`:

```python
"""
Example Airflow DAG loader for Runnable pipelines.

Place in your Airflow DAGs folder and configure the factory.

Config files are auto-discovered using the convention:
{pipeline_name}.airflow.yaml in the same directory as the pipeline.
"""

from extensions.pipeline_executor.airflow import AirflowDagFactory

factory = AirflowDagFactory(
    image="your-runnable-image:latest",
    volumes=[
        "/tmp/run_logs:/tmp/run_logs",
        "/tmp/catalog:/tmp/catalog",
    ],
    default_args={"owner": "runnable", "retries": 1},
    catchup=False,
    tags=["runnable"],
)

# Create DAG from pipeline
# Config auto-discovered: examples/02-sequential/traversal.airflow.yaml
traversal_dag = factory.create_dag(
    pipeline_file="examples/02-sequential/traversal.py",
    dag_id="runnable-traversal",
    schedule="@daily",
)
```

**Step 2: Create config file alongside pipeline**

Create `examples/02-sequential/traversal.airflow.yaml`:

```yaml
# Airflow-specific config for traversal.py
# This file is auto-discovered by AirflowDagFactory

# Runtime configuration (used by runnable CLI in containers)
run-log-store:
  type: file-system
  config:
    log_folder: /tmp/run_logs

catalog:
  type: file-system
  config:
    catalog_location: /tmp/catalog

secrets:
  type: do-nothing

# Airflow/Docker configuration (used by factory at DAG creation time)
executor:
  type: airflow
  config:
    image: your-runnable-image:latest
    docker_url: unix://var/run/docker.sock
    network_mode: bridge
    auto_remove: success
    mount_tmp_dir: false
    # Volume mounts - MUST share run-log and catalog across containers
    volumes:
      - /tmp/run_logs:/tmp/run_logs
      - /tmp/catalog:/tmp/catalog
    environment: {}
    # Optional: Per-step image overrides
    # overrides:
    #   step_name:
    #     image: custom-image:latest
```

**Step 3: Verify files exist**

Run: `ls -la examples/airflow/dag_loader.py examples/02-sequential/traversal.airflow.yaml`

**Step 4: Commit**

```bash
git add examples/airflow/ examples/02-sequential/traversal.airflow.yaml
git commit -m "docs(airflow): add example DAG loader and config"
```

---

### Task 10: Run Full Test Suite and Pre-commit

**Step 1: Run all airflow tests**

Run: `uv run pytest tests/extensions/pipeline_executor/test_airflow.py -v`

Expected: All tests pass

**Step 2: Run pre-commit**

Run: `uv run pre-commit run --all-files`

Expected: All checks pass

**Step 3: Final commit if needed**

```bash
git add -A
git commit -m "style: format airflow module"
```

---

## Design Gaps Status

### 1. Fan-out as first step ✅ FIXED
Added `--init-run-log` flag to `fan` CLI command with `envvar="error_on_existing_run_id"` for backward compatibility. Updated `entrypoints.fan()` to accept and handle the parameter.

### 2. Simplify run log existence check ✅ FIXED
The `--init-run-log` flag approach works well. When flag is set, the entrypoint sets `os.environ["error_on_existing_run_id"] = "true"` for downstream `_set_up_run_log` to consume.

### 3. Airflow executor in node callable command ✅ FIXED
Created `AirflowExecutor` class that:
- Inherits from `GenericPipelineExecutor`
- Overrides `execute_node`, `fan_out`, `fan_in` to call `_set_up_run_log()` before execution
- Registered in `pyproject.toml` entry points as `"airflow"`

### 4. Parameter file discovery ✅ FIXED
Changed from auto-discovery to explicit `parameters_file` and `config_file` fields in both `AirflowExecutor` and `AirflowDagFactory`. This provides clarity and flexibility.

### 5. Step override implementation concerns ⏳ PENDING
Still needs review. Current approach passes `docker_config` dict through the call chain.

---

## Future Considerations: KubernetesPodOperator Support

Currently implementing DockerOperator only. When KubernetesPodOperator is needed:

### Config Structure Extension

```yaml
executor:
  type: airflow
  config:
    operator: docker  # or: kubernetes
    image: my-image:latest
    environment:
      KEY: value

    # Docker-specific (ignored if operator: kubernetes)
    docker:
      docker_url: unix://var/run/docker.sock
      network_mode: bridge
      volumes:
        - /host/run_logs:/tmp/run_logs

    # Kubernetes-specific (ignored if operator: docker)
    kubernetes:
      namespace: airflow
      service_account_name: runnable
      in_cluster: true
      volumes:
        - name: run-logs
          persistentVolumeClaim:
            claimName: run-logs-pvc
      volume_mounts:
        - name: run-logs
          mountPath: /tmp/run_logs
```

### Code Changes Required

1. **`_load_docker_config`** → **`_load_operator_config`** - returns operator type + config
2. **`_create_docker_task`** → **`_create_operator_task`** - switches on operator type
3. **New method**: `_create_kubernetes_task` for K8s-specific logic

### Key Differences

| Aspect | DockerOperator | KubernetesPodOperator |
|--------|---------------|----------------------|
| Volumes | `mounts` (bind mounts) | `volumes` + `volume_mounts` (PVC) |
| Network | `docker_url`, `network_mode` | `namespace`, `in_cluster` |
| Resources | N/A | `resources` (CPU, memory) |

The abstraction point is `_create_operator_task` - graph traversal and command building stay unchanged.

---

## Implementation Notes

**Composite node handlers** (`_create_parallel_group`, `_create_map_group`, `_create_conditional_group`, `_create_fan_task`) need the same updates:
- Accept `docker_config: dict[str, Any]` instead of `image: str`
- Pass `docker_config` to recursive `_build_dag_from_graph` calls
- Use `docker_config["image"]` for fan tasks (or allow step overrides)

The pattern shown in `_build_dag_from_graph` and `_create_docker_task` should be applied consistently.

---

## Verification Checklist

- [x] CLI: `--init-run-log` flag added with `envvar="error_on_existing_run_id"` fallback (in both `execute_single_node` and `fan` commands)
- [x] CLI: `--parameters-file` flag added
- [x] Argo backward compatibility: env var still works
- [x] Pipeline loading via `runpy.run_path()` with context marker (replaces `get_pipeline_spec_from_python`)
- [ ] All commands include `--mode python`
- [ ] All commands use `node._command_friendly_name()`
- [x] Config file: explicit `config_file` field (changed from auto-discovery)
- [x] Parameters file: explicit `parameters_file` field (changed from auto-discovery)
- [ ] Docker config loaded from `executor.config` in YAML (not factory constructor)
- [ ] Volume mounts configured to share run-log and catalog across containers
- [ ] Config file passed to all CLI commands via `--config` / `--config-file`
- [ ] First step only: `--init-run-log --parameters-file {params}` added
- [ ] Per-step image overrides via `executor.config.overrides`
- [x] Linear graph traversal works (tested with stub pipeline: step1 → step2 → step3 → success)
- [ ] Parallel nodes create TaskGroup with fan-out/fan-in
- [ ] Map nodes support dynamic task mapping
- [ ] Conditional nodes use BranchPythonOperator
- [x] `pyproject.toml` has airflow dependency (`>=2.8.0,<3.0.0`)
- [ ] Example files created (dag_loader.py + traversal.airflow.yaml)
- [x] All tests pass (479 passed, excluding minio infrastructure tests)
- [ ] Pre-commit passes

### Additional Items Completed

- [x] `AirflowExecutor` class created for container execution with run log setup
- [x] `--init-run-log` flag added to `fan` CLI command (not just `execute_single_node`)
- [x] `entrypoints.fan()` updated to accept `init_run_log` parameter
- [x] Test context isolation bug fixed (`test_context_isolation_sync` now cleans up context)
