# Airflow DAG Factory Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create an AirflowDagFactory that converts Runnable pipelines to native Airflow DAGs at import time.

**Architecture:** The factory uses `get_pipeline_spec_from_python()` to load a Graph, traverses it, and builds Airflow DAG objects with DockerOperator tasks executing `runnable` CLI commands (`execute-single-node`, `fan`).

**Tech Stack:** Python, Pydantic, Apache Airflow 2.7+, Docker

---

## Key Reference Files

- **Pipeline loading:** `runnable/context.py:52-62` - `get_pipeline_spec_from_python()`
- **Command formats:** `runnable/context.py:331-416` - `get_node_callable_command()`, `get_fan_command()`
- **CLI definitions:** `runnable/cli.py:98-226` - `execute-single-node`, `fan` commands
- **Existing executor reference:** `extensions/pipeline_executor/argo.py` - ArgoExecutor pattern

## CLI Command Formats

**execute-single-node:**
```
runnable execute-single-node {run_id} {pipeline_file} {step_name} --mode python [--config {config}] [--iter-variable '{json}']
```

**fan:**
```
runnable fan {run_id} {step_name} {pipeline_file} {in|out} --mode python [--config-file {config}] [--iter-variable '{json}']
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

    Example:
        factory = AirflowDagFactory(image="my-image:latest")
        dag = factory.create_dag("pipeline.py", dag_id="my-dag")
    """

    # Docker configuration
    image: str
    docker_url: str = Field(default="unix://var/run/docker.sock")
    network_mode: str = Field(default="bridge")
    auto_remove: str = Field(default="success")
    mount_tmp_dir: bool = Field(default=False)
    volumes: list[str] = Field(default_factory=list)
    environment: dict[str, str] = Field(default_factory=dict)

    # Airflow DAG defaults
    default_args: dict[str, Any] = Field(default_factory=dict)
    schedule: Optional[str] = Field(default=None)
    catchup: bool = Field(default=False)
    tags: list[str] = Field(default_factory=list)

    # Runnable configuration
    config_file: Optional[str] = Field(default=None)

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, **data):
        _check_airflow_available()
        super().__init__(**data)
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

        # Resolve configuration
        effective_image = image or self.image
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
                image=effective_image,
            )

        return dag

    def _build_dag_from_graph(
        self,
        dag: "DAG",
        graph: Any,
        pipeline_file: str,
        image: str,
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
        image: str,
        iter_variable: Optional[dict] = None,
    ) -> tuple[Any, Any]:
        """
        Build Airflow tasks from Runnable graph.

        Returns:
            Tuple of (first_task, last_task) for dependency chaining
        """
        current_node_name = graph.start_at
        first_task = None
        previous_task = None

        while current_node_name:
            node = graph.get_node_by_name(current_node_name)
            task_id = self._sanitize_task_id(node.internal_name)

            match node.node_type:
                case "task" | "stub":
                    task = self._create_docker_task(
                        node=node,
                        task_id=task_id,
                        pipeline_file=pipeline_file,
                        image=image,
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
        image: str,
        iter_variable: Optional[dict] = None,
    ) -> "DockerOperator":
        """Create DockerOperator for a task node."""
        # Build command: runnable execute-single-node {run_id} {file} {step} --mode python
        command = self._build_execute_command(
            node=node,
            pipeline_file=pipeline_file,
            iter_variable=iter_variable,
        )

        return DockerOperator(
            task_id=task_id,
            image=image,
            command=command,
            docker_url=self.docker_url,
            network_mode=self.network_mode,
            auto_remove=self.auto_remove,
            mount_tmp_dir=self.mount_tmp_dir,
            environment=self.environment,
        )

    def _build_execute_command(
        self,
        node: Any,
        pipeline_file: str,
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

        if self.config_file:
            cmd += f" --config {self.config_file}"

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
        iter_variable: Optional[dict] = None,
        trigger_rule: Optional["TriggerRule"] = None,
    ) -> "DockerOperator":
        """Create fan-out or fan-in DockerOperator."""
        command = self._build_fan_command(
            node=node,
            mode=mode,
            pipeline_file=pipeline_file,
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

        if self.config_file:
            cmd += f" --config-file {self.config_file}"

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
- Create: `examples/configs/airflow-config.yaml`

**Step 1: Create directory and example DAG loader**

Run: `mkdir -p examples/airflow`

Create `examples/airflow/dag_loader.py`:

```python
"""
Example Airflow DAG loader for Runnable pipelines.

Place in your Airflow DAGs folder and configure the factory.
"""

from extensions.pipeline_executor.airflow import AirflowDagFactory

factory = AirflowDagFactory(
    image="your-runnable-image:latest",
    volumes=[
        "/tmp/run_logs:/tmp/run_logs",
        "/tmp/catalog:/tmp/catalog",
    ],
    config_file="examples/configs/airflow-config.yaml",
    default_args={"owner": "runnable", "retries": 1},
    catchup=False,
    tags=["runnable"],
)

# Create DAG from pipeline
traversal_dag = factory.create_dag(
    pipeline_file="examples/02-sequential/traversal.py",
    dag_id="runnable-traversal",
    schedule="@daily",
)
```

**Step 2: Create config file**

Create `examples/configs/airflow-config.yaml`:

```yaml
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
```

**Step 3: Verify files exist**

Run: `ls -la examples/airflow/dag_loader.py examples/configs/airflow-config.yaml`

**Step 4: Commit**

```bash
git add examples/airflow/ examples/configs/airflow-config.yaml
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

## Verification Checklist

- [ ] `get_pipeline_spec_from_python` used for loading
- [ ] All commands include `--mode python`
- [ ] All commands use `node._command_friendly_name()`
- [ ] Linear graph traversal works
- [ ] Parallel nodes create TaskGroup with fan-out/fan-in
- [ ] Map nodes support dynamic task mapping
- [ ] Conditional nodes use BranchPythonOperator
- [ ] `pyproject.toml` has airflow dependency
- [ ] Example files created
- [ ] All tests pass
- [ ] Pre-commit passes
