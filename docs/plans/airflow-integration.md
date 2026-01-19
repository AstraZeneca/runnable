# Airflow Integration Plan

## Overview

Create an Airflow DAG factory that converts Runnable pipelines to Airflow DAGs at import time, following the dag-factory pattern. This enables the same Runnable pipeline to run locally, in Argo, or in Airflow.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Runnable Pipeline (traversal.py)                           │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  AirflowDagFactory                                          │
│  - Loads pipeline definition                                │
│  - Traverses graph                                          │
│  - Builds Airflow DAG objects programmatically              │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  Airflow DAG (native Python objects)                        │
│  - DockerOperator tasks with get_node_callable_command()    │
│  - TaskGroups for parallel/conditional                      │
│  - Dynamic task mapping for map nodes                       │
└─────────────────────────────────────────────────────────────┘
```

## Usage Pattern

```python
# airflow/dags/my_pipelines.py
from runnable.airflow import AirflowDagFactory

factory = AirflowDagFactory(
    image="my-runnable-image:latest",
    # shared Docker config
)

# Each call returns an Airflow DAG object
traversal_dag = factory.create_dag(
    "examples/02-sequential/traversal.py",
    dag_id="traversal-pipeline",
)
```

## File Structure

```
extensions/pipeline_executor/
├── __init__.py          # existing
├── airflow.py           # NEW: AirflowDagFactory implementation
├── argo.py              # existing reference
├── local.py             # existing
└── local_container.py   # existing
```

## Implementation Details

### 1. AirflowDagFactory Class

```python
class AirflowDagFactory:
    """Factory for creating Airflow DAGs from Runnable pipelines."""

    def __init__(
        self,
        image: str,
        # Docker operator defaults
        docker_url: str = "unix://var/run/docker.sock",
        network_mode: str = "bridge",
        auto_remove: str = "success",
        mount_tmp_dir: bool = False,
        # Volume mounts for run logs and catalog
        volumes: list[str] = None,
        environment: dict[str, str] = None,
        # Airflow DAG defaults
        default_args: dict = None,
        schedule: str = None,
        catchup: bool = False,
        tags: list[str] = None,
    ):
        ...

    def create_dag(
        self,
        pipeline_file: str,
        dag_id: str,
        # Per-DAG overrides
        image: str = None,
        schedule: str = None,
        **dag_kwargs,
    ) -> DAG:
        """Create an Airflow DAG from a Runnable pipeline file."""
        ...
```

### 2. Node Type Mapping

| Runnable Node | Airflow Construct |
|---------------|-------------------|
| task (python/shell/notebook) | DockerOperator with `get_node_callable_command()` |
| stub | DockerOperator (minimal execution) |
| success | EmptyOperator with `trigger_rule=TriggerRule.ALL_SUCCESS` |
| fail | EmptyOperator with `trigger_rule=TriggerRule.ONE_FAILED` |
| parallel | TaskGroup with parallel operators |
| map | DockerOperator.partial().expand() (dynamic task mapping) |
| conditional | BranchPythonOperator + TaskGroup per branch |

### 3. Graph Traversal

The factory will traverse the Runnable graph similarly to ArgoExecutor:

```python
def _build_dag_tasks(
    self,
    dag: DAG,
    graph: Graph,
    parent_group: TaskGroup = None,
) -> tuple[BaseOperator, BaseOperator]:
    """
    Recursively build Airflow tasks from Runnable graph.
    Returns (first_task, last_task) for dependency chaining.
    """
```

### 4. Command Generation

Reuse the existing `get_node_callable_command()` from context:

```python
# For task nodes
command = f"runnable execute-single-node {run_id} {pipeline_file} {node_name} ..."

# For fan-out (map/parallel/conditional)
command = f"runnable fan {run_id} {node_name} {pipeline_file} out ..."

# For fan-in
command = f"runnable fan {run_id} {node_name} {pipeline_file} in ..."
```

### 5. Handling Composite Nodes

#### Map Node

The iter_variable must be passed to each branch task via the `--iter-variable` flag. With Airflow's dynamic task mapping:

```python
# Fan-out task - executes runnable fan command, returns iteration list via XCom
fan_out = DockerOperator(
    task_id=f"{node_name}_fan_out",
    command=get_fan_command(node, "out"),
    # Must configure to capture output for XCom
    do_xcom_push=True,
    ...
)

# Branch tasks with dynamic mapping
# The command uses Jinja templating to inject iter_variable per task instance
iterate_as = node.iterate_as  # e.g., "chunk"
branch = DockerOperator.partial(
    task_id=f"{node_name}_branch",
    image=self.image,
    # Jinja template accesses the iteration value via map_index
    command=f"""runnable execute-single-node {{{{ params.run_id }}}} {pipeline_file} {branch_node_name} \
        --iter-variable '{{"map_variable":{{"{iterate_as}":{{"value":"{{{{ ti.xcom_pull(task_ids='{node_name}_fan_out')[ti.map_index] }}}}"}}}}}}'""",
).expand(op_args=["{{ ti.xcom_pull(task_ids='" + f"{node_name}_fan_out" + "') }}"])

# Fan-in task
fan_in = DockerOperator(
    task_id=f"{node_name}_fan_in",
    command=get_fan_command(node, "in"),
    trigger_rule=TriggerRule.ALL_DONE,
    ...
)

fan_out >> branch >> fan_in
```

**Key insight**: The fan-out container must write the iteration list to stdout/XCom. The existing `runnable fan ... out` command writes to `/tmp/output.txt` for Argo - we'll need to also echo it for Airflow's XCom capture, or use a wrapper that reads the file and outputs it.

#### Parallel Node

```python
with TaskGroup(group_id=node_name) as parallel_group:
    fan_out = DockerOperator(...)

    branch_ends = []
    for branch_name, branch_graph in node.branches.items():
        with TaskGroup(group_id=branch_name):
            first, last = self._build_dag_tasks(dag, branch_graph)
            fan_out >> first
            branch_ends.append(last)

    fan_in = DockerOperator(trigger_rule=TriggerRule.ALL_DONE, ...)
    branch_ends >> fan_in
```

#### Conditional Node

```python
# Fan-out determines which branch to execute
fan_out = DockerOperator(...)  # outputs branch name to XCom

def branch_selector(**context):
    return context['ti'].xcom_pull(task_ids=f'{node_name}_fan_out')

branch_op = BranchPythonOperator(
    task_id=f"{node_name}_branch",
    python_callable=branch_selector,
)

for branch_name, branch_graph in node.branches.items():
    with TaskGroup(group_id=branch_name):
        first, last = self._build_dag_tasks(dag, branch_graph)
        branch_op >> first
        last >> fan_in

fan_in = DockerOperator(trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS, ...)
```

### 6. Configuration Example

```yaml
# airflow-config.yaml (for run-log-store, catalog, etc.)
run-log-store:
  type: file-system
  config:
    log_folder: /tmp/run_logs

catalog:
  type: file-system
  config:
    catalog_location: /tmp/catalog
```

## Implementation Steps

### Step 1: Create Base Factory Structure

- [ ] Create `extensions/pipeline_executor/airflow.py`
- [ ] Implement `AirflowDagFactory` class with configuration
- [ ] Add imports guard for optional Airflow dependency

### Step 2: Implement Simple Task Conversion

- [ ] Load Runnable pipeline from Python file
- [ ] Traverse linear graph (no composite nodes)
- [ ] Generate DockerOperator for each task node
- [ ] Wire up dependencies

### Step 3: Add Composite Node Support

- [ ] Implement parallel node handling with TaskGroup
- [ ] Implement map node with dynamic task mapping
- [ ] Implement conditional node with BranchPythonOperator

### Step 4: Add Fan-out/Fan-in Support

- [ ] Integrate `get_fan_command()` for composite nodes
- [ ] Handle XCom for passing iteration values
- [ ] Set proper trigger rules

### Step 5: Configuration and Overrides

- [ ] Support per-node Docker image overrides
- [ ] Support environment variables
- [ ] Support volume mounts for catalog/run-log

### Step 6: Testing

- [ ] Unit tests for graph traversal
- [ ] Integration test with simple pipeline
- [ ] Integration test with map/parallel/conditional

### Step 7: Documentation

- [ ] Add example in `examples/configs/`
- [ ] Document usage in docs/

## Key Files to Modify

1. **NEW**: `extensions/pipeline_executor/airflow.py` - Main implementation
2. **MODIFY**: `pyproject.toml` - Add optional `airflow` dependency group
3. **NEW**: `examples/configs/airflow-config.yaml` - Example config
4. **NEW**: `examples/airflow/` - Example Airflow DAG loader

## Dependencies

Add to `pyproject.toml`:

```toml
[project.optional-dependencies]
airflow = [
    "apache-airflow>=2.7.0",
    "apache-airflow-providers-docker>=3.0.0",
]
```

## Verification

1. Create a simple Runnable pipeline
2. Use factory to create Airflow DAG
3. Run with `airflow dags test`
4. Verify tasks execute via DockerOperator
5. Check run logs are created properly

## Design Decisions

1. **XCom for Map**: Use XCom for passing iteration values (JSON-friendly types only)
   - Fan-out outputs list to XCom
   - Dynamic tasks use Jinja templating with `ti.map_index` to get their value
   - Requires fan-out to echo iteration list to stdout (XCom capture)

2. **Run ID**: Use Airflow's `{{ run_id }}` template variable
   - Provides traceability between Airflow and Runnable logs
   - Passed as DAG param, templated into all commands

3. **Error Handling**: Use `trigger_rule=TriggerRule.ALL_DONE` for fan-in
   - Allows fan-in to run even if some branches fail
   - Fan-in updates step status based on branch outcomes

## Remaining Considerations

1. **XCom Output**: The existing `runnable fan ... out` writes to `/tmp/output.txt`. We need either:
   - A wrapper script that also echoes to stdout for XCom
   - Or modify the fan command to optionally output to stdout

2. **Nested Map Variables**: For maps inside maps, the iter_variable JSON grows. Verify Jinja templating handles nested escaping correctly.
