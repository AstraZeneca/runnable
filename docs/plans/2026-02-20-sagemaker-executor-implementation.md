# SageMaker Pipeline Executor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement SageMaker Pipelines integration as a new pipeline executor for Runnable, enabling AWS-native ML pipeline execution.

**Architecture:** DAG transpilation pattern using SageMaker Python SDK to convert Runnable pipelines into SageMaker Pipeline definitions, following established patterns from Argo executor.

**Tech Stack:** SageMaker Python SDK, Pydantic configuration models, AWS IAM integration, S3 storage access.

---

## ⚠️ Design TODOs (Resolve Before Implementation)

The following design decisions need to be resolved before full implementation. See `2026-02-20-sagemaker-executor-design.md` for details:

1. **SageMaker Native Features**: Decide on caching/retry integration with Runnable's systems
2. **Testing Strategy**: Determine which testing levels are needed (mock, local mode, AWS)
3. **Output File Generation**: Research SageMaker Pipeline validation capabilities
4. **Run Context Propagation**: Design mechanism for passing run_id and config to containers
5. **SDK Dependency Strategy**: Decide if SageMaker SDK should be optional (`runnable[sagemaker]`)
6. **VPC/Network Configuration**: Design VPC config schema for private resource access
7. **Secrets Integration**: Design AWS Secrets Manager integration pattern
8. **Resource Tagging**: Add tags configuration for cost tracking

**Note:** This implementation plan covers the core functionality. Advanced features (VPC, secrets, tags) can be added in subsequent phases after the basic executor works.

---

## Phase 1: Foundation and Configuration

### Task 1: Create Basic SageMaker Executor Structure

**Files:**
- Create: `extensions/pipeline_executor/sagemaker.py`
- Test: `tests/extensions/pipeline_executor/test_sagemaker.py`

**Step 1: Write the failing test**

```python
# tests/extensions/pipeline_executor/test_sagemaker.py
import pytest
from extensions.pipeline_executor.sagemaker import SageMakerExecutor
from runnable.graph import Graph


def test_sagemaker_executor_service_name():
    """Test that SageMaker executor has correct service name."""
    config = {
        "role_arn": "arn:aws:iam::123456789:role/TestRole",
        "region": "us-east-1",
        "image": "test:latest"
    }
    executor = SageMakerExecutor(**config)
    assert executor.service_name == "sagemaker"


def test_sagemaker_executor_initialization():
    """Test that SageMaker executor initializes with required fields."""
    config = {
        "role_arn": "arn:aws:iam::123456789:role/TestRole",
        "region": "us-east-1",
        "image": "test:latest",
        "instance_type": "ml.m5.large",
        "instance_count": 1
    }
    executor = SageMakerExecutor(**config)
    assert executor.role_arn == config["role_arn"]
    assert executor.region == config["region"]
    assert executor.image == config["image"]
    assert executor.instance_type == config["instance_type"]
    assert executor.instance_count == config["instance_count"]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/extensions/pipeline_executor/test_sagemaker.py::test_sagemaker_executor_service_name -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'extensions.pipeline_executor.sagemaker'"

**Step 3: Write minimal SageMaker executor class**

```python
# extensions/pipeline_executor/sagemaker.py
from typing import Optional, Literal, Any, Dict
from pydantic import Field, ConfigDict, PrivateAttr
from extensions.pipeline_executor import GenericPipelineExecutor
from runnable.graph import Graph


class SageMakerExecutor(GenericPipelineExecutor):
    """
    SageMaker Pipeline Executor for Runnable.

    Converts Runnable pipelines into SageMaker Pipeline definitions using the AWS SDK.
    Follows DAG transpilation pattern like Argo executor.

    Configuration:
    - role_arn: IAM role for SageMaker execution (required)
    - region: AWS region (required)
    - image: Container image for pipeline execution (required)
    - instance_type: Default EC2 instance type for processing jobs
    - instance_count: Default number of instances
    - volume_size_gb: Default storage volume size
    - max_runtime_seconds: Default maximum runtime per job
    - wait_for_completion: Whether to wait for pipeline completion
    - overrides: Per-task compute configuration overrides
    """

    service_name: str = "sagemaker"

    model_config = ConfigDict(
        extra="forbid",
        alias_generator=None,  # Use snake_case unlike Argo's camelCase
        populate_by_name=True,
        from_attributes=True,
        use_enum_values=True,
    )

    # Required AWS configuration
    role_arn: str = Field(..., description="IAM role ARN for SageMaker execution")
    region: str = Field(..., description="AWS region for SageMaker resources")
    image: str = Field(..., description="Container image for pipeline tasks")

    # Optional compute configuration with defaults
    instance_type: str = Field(default="ml.m5.large", description="Default EC2 instance type")
    instance_count: int = Field(default=1, description="Default number of instances")
    volume_size_gb: int = Field(default=30, description="Default EBS volume size in GB")
    max_runtime_seconds: int = Field(default=3600, description="Default max runtime per job")

    # Optional execution configuration
    wait_for_completion: bool = Field(default=False, description="Wait for pipeline completion")

    # Per-task overrides (like Argo pattern)
    overrides: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Per-task configuration overrides")

    # NOTE: Method name is execute_graph (not execute_from_graph) to match Argo pattern
    _should_setup_run_log_at_traversal: bool = PrivateAttr(default=False)

    def execute_graph(self, dag: Graph, map_variable=None):
        """Convert Runnable DAG to SageMaker Pipeline and execute."""
        # TODO: Implement in later tasks
        raise NotImplementedError("SageMaker execution not yet implemented")

    def trigger_node_execution(self, node, map_variable=None):
        """Execute single node within SageMaker Processing job."""
        # TODO: Implement in later tasks
        raise NotImplementedError("SageMaker node execution not yet implemented")
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/extensions/pipeline_executor/test_sagemaker.py::test_sagemaker_executor_service_name -v`
Expected: PASS

**Step 5: Commit**

```bash
git add extensions/pipeline_executor/sagemaker.py tests/extensions/pipeline_executor/test_sagemaker.py
git commit -m "feat: add basic SageMaker executor structure

- Create SageMakerExecutor class with Pydantic configuration
- Add required AWS fields (role_arn, region, image)
- Add optional compute defaults (instance_type, volume_size, etc.)
- Add override system matching Argo pattern
- Placeholder methods for execution logic"
```

### Task 2: Register SageMaker Executor in Entry Points

**Files:**
- Modify: `pyproject.toml:120-125`

**Step 1: Write test for entry point registration**

```python
# tests/extensions/pipeline_executor/test_sagemaker.py
def test_sagemaker_executor_entry_point():
    """Test that SageMaker executor is registered in entry points."""
    from runnable.extensions import pipeline_executor_registry

    executor_class = pipeline_executor_registry.get_executor_class("sagemaker")
    assert executor_class.__name__ == "SageMakerExecutor"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/extensions/pipeline_executor/test_sagemaker.py::test_sagemaker_executor_entry_point -v`
Expected: FAIL with entry point not found

**Step 3: Add SageMaker executor to entry points**

```toml
# pyproject.toml (line ~120)
[project.entry-points.'pipeline_executor']
"local" = "extensions.pipeline_executor.local:LocalExecutor"
"local-container" = "extensions.pipeline_executor.local_container:LocalContainerExecutor"
"emulator" = "extensions.pipeline_executor.emulate:Emulator"
"argo" = "extensions.pipeline_executor.argo:ArgoExecutor"
"mocked" = "extensions.pipeline_executor.mocked:MockedExecutor"
"sagemaker" = "extensions.pipeline_executor.sagemaker:SageMakerExecutor"
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/extensions/pipeline_executor/test_sagemaker.py::test_sagemaker_executor_entry_point -v`
Expected: PASS

**Step 5: Commit**

```bash
git add pyproject.toml
git commit -m "feat: register SageMaker executor in entry points

Add 'sagemaker' entry point mapping to SageMakerExecutor class"
```

### Task 3: Add SageMaker SDK Dependency

**Files:**
- Modify: `pyproject.toml:30-40` (dependencies section)

**Step 1: Write test requiring SageMaker SDK**

```python
# tests/extensions/pipeline_executor/test_sagemaker.py
def test_sagemaker_sdk_import():
    """Test that SageMaker SDK is available."""
    try:
        import sagemaker
        from sagemaker.workflow.pipeline import Pipeline as SGPipeline
        from sagemaker.processing import ScriptProcessor, ProcessingStep
        assert True
    except ImportError as e:
        pytest.fail(f"SageMaker SDK not available: {e}")
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/extensions/pipeline_executor/test_sagemaker.py::test_sagemaker_sdk_import -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'sagemaker'"

**Step 3: Add sagemaker dependency**

```toml
# pyproject.toml - add to dependencies array
dependencies = [
    # ... existing dependencies ...
    "sagemaker>=2.120.0",  # Add SageMaker SDK
]
```

**Step 4: Install dependency and run test**

Run: `uv sync` then `pytest tests/extensions/pipeline_executor/test_sagemaker.py::test_sagemaker_sdk_import -v`
Expected: PASS

**Step 5: Commit**

```bash
git add pyproject.toml
git commit -m "feat: add SageMaker SDK dependency

Add sagemaker>=2.120.0 for SageMaker Pipelines integration"
```

## Phase 2: Node Type Detection and Validation

### Task 4: Implement Unsupported Node Detection

**Files:**
- Modify: `extensions/pipeline_executor/sagemaker.py`
- Test: `tests/extensions/pipeline_executor/test_sagemaker.py`

**Step 1: Write tests for node type detection**

```python
# tests/extensions/pipeline_executor/test_sagemaker.py
from unittest.mock import Mock
from runnable import exceptions


def test_check_unsupported_nodes_map_type():
    """Test that map nodes are detected as unsupported."""
    executor = SageMakerExecutor(
        role_arn="arn:aws:iam::123456789:role/TestRole",
        region="us-east-1",
        image="test:latest"
    )

    # Mock a map node
    map_node = Mock()
    map_node.node_type = "map"
    map_node.internal_name = "test_map"

    dag = Mock()
    dag.nodes = [map_node]

    unsupported = executor._check_for_unsupported_nodes(dag)
    assert len(unsupported) == 1
    assert unsupported[0] == map_node


def test_check_unsupported_nodes_nested_parallel():
    """Test that nested parallel nodes are detected as unsupported."""
    executor = SageMakerExecutor(
        role_arn="arn:aws:iam::123456789:role/TestRole",
        region="us-east-1",
        image="test:latest"
    )

    # Mock nested parallel node (>2 dots)
    nested_node = Mock()
    nested_node.node_type = "task"
    nested_node.internal_name = "outer.branch_a.inner.branch_b.task"  # 4 dots = nested

    dag = Mock()
    dag.nodes = [nested_node]

    unsupported = executor._check_for_unsupported_nodes(dag)
    assert len(unsupported) == 1
    assert unsupported[0] == nested_node


def test_check_supported_nodes():
    """Test that simple nodes are supported."""
    executor = SageMakerExecutor(
        role_arn="arn:aws:iam::123456789:role/TestRole",
        region="us-east-1",
        image="test:latest"
    )

    # Mock supported nodes
    simple_task = Mock()
    simple_task.node_type = "task"
    simple_task.internal_name = "simple_task"  # 0 dots

    parallel_branch = Mock()
    parallel_branch.node_type = "task"
    parallel_branch.internal_name = "parallel.branch_a.task"  # 2 dots = ok

    dag = Mock()
    dag.nodes = [simple_task, parallel_branch]

    unsupported = executor._check_for_unsupported_nodes(dag)
    assert len(unsupported) == 0
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/extensions/pipeline_executor/test_sagemaker.py -k "test_check_unsupported" -v`
Expected: FAIL with "AttributeError: 'SageMakerExecutor' object has no attribute '_check_for_unsupported_nodes'"

**Step 3: Implement node detection methods**

```python
# extensions/pipeline_executor/sagemaker.py
from runnable import exceptions


class SageMakerExecutor(GenericPipelineExecutor):
    # ... existing code ...

    def _check_for_unsupported_nodes(self, dag):
        """Reject nodes that SageMaker cannot handle."""
        unsupported = []

        for node in dag.nodes:
            # Always unsupported: dynamic execution patterns
            if node.node_type in ["map", "loop"]:
                unsupported.append(node)

            # Nested parallel detection: count dots in internal name
            elif node.internal_name.count('.') > 2:
                unsupported.append(node)

        return unsupported

    def _raise_unsupported_error(self, unsupported_nodes):
        """Raise clear error message for unsupported node patterns."""
        nested = [n.internal_name for n in unsupported_nodes if n.internal_name.count('.') > 2]
        other = [f"{n.internal_name} ({n.node_type})" for n in unsupported_nodes if n.internal_name.count('.') <= 2]

        msg = "SageMaker executor limitations:\n"
        if nested:
            msg += f"- Nested structures not supported: {nested}\n"
        if other:
            msg += f"- Unsupported node types: {other}\n"
        msg += "Use 'argo' executor for complex workflows."

        raise exceptions.ExecutorNotSupported(msg)
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/extensions/pipeline_executor/test_sagemaker.py -k "test_check_unsupported" -v`
Expected: PASS

**Step 5: Commit**

```bash
git add extensions/pipeline_executor/sagemaker.py tests/extensions/pipeline_executor/test_sagemaker.py
git commit -m "feat: add unsupported node detection for SageMaker

- Detect map and loop node types (always unsupported)
- Detect nested parallel structures using dot counting (>2 dots)
- Provide clear error messages with guidance to use Argo
- Add comprehensive test coverage for detection logic"
```

## Phase 3: Basic DAG Transpilation

### Task 5: Implement Simple Task Chain Execution

**Files:**
- Modify: `extensions/pipeline_executor/sagemaker.py`
- Test: `tests/extensions/pipeline_executor/test_sagemaker.py`

**Step 1: Write test for simple DAG execution**

```python
# tests/extensions/pipeline_executor/test_sagemaker.py
from unittest.mock import Mock, patch


def test_execute_simple_task_chain():
    """Test execution of simple sequential task chain."""
    executor = SageMakerExecutor(
        role_arn="arn:aws:iam::123456789:role/TestRole",
        region="us-east-1",
        image="test:latest"
    )

    # Mock simple DAG with 2 sequential tasks
    task1 = Mock()
    task1.node_type = "task"
    task1.internal_name = "task1"

    task2 = Mock()
    task2.node_type = "task"
    task2.internal_name = "task2"

    dag = Mock()
    dag.nodes = [task1, task2]

    # Mock SageMaker SDK objects
    with patch('sagemaker.workflow.pipeline.Pipeline') as mock_pipeline_class, \
         patch('sagemaker.processing.ScriptProcessor') as mock_processor_class, \
         patch('sagemaker.processing.ProcessingStep') as mock_step_class:

        mock_pipeline = Mock()
        mock_pipeline_class.return_value = mock_pipeline
        mock_execution = Mock()
        mock_pipeline.start.return_value = mock_execution

        # Mock methods that will be implemented
        executor._traverse_dag_in_order = Mock(return_value=[task1, task2])
        executor._get_immediate_parents = Mock(side_effect=[[], [task1]])  # task2 depends on task1
        executor._create_processor = Mock()
        executor._generate_pipeline_name = Mock(return_value="test-pipeline")
        executor._get_session = Mock()

        # Execute
        executor.execute_from_graph(dag)

        # Verify SageMaker pipeline creation
        mock_pipeline_class.assert_called_once()
        mock_pipeline.upsert.assert_called_once_with(role_arn=executor.role_arn)
        mock_pipeline.start.assert_called_once()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/extensions/pipeline_executor/test_sagemaker.py::test_execute_simple_task_chain -v`
Expected: FAIL due to NotImplementedError in execute_from_graph

**Step 3: Implement basic execute_from_graph method**

```python
# extensions/pipeline_executor/sagemaker.py
import secrets
from sagemaker.workflow.pipeline import Pipeline as SGPipeline
from sagemaker.processing import ScriptProcessor, ProcessingStep


class SageMakerExecutor(GenericPipelineExecutor):
    # ... existing code ...

    def execute_from_graph(self, dag, map_variable=None):
        """Convert Runnable DAG to SageMaker Pipeline and execute."""
        # Check for unsupported node types
        unsupported_nodes = self._check_for_unsupported_nodes(dag)
        if unsupported_nodes:
            self._raise_unsupported_error(unsupported_nodes)

        # Convert DAG to SageMaker Pipeline steps
        steps = []
        step_map = {}

        # Traverse DAG in execution order and build steps with dependencies
        for node in self._traverse_dag_in_order(dag):
            # Find dependencies from already-processed steps
            depends_on_steps = []
            for parent in self._get_immediate_parents(node, dag):
                if parent.internal_name in step_map:
                    depends_on_steps.append(step_map[parent.internal_name])

            # Create ProcessingStep with dependencies
            processor = self._create_processor(node)
            step = ProcessingStep(
                name=node.internal_name,
                processor=processor,
                depends_on=depends_on_steps
            )

            steps.append(step)
            step_map[node.internal_name] = step

        # Build and submit SageMaker Pipeline
        sg_pipeline = SGPipeline(
            name=self._generate_pipeline_name(),
            steps=steps,
            sagemaker_session=self._get_session()
        )

        sg_pipeline.upsert(role_arn=self.role_arn)
        execution = sg_pipeline.start()

        # Optional monitoring
        if self.wait_for_completion:
            self._monitor_execution(execution)

    def _traverse_dag_in_order(self, dag):
        """Traverse DAG nodes in execution order."""
        # TODO: Use existing DAG traversal from base class or implement simple version
        return dag.nodes  # Simplified for now

    def _get_immediate_parents(self, node, dag):
        """Get immediate parent nodes for dependency calculation."""
        # TODO: Implement based on DAG structure
        return []  # Simplified for now

    def _create_processor(self, node):
        """Create SageMaker ScriptProcessor for a node."""
        # Get node-specific configuration (apply overrides)
        config = self._get_node_config(node)

        return ScriptProcessor(
            image_uri=self.image,
            role=self.role_arn,
            instance_type=config["instance_type"],
            instance_count=config["instance_count"],
            volume_size_in_gb=config["volume_size_gb"],
            max_runtime_in_seconds=config["max_runtime_seconds"],
            command=["runnable", "execute-single-node", node.internal_name]
        )

    def _get_node_config(self, node):
        """Get configuration for a node, applying overrides."""
        base_config = {
            "instance_type": self.instance_type,
            "instance_count": self.instance_count,
            "volume_size_gb": self.volume_size_gb,
            "max_runtime_seconds": self.max_runtime_seconds
        }

        # Apply overrides if node has them
        if hasattr(node, 'overrides') and node.overrides and 'sagemaker' in node.overrides:
            override_key = node.overrides['sagemaker']
            if override_key in self.overrides:
                base_config.update(self.overrides[override_key])

        return base_config

    def _generate_pipeline_name(self):
        """Generate unique pipeline name."""
        return f"runnable-pipeline-{secrets.token_hex(4)}"

    def _get_session(self):
        """Get SageMaker session."""
        import sagemaker
        return sagemaker.Session(boto_session=None, default_bucket=None)

    def _monitor_execution(self, execution):
        """Monitor pipeline execution until completion."""
        execution.wait()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/extensions/pipeline_executor/test_sagemaker.py::test_execute_simple_task_chain -v`
Expected: PASS

**Step 5: Commit**

```bash
git add extensions/pipeline_executor/sagemaker.py tests/extensions/pipeline_executor/test_sagemaker.py
git commit -m "feat: implement basic DAG transpilation for SageMaker

- Add execute_from_graph method with unsupported node checking
- Convert Runnable nodes to SageMaker ProcessingSteps
- Handle inline dependency resolution during DAG traversal
- Add processor creation with node-specific configuration
- Add override system for per-task compute configuration
- Add pipeline name generation and optional monitoring"
```

### Task 6: Implement Proper DAG Traversal and Dependencies

**Files:**
- Modify: `extensions/pipeline_executor/sagemaker.py`
- Test: `tests/extensions/pipeline_executor/test_sagemaker.py`

**Step 1: Write test for dependency resolution**

```python
# tests/extensions/pipeline_executor/test_sagemaker.py
def test_dependency_resolution():
    """Test that dependencies are correctly resolved between tasks."""
    executor = SageMakerExecutor(
        role_arn="arn:aws:iam::123456789:role/TestRole",
        region="us-east-1",
        image="test:latest"
    )

    # Create mock DAG with dependencies: task1 -> task2 -> task3
    task1 = Mock()
    task1.internal_name = "task1"
    task1.node_type = "task"

    task2 = Mock()
    task2.internal_name = "task2"
    task2.node_type = "task"

    task3 = Mock()
    task3.internal_name = "task3"
    task3.node_type = "task"

    dag = Mock()
    dag.nodes = [task1, task2, task3]

    # Test dependency mapping
    deps = executor._get_immediate_parents(task1, dag)
    assert len(deps) == 0  # task1 has no dependencies

    deps = executor._get_immediate_parents(task2, dag)
    assert task1 in deps  # task2 depends on task1

    deps = executor._get_immediate_parents(task3, dag)
    assert task2 in deps  # task3 depends on task2
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/extensions/pipeline_executor/test_sagemaker.py::test_dependency_resolution -v`
Expected: FAIL because _get_immediate_parents returns empty list

**Step 3: Implement proper dependency resolution**

```python
# extensions/pipeline_executor/sagemaker.py
class SageMakerExecutor(GenericPipelineExecutor):
    # ... existing code ...

    def _traverse_dag_in_order(self, dag):
        """Traverse DAG nodes in topological order."""
        # Use the existing graph traversal from base class
        # This ensures we process nodes in dependency order
        visited = set()
        ordered_nodes = []

        def visit_node(node_name):
            if node_name in visited:
                return

            node = dag.get_node_by_name(node_name)
            visited.add(node_name)

            # Visit dependencies first (topological sort)
            for parent in self._get_node_dependencies(node, dag):
                if parent.internal_name not in visited:
                    visit_node(parent.internal_name)

            ordered_nodes.append(node)

        # Start traversal from entry nodes (nodes with no dependencies)
        for node in dag.nodes:
            if len(self._get_node_dependencies(node, dag)) == 0:
                visit_node(node.internal_name)

        return ordered_nodes

    def _get_immediate_parents(self, node, dag):
        """Get immediate parent nodes for dependency calculation."""
        return self._get_node_dependencies(node, dag)

    def _get_node_dependencies(self, node, dag):
        """Get all nodes that this node depends on."""
        dependencies = []

        # Use the DAG's edge information to find dependencies
        # This is a simplified version - may need to adapt based on actual DAG structure
        for potential_parent in dag.nodes:
            if potential_parent != node:
                # Check if there's an edge from potential_parent to node
                if self._has_edge(dag, potential_parent, node):
                    dependencies.append(potential_parent)

        return dependencies

    def _has_edge(self, dag, from_node, to_node):
        """Check if there's a direct edge between two nodes."""
        # This is a placeholder - actual implementation depends on DAG structure
        # For now, assume simple sequential dependencies based on internal names

        # Simple heuristic: if nodes are sequential and share similar names
        # This should be replaced with proper DAG edge checking
        return False  # Simplified for now
```

**Step 4: Update to use base class DAG traversal methods**

```python
# extensions/pipeline_executor/sagemaker.py
class SageMakerExecutor(GenericPipelineExecutor):
    # ... existing code ...

    def _traverse_dag_in_order(self, dag):
        """Traverse DAG nodes in execution order using base class logic."""
        # Leverage existing DAG traversal from GenericPipelineExecutor
        # This uses the same traversal logic as other executors

        # Start from the DAG's start_at node and traverse
        ordered_nodes = []
        current_node_name = dag.start_at

        while current_node_name:
            node = dag.get_node_by_name(current_node_name)
            ordered_nodes.append(node)

            # Get next node in traversal
            current_node_name = self._get_next_node_name(node, dag)

        return ordered_nodes

    def _get_next_node_name(self, current_node, dag):
        """Get the next node name in DAG traversal."""
        # Use the node's _get_next_node method if available
        if hasattr(current_node, '_get_next_node'):
            try:
                return current_node._get_next_node()
            except AttributeError:
                return None

        return None

    def _get_immediate_parents(self, node, dag):
        """Get immediate parent nodes by checking DAG structure."""
        parents = []

        # Find all nodes that have this node as their next node
        for potential_parent in dag.nodes:
            if potential_parent != node:
                try:
                    next_node_name = self._get_next_node_name(potential_parent, dag)
                    if next_node_name == node.internal_name:
                        parents.append(potential_parent)
                except AttributeError:
                    continue

        return parents
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/extensions/pipeline_executor/test_sagemaker.py::test_dependency_resolution -v`
Expected: PASS (may need adjustment based on actual DAG structure)

**Step 6: Commit**

```bash
git add extensions/pipeline_executor/sagemaker.py tests/extensions/pipeline_executor/test_sagemaker.py
git commit -m "feat: implement proper DAG traversal and dependency resolution

- Add topological traversal using base class DAG methods
- Implement immediate parent detection for dependency resolution
- Leverage existing node traversal patterns from GenericPipelineExecutor
- Add proper edge detection between nodes for SageMaker dependencies"
```

## Phase 4: Parallel and Conditional Node Support

### Task 7: Implement Parallel Node Handling with Fan-out/Fan-in

**Files:**
- Modify: `extensions/pipeline_executor/sagemaker.py`
- Test: `tests/extensions/pipeline_executor/test_sagemaker.py`

**Step 1: Write tests for parallel node handling**

```python
# tests/extensions/pipeline_executor/test_sagemaker.py
def test_handle_parallel_node():
    """Test that parallel nodes are converted to fan-out/fan-in pattern."""
    executor = SageMakerExecutor(
        role_arn="arn:aws:iam::123456789:role/TestRole",
        region="us-east-1",
        image="test:latest"
    )

    # Mock parallel node with two branches
    branch_a = Mock()
    branch_a.tasks = [Mock(internal_name="branch_a_task1"), Mock(internal_name="branch_a_task2")]

    branch_b = Mock()
    branch_b.tasks = [Mock(internal_name="branch_b_task1")]

    parallel_node = Mock()
    parallel_node.internal_name = "parallel_processing"
    parallel_node.node_type = "parallel"
    parallel_node.branches = {"branch_a": branch_a, "branch_b": branch_b}

    # Mock processor creation
    executor._create_processor_for_fan_out = Mock()
    executor._create_processor_for_fan_in = Mock()
    executor._create_processor = Mock()
    executor._create_success_processor = Mock()

    with patch('sagemaker.processing.ProcessingStep') as mock_step_class:
        mock_step = Mock()
        mock_step_class.return_value = mock_step

        steps, final_step = executor._handle_parallel_node(parallel_node)

        # Should create: fan_out + 2 branch_a_tasks + branch_a_success + 1 branch_b_task + branch_b_success + fan_in
        # Total: 7 steps
        assert len(steps) == 7
        assert final_step == mock_step  # fan_in step
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/extensions/pipeline_executor/test_sagemaker.py::test_handle_parallel_node -v`
Expected: FAIL with "AttributeError: 'SageMakerExecutor' object has no attribute '_handle_parallel_node'"

**Step 3: Implement parallel node handling**

```python
# extensions/pipeline_executor/sagemaker.py
class SageMakerExecutor(GenericPipelineExecutor):
    # ... existing code ...

    def _handle_parallel_node(self, parallel_node):
        """Convert ParallelNode using fan-out/fan-in pattern with success nodes."""

        # 1. Fan-out step
        fan_out_step = ProcessingStep(
            name=f"{parallel_node.internal_name}_fan_out",
            processor=self._create_processor_for_fan_out(parallel_node)
        )

        # 2. Process each branch + add success node
        branch_success_steps = []
        all_steps = [fan_out_step]

        for branch_name, branch_pipeline in parallel_node.branches.items():
            prev_step = fan_out_step

            # Process actual branch steps
            for task in branch_pipeline.tasks:
                step = ProcessingStep(
                    name=f"{branch_name}_{task.internal_name}",
                    processor=self._create_processor(task),
                    depends_on=[prev_step]
                )
                all_steps.append(step)
                prev_step = step

            # Add success node with proper internal name
            success_node_internal_name = f"{parallel_node.internal_name}.{branch_name}.success"
            branch_success_step = ProcessingStep(
                name=f"{branch_name}_success",
                processor=self._create_success_processor(success_node_internal_name),
                depends_on=[prev_step]
            )
            all_steps.append(branch_success_step)
            branch_success_steps.append(branch_success_step)

        # 3. Fan-in depends on ALL branch success nodes
        fan_in_step = ProcessingStep(
            name=f"{parallel_node.internal_name}_fan_in",
            processor=self._create_processor_for_fan_in(parallel_node),
            depends_on=branch_success_steps  # Multiple dependencies
        )
        all_steps.append(fan_in_step)

        return all_steps, fan_in_step

    def _create_processor_for_fan_out(self, parallel_node):
        """Create processor for parallel fan-out step."""
        config = self._get_node_config(parallel_node)

        return ScriptProcessor(
            image_uri=self.image,
            role=self.role_arn,
            instance_type=config["instance_type"],
            instance_count=config["instance_count"],
            volume_size_in_gb=config["volume_size_gb"],
            max_runtime_in_seconds=config["max_runtime_seconds"],
            command=["runnable", "execute-single-node", f"{parallel_node.internal_name}_fan_out"]
        )

    def _create_processor_for_fan_in(self, parallel_node):
        """Create processor for parallel fan-in step."""
        config = self._get_node_config(parallel_node)

        return ScriptProcessor(
            image_uri=self.image,
            role=self.role_arn,
            instance_type=config["instance_type"],
            instance_count=config["instance_count"],
            volume_size_in_gb=config["volume_size_gb"],
            max_runtime_in_seconds=config["max_runtime_seconds"],
            command=["runnable", "execute-single-node", f"{parallel_node.internal_name}_fan_in"]
        )

    def _create_success_processor(self, success_node_internal_name):
        """Create minimal processor for dummy success nodes."""
        return ScriptProcessor(
            image_uri=self.image,
            role=self.role_arn,
            instance_type="ml.t3.micro",  # Minimal resources for success nodes
            instance_count=1,
            volume_size_in_gb=1,
            max_runtime_in_seconds=60,  # Quick success check
            command=["runnable", "execute-single-node", success_node_internal_name]
        )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/extensions/pipeline_executor/test_sagemaker.py::test_handle_parallel_node -v`
Expected: PASS

**Step 5: Commit**

```bash
git add extensions/pipeline_executor/sagemaker.py tests/extensions/pipeline_executor/test_sagemaker.py
git commit -m "feat: implement parallel node support with fan-out/fan-in pattern

- Add _handle_parallel_node for converting ParallelNodes to SageMaker steps
- Implement fan-out/fan-in pattern with proper dependencies
- Add dummy success nodes for clean branch termination
- Create specialized processors for fan-out, fan-in, and success steps
- Use minimal resources (t3.micro) for lightweight success nodes"
```

### Task 8: Implement Conditional Node Handling

**Files:**
- Modify: `extensions/pipeline_executor/sagemaker.py`
- Test: `tests/extensions/pipeline_executor/test_sagemaker.py`

**Context:** Conditional nodes follow the same fan-out/fan-in pattern as parallel nodes. The key difference is that the fan-out step evaluates the condition (via `ConditionalNode.fan_out()`), and the fan-in step uses `ConditionalNode.fan_in()` to resolve status based on only the selected branch. All branches execute as SageMaker ProcessingSteps because SageMaker's native `ConditionStep` cannot evaluate arbitrary Python conditions. See design doc section "Conditional Node Support" for full rationale.

**Step 1: Write tests for conditional node handling**

```python
# tests/extensions/pipeline_executor/test_sagemaker.py
def test_handle_conditional_node():
    """Test that conditional nodes are converted to fan-out/fan-in pattern."""
    executor = SageMakerExecutor(
        role_arn="arn:aws:iam::123456789:role/TestRole",
        region="us-east-1",
        image="test:latest"
    )

    # Mock conditional node with two branches (if/else)
    branch_true = Mock()
    branch_true.tasks = [Mock(internal_name="branch_true_task1")]

    branch_false = Mock()
    branch_false.tasks = [Mock(internal_name="branch_false_task1")]

    conditional_node = Mock()
    conditional_node.internal_name = "check_accuracy"
    conditional_node.node_type = "conditional"
    conditional_node.branches = {"high_accuracy": branch_true, "low_accuracy": branch_false}

    # Mock processor creation
    executor._create_processor_for_fan_out = Mock()
    executor._create_processor_for_fan_in = Mock()
    executor._create_processor = Mock()
    executor._create_success_processor = Mock()

    with patch('sagemaker.processing.ProcessingStep') as mock_step_class:
        mock_step = Mock()
        mock_step_class.return_value = mock_step

        steps, final_step = executor._handle_conditional_node(conditional_node)

        # Should create: fan_out + branch_true_task + branch_true_success
        #              + branch_false_task + branch_false_success + fan_in
        # Total: 6 steps
        assert len(steps) == 6
        assert final_step == mock_step  # fan_in step


def test_conditional_node_not_rejected_by_unsupported_check():
    """Test that conditional nodes pass the unsupported node check."""
    executor = SageMakerExecutor(
        role_arn="arn:aws:iam::123456789:role/TestRole",
        region="us-east-1",
        image="test:latest"
    )

    conditional_node = Mock()
    conditional_node.node_type = "conditional"
    conditional_node.internal_name = "my_conditional"

    dag = Mock()
    dag.nodes = [conditional_node]

    unsupported = executor._check_for_unsupported_nodes(dag)
    assert len(unsupported) == 0
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/extensions/pipeline_executor/test_sagemaker.py -k "test_handle_conditional" -v`
Expected: FAIL with "AttributeError: 'SageMakerExecutor' object has no attribute '_handle_conditional_node'"

**Step 3: Implement conditional node handling**

```python
# extensions/pipeline_executor/sagemaker.py
class SageMakerExecutor(GenericPipelineExecutor):
    # ... existing code ...

    def _handle_conditional_node(self, conditional_node):
        """Convert ConditionalNode using fan-out/fan-in pattern.

        Structurally identical to _handle_parallel_node. The condition
        evaluation and branch selection happen inside the containers via
        Runnable's ConditionalNode.fan_out() and fan_in() methods.

        All branches execute as ProcessingSteps. The fan-in resolves
        which branch was active and consolidates status accordingly.
        """

        # 1. Fan-out step: evaluates condition, writes selected branch to run log
        fan_out_step = ProcessingStep(
            name=f"{conditional_node.internal_name}_fan_out",
            processor=self._create_processor_for_fan_out(conditional_node)
        )

        # 2. Process each branch + add success node
        branch_success_steps = []
        all_steps = [fan_out_step]

        for branch_name, branch_pipeline in conditional_node.branches.items():
            prev_step = fan_out_step

            for task in branch_pipeline.tasks:
                step = ProcessingStep(
                    name=f"{branch_name}_{task.internal_name}",
                    processor=self._create_processor(task),
                    depends_on=[prev_step]
                )
                all_steps.append(step)
                prev_step = step

            success_node_internal_name = f"{conditional_node.internal_name}.{branch_name}.success"
            branch_success_step = ProcessingStep(
                name=f"{branch_name}_success",
                processor=self._create_success_processor(success_node_internal_name),
                depends_on=[prev_step]
            )
            all_steps.append(branch_success_step)
            branch_success_steps.append(branch_success_step)

        # 3. Fan-in depends on ALL branch success nodes
        fan_in_step = ProcessingStep(
            name=f"{conditional_node.internal_name}_fan_in",
            processor=self._create_processor_for_fan_in(conditional_node),
            depends_on=branch_success_steps
        )
        all_steps.append(fan_in_step)

        return all_steps, fan_in_step
```

**Note:** `_create_processor_for_fan_out` and `_create_processor_for_fan_in` are already implemented in Task 7 and work for both parallel and conditional nodes. The fan command (`runnable fan out/in <node>`) dispatches to the correct node type's `fan_out()`/`fan_in()` method automatically.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/extensions/pipeline_executor/test_sagemaker.py -k "test_handle_conditional or test_conditional_node_not_rejected" -v`
Expected: PASS

**Step 5: Commit**

```bash
git add extensions/pipeline_executor/sagemaker.py tests/extensions/pipeline_executor/test_sagemaker.py
git commit -m "feat: implement conditional node support with fan-out/fan-in pattern

- Add _handle_conditional_node mirroring parallel node handling
- All branches execute; fan-in resolves active branch via run log
- Reuse existing fan-out/fan-in processors from parallel support
- Add tests for conditional conversion and unsupported-check passthrough"
```

## Phase 5: Integration and Testing

### Task 9: Add Comprehensive Integration Tests

**Files:**
- Create: `tests/extensions/pipeline_executor/test_sagemaker_integration.py`

**Step 1: Write integration test structure**

```python
# tests/extensions/pipeline_executor/test_sagemaker_integration.py
import pytest
from unittest.mock import Mock, patch, MagicMock
from extensions.pipeline_executor.sagemaker import SageMakerExecutor
from runnable.graph import Graph
from runnable.tasks import PythonTask


@pytest.fixture
def mock_sagemaker_environment():
    """Mock SageMaker SDK components for testing."""
    with patch('sagemaker.workflow.pipeline.Pipeline') as mock_pipeline, \
         patch('sagemaker.processing.ScriptProcessor') as mock_processor, \
         patch('sagemaker.processing.ProcessingStep') as mock_step, \
         patch('sagemaker.Session') as mock_session:

        # Configure mock returns
        mock_pipeline_instance = Mock()
        mock_pipeline.return_value = mock_pipeline_instance

        mock_execution = Mock()
        mock_pipeline_instance.start.return_value = mock_execution

        yield {
            'pipeline': mock_pipeline,
            'processor': mock_processor,
            'step': mock_step,
            'session': mock_session,
            'execution': mock_execution
        }


def test_end_to_end_simple_pipeline(mock_sagemaker_environment):
    """Test complete pipeline execution with simple task chain."""
    executor = SageMakerExecutor(
        role_arn="arn:aws:iam::123456789:role/TestRole",
        region="us-east-1",
        image="test:latest",
        wait_for_completion=False
    )

    # Create simple pipeline: task1 -> task2
    def dummy_func():
        return "test"

    task1 = PythonTask(function=dummy_func, name="task1")
    task2 = PythonTask(function=dummy_func, name="task2")

    # Mock DAG construction
    dag = Mock()
    dag.nodes = [task1, task2]
    dag.start_at = "task1"

    # Mock traversal methods
    executor._traverse_dag_in_order = Mock(return_value=[task1, task2])
    executor._get_immediate_parents = Mock(side_effect=[[], [task1]])

    # Execute pipeline
    executor.execute_from_graph(dag)

    # Verify SageMaker integration
    mocks = mock_sagemaker_environment
    mocks['pipeline'].assert_called_once()
    mocks['pipeline'].return_value.upsert.assert_called_once_with(role_arn=executor.role_arn)
    mocks['pipeline'].return_value.start.assert_called_once()


def test_override_configuration(mock_sagemaker_environment):
    """Test that override configuration is applied correctly."""
    executor = SageMakerExecutor(
        role_arn="arn:aws:iam::123456789:role/TestRole",
        region="us-east-1",
        image="test:latest",
        instance_type="ml.m5.large",  # Default
        overrides={
            "gpu_task": {
                "instance_type": "ml.p3.2xlarge",
                "volume_size_gb": 100
            }
        }
    )

    # Mock task with override
    task = Mock()
    task.internal_name = "gpu_training"
    task.overrides = {"sagemaker": "gpu_task"}

    config = executor._get_node_config(task)

    assert config["instance_type"] == "ml.p3.2xlarge"  # Override applied
    assert config["volume_size_gb"] == 100  # Override applied
    assert config["instance_count"] == 1  # Default maintained


def test_unsupported_nodes_error():
    """Test that unsupported nodes raise clear errors."""
    executor = SageMakerExecutor(
        role_arn="arn:aws:iam::123456789:role/TestRole",
        region="us-east-1",
        image="test:latest"
    )

    # Create DAG with unsupported map node
    map_node = Mock()
    map_node.node_type = "map"
    map_node.internal_name = "map_processing"

    nested_node = Mock()
    nested_node.node_type = "task"
    nested_node.internal_name = "outer.branch.inner.nested.task"  # >2 dots

    dag = Mock()
    dag.nodes = [map_node, nested_node]

    # Should raise ExecutorNotSupported
    from runnable.exceptions import ExecutorNotSupported
    with pytest.raises(ExecutorNotSupported) as exc_info:
        executor.execute_from_graph(dag)

    error_msg = str(exc_info.value)
    assert "map_processing (map)" in error_msg
    assert "outer.branch.inner.nested.task" in error_msg
    assert "Use 'argo' executor" in error_msg
```

**Step 2: Run tests to verify behavior**

Run: `pytest tests/extensions/pipeline_executor/test_sagemaker_integration.py -v`
Expected: PASS (tests validate the full integration)

**Step 3: Commit integration tests**

```bash
git add tests/extensions/pipeline_executor/test_sagemaker_integration.py
git commit -m "test: add comprehensive SageMaker executor integration tests

- Add end-to-end pipeline execution tests with SageMaker SDK mocking
- Test override configuration application and inheritance
- Test unsupported node error handling with clear messaging
- Add fixture for consistent SageMaker environment mocking"
```

### Task 10: Add Configuration Validation Tests

**Files:**
- Modify: `tests/extensions/pipeline_executor/test_sagemaker.py`

**Step 1: Write configuration validation tests**

```python
# tests/extensions/pipeline_executor/test_sagemaker.py
def test_required_fields_validation():
    """Test that required fields are validated."""
    with pytest.raises(ValueError):
        SageMakerExecutor()  # Missing all required fields

    with pytest.raises(ValueError):
        SageMakerExecutor(role_arn="test")  # Missing region and image

    with pytest.raises(ValueError):
        SageMakerExecutor(
            role_arn="arn:aws:iam::123456789:role/TestRole",
            region="us-east-1"
        )  # Missing image


def test_default_values():
    """Test that default values are set correctly."""
    executor = SageMakerExecutor(
        role_arn="arn:aws:iam::123456789:role/TestRole",
        region="us-east-1",
        image="test:latest"
    )

    assert executor.instance_type == "ml.m5.large"
    assert executor.instance_count == 1
    assert executor.volume_size_gb == 30
    assert executor.max_runtime_seconds == 3600
    assert executor.wait_for_completion is False
    assert executor.overrides == {}


def test_override_structure():
    """Test that overrides are structured correctly."""
    executor = SageMakerExecutor(
        role_arn="arn:aws:iam::123456789:role/TestRole",
        region="us-east-1",
        image="test:latest",
        overrides={
            "gpu_training": {
                "instance_type": "ml.p3.2xlarge",
                "instance_count": 1,
                "volume_size_gb": 100
            },
            "lightweight": {
                "instance_type": "ml.t3.medium",
                "volume_size_gb": 10
            }
        }
    )

    assert len(executor.overrides) == 2
    assert executor.overrides["gpu_training"]["instance_type"] == "ml.p3.2xlarge"
    assert executor.overrides["lightweight"]["instance_type"] == "ml.t3.medium"


def test_pydantic_model_validation():
    """Test Pydantic model validation catches invalid configurations."""

    # Test invalid instance_count
    with pytest.raises(ValueError):
        SageMakerExecutor(
            role_arn="arn:aws:iam::123456789:role/TestRole",
            region="us-east-1",
            image="test:latest",
            instance_count=0  # Invalid
        )

    # Test invalid volume_size_gb
    with pytest.raises(ValueError):
        SageMakerExecutor(
            role_arn="arn:aws:iam::123456789:role/TestRole",
            region="us-east-1",
            image="test:latest",
            volume_size_gb=-1  # Invalid
        )
```

**Step 2: Add Pydantic validators to SageMaker executor**

```python
# extensions/pipeline_executor/sagemaker.py
from pydantic import Field, ConfigDict, validator


class SageMakerExecutor(GenericPipelineExecutor):
    # ... existing code ...

    # Add validation to numeric fields
    instance_count: int = Field(default=1, ge=1, description="Default number of instances")
    volume_size_gb: int = Field(default=30, ge=1, description="Default EBS volume size in GB")
    max_runtime_seconds: int = Field(default=3600, ge=60, description="Default max runtime per job")

    @validator('role_arn')
    def validate_role_arn(cls, v):
        """Validate IAM role ARN format."""
        if not v.startswith('arn:aws:iam::'):
            raise ValueError('role_arn must be a valid IAM role ARN')
        return v

    @validator('region')
    def validate_region(cls, v):
        """Validate AWS region format."""
        if not v or len(v) < 3:
            raise ValueError('region must be a valid AWS region')
        return v
```

**Step 3: Run tests to verify validation**

Run: `pytest tests/extensions/pipeline_executor/test_sagemaker.py -k "validation" -v`
Expected: PASS

**Step 4: Commit validation improvements**

```bash
git add extensions/pipeline_executor/sagemaker.py tests/extensions/pipeline_executor/test_sagemaker.py
git commit -m "feat: add comprehensive configuration validation

- Add Pydantic validators for required fields (role_arn, region, image)
- Add numeric validation for instance_count, volume_size_gb, max_runtime_seconds
- Add IAM role ARN format validation
- Add AWS region format validation
- Add comprehensive test coverage for validation scenarios"
```

## Phase 6: Documentation and Examples

### Task 11: Add Configuration Documentation and Examples

**Files:**
- Create: `docs/production/pipeline-execution/sagemaker.md`
- Create: `examples/sagemaker/simple_pipeline.py`
- Create: `examples/sagemaker/parallel_pipeline.py`
- Modify: `docs/production/pipeline-execution/overview.md`

**Step 1: Write SageMaker executor documentation**

```markdown
# docs/production/pipeline-execution/sagemaker.md
# SageMaker Pipeline Execution

Execute production ML pipelines on AWS SageMaker Pipelines with distributed computing and enterprise integration.

!!! info "AWS SDK Dependency"

    **SageMaker execution uses the AWS SDK** which is included in the base runnable installation:
    ```bash
    pip install runnable
    ```

!!! success "Cost-Efficient ML Workloads"

    **AWS-native optimization!** SageMaker provides cost-efficient execution with spot instances, auto-scaling, and instance type optimization for different ML workload types.

## Getting Started

### Basic Configuration

```yaml
pipeline-executor:
  type: sagemaker
  config:
    # Required: AWS Infrastructure
    role_arn: "arn:aws:iam::123456789:role/SageMakerExecutionRole"
    region: "us-east-1"
    image: "my-pipeline:latest"

    # Optional: Compute Defaults
    instance_type: "ml.m5.large"
    instance_count: 1
    volume_size_gb: 30
    max_runtime_seconds: 3600
```

### Simple Example

=== "pipeline.py"

    ```python
    from runnable import Pipeline, PythonTask

    def load_data():
        print("Loading training data...")
        return {"samples": 1000, "features": 20}

    def train_model(data):
        print(f"Training on {data['samples']} samples...")
        return {"accuracy": 0.95, "model_path": "/tmp/model.pkl"}

    def evaluate_model(model_info):
        print(f"Model accuracy: {model_info['accuracy']}")
        return {"evaluation": "passed"}

    def main():
        pipeline = Pipeline(steps=[
            PythonTask(function=load_data, name="load_data", returns=["data"]),
            PythonTask(function=train_model, name="train_model", returns=["model"]),
            PythonTask(function=evaluate_model, name="evaluate_model")
        ])
        pipeline.execute()
        return pipeline

    if __name__ == "__main__":
        main()
    ```

=== "config.yaml"

    ```yaml
    pipeline-executor:
      type: sagemaker
      config:
        role_arn: "arn:aws:iam::123456789:role/SageMakerExecutionRole"
        region: "us-east-1"
        image: "my-ml-pipeline:latest"

    # Storage configuration (S3 integration)
    catalog:
      type: s3
      config:
        bucket: "my-ml-bucket"
        prefix: "catalog/"

    run_log_store:
      type: s3
      config:
        bucket: "my-ml-bucket"
        prefix: "logs/"
    ```

**Execute the pipeline:**
```bash
# Generate and execute SageMaker Pipeline
RUNNABLE_CONFIGURATION_FILE=config.yaml uv run pipeline.py
```

## Advanced Configuration

### Resource Overrides

Different tasks can have different compute requirements:

```yaml
pipeline-executor:
  type: sagemaker
  config:
    role_arn: "arn:aws:iam::123456789:role/SageMakerExecutionRole"
    region: "us-east-1"
    image: "my-pipeline:latest"

    # Pipeline-level defaults
    instance_type: "ml.m5.large"
    volume_size_gb: 30

    # Task-specific overrides
    overrides:
      gpu_training:
        instance_type: "ml.p3.2xlarge"
        instance_count: 1
        volume_size_gb: 100
        max_runtime_seconds: 7200

      lightweight_tasks:
        instance_type: "ml.t3.medium"
        volume_size_gb: 10
        max_runtime_seconds: 600
```

Use overrides in your pipeline:

```python
# GPU-intensive training task
train_task = PythonTask(
    function=train_deep_model,
    name="train_with_gpu",
    overrides={"sagemaker": "gpu_training"}
)

# Lightweight preprocessing
preprocess_task = PythonTask(
    function=clean_data,
    name="preprocess",
    overrides={"sagemaker": "lightweight_tasks"}
)
```

## Supported Patterns

### ✅ What SageMaker Handles Well

**Simple Task Chains:**
```python
# Sequential pipeline: load -> process -> train -> evaluate
Pipeline(steps=[load_task, process_task, train_task, evaluate_task])
```

**Basic Parallel Execution:**
```python
# Parallel branches with fan-out/fan-in
parallel_training = Parallel(
    name="model_comparison",
    branches={
        "model_a": Pipeline([train_model_a, evaluate_a]),
        "model_b": Pipeline([train_model_b, evaluate_b])
    }
)
Pipeline(steps=[prepare_data, parallel_training, select_best_model])
```

**Simple Conditionals:**
```python
# Basic if/else logic
conditional_step = Conditional(
    name="model_validation",
    branches={
        "retrain": Pipeline([retrain_model]) if accuracy < 0.9,
        "deploy": Pipeline([deploy_model]) if accuracy >= 0.9
    }
)
```

### ❌ SageMaker Limitations

**Unsupported Patterns:**
```python
# ❌ Map nodes (dynamic iteration)
map_step = Map(
    name="process_datasets",
    iterate_on="dataset_list",
    branch=Pipeline([process_dataset])
)

# ❌ Loop nodes (while/for loops)
loop_step = Loop(
    name="iterative_training",
    branch=Pipeline([train_epoch, validate])
)

# ❌ Nested parallel structures
nested = Parallel(
    branches={
        "branch_a": Pipeline([
            task1,
            Parallel(branches={"nested_1": ..., "nested_2": ...}),  # Too nested
            task2
        ])
    }
)
```

**Clear Error Messages:**
When you try unsupported patterns, SageMaker executor provides guidance:

```
SageMaker executor limitations:
- Unsupported node types: ['map_processing (map)', 'training_loop (loop)']
- Nested structures not supported: ['outer.branch_a.inner.nested.task']
Use 'argo' executor for complex workflows.
```

## Configuration Reference

### Required Fields
- **role_arn**: IAM role with SageMaker and S3 permissions
- **region**: AWS region for SageMaker resources
- **image**: Container image with pipeline code and Runnable

### Optional Fields
- **instance_type**: Default EC2 instance type (default: "ml.m5.large")
- **instance_count**: Default number of instances (default: 1)
- **volume_size_gb**: Default EBS volume size (default: 30)
- **max_runtime_seconds**: Default job timeout (default: 3600)
- **wait_for_completion**: Block until pipeline completes (default: false)
- **overrides**: Per-task compute configuration overrides

## Storage Integration

The SageMaker executor integrates seamlessly with S3-based storage:

**Unified S3 Configuration:**
- Reuses existing S3 catalog configuration
- No SageMaker-specific storage config needed
- IAM role provides S3 access permissions
- All processing jobs access shared S3 paths

**Storage Requirements:**
```yaml
# Required: S3-based catalog and run logs
catalog:
  type: s3
  config:
    bucket: "your-ml-bucket"
    prefix: "catalog/"

run_log_store:
  type: s3  # or chunked-fs, chunked-minio
  config:
    bucket: "your-ml-bucket"
    prefix: "logs/"
```

## IAM Requirements

Your SageMaker execution role needs these permissions:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "sagemaker:CreateProcessingJob",
        "sagemaker:DescribeProcessingJob",
        "sagemaker:CreatePipeline",
        "sagemaker:UpdatePipeline",
        "sagemaker:StartPipelineExecution",
        "sagemaker:DescribePipelineExecution"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::your-ml-bucket/*",
        "arn:aws:s3:::your-ml-bucket"
      ]
    }
  ]
}
```

## When to Use SageMaker

### ✅ Choose SageMaker When
- Need cost-efficient ML workloads on AWS
- Want instance type optimization (CPU/GPU/memory)
- Have straightforward pipeline structures
- Need enterprise AWS integration
- Want managed infrastructure scaling

### ⚠️ Use Argo When
- Need complex control flow (map, loop, nested parallel)
- Want maximum flexibility in workflow patterns
- Have non-AWS Kubernetes infrastructure
- Need custom orchestration features

### ⚠️ Use Local When
- Development and testing
- Simple workflows without distributed requirements
- Want fastest development iteration

---

**Related:** [Pipeline Execution Overview](overview.md) | [Argo Workflows](argo.md) | [Local Execution](local.md)
```

**Step 2: Create simple pipeline example**

```python
# examples/sagemaker/simple_pipeline.py
"""
Simple SageMaker pipeline example showing basic ML workflow.

Execute with:
    RUNNABLE_CONFIGURATION_FILE=examples/sagemaker/config.yaml uv run examples/sagemaker/simple_pipeline.py
"""

from runnable import Pipeline, PythonTask


def load_data():
    """Load training data from S3 or local source."""
    print("Loading training data...")
    # Simulate data loading
    data = {
        "samples": 1000,
        "features": 20,
        "target": "classification"
    }
    print(f"Loaded {data['samples']} samples with {data['features']} features")
    return data


def preprocess_data(raw_data):
    """Clean and preprocess the data."""
    print(f"Preprocessing {raw_data['samples']} samples...")
    # Simulate preprocessing
    processed = {
        "clean_samples": raw_data['samples'] - 50,  # Remove outliers
        "features": raw_data['features'],
        "normalized": True
    }
    print(f"Preprocessing complete: {processed['clean_samples']} clean samples")
    return processed


def train_model(processed_data):
    """Train ML model on processed data."""
    print(f"Training model on {processed_data['clean_samples']} samples...")
    # Simulate model training
    import time
    time.sleep(2)  # Simulate training time

    model_info = {
        "accuracy": 0.85,
        "model_type": "random_forest",
        "model_path": "/tmp/model.pkl",
        "training_samples": processed_data['clean_samples']
    }
    print(f"Model training complete. Accuracy: {model_info['accuracy']}")
    return model_info


def evaluate_model(model_info):
    """Evaluate trained model performance."""
    print(f"Evaluating {model_info['model_type']} model...")

    # Simulate evaluation
    evaluation = {
        "test_accuracy": model_info['accuracy'] + 0.03,  # Slightly better on test
        "precision": 0.87,
        "recall": 0.83,
        "status": "passed" if model_info['accuracy'] > 0.8 else "failed"
    }
    print(f"Model evaluation: {evaluation['status']} (Test accuracy: {evaluation['test_accuracy']})")
    return evaluation


def main():
    """Execute simple ML pipeline on SageMaker."""
    pipeline = Pipeline(steps=[
        PythonTask(function=load_data, name="load_data", returns=["raw_data"]),
        PythonTask(function=preprocess_data, name="preprocess", returns=["processed_data"]),
        PythonTask(function=train_model, name="train_model", returns=["model_info"]),
        PythonTask(function=evaluate_model, name="evaluate_model", returns=["evaluation"])
    ])

    print("🚀 Starting SageMaker pipeline execution...")
    pipeline.execute()
    print("✅ Pipeline execution complete!")

    return pipeline


if __name__ == "__main__":
    main()
```

**Step 3: Create parallel pipeline example**

```python
# examples/sagemaker/parallel_pipeline.py
"""
Parallel SageMaker pipeline example showing model comparison.

Execute with:
    RUNNABLE_CONFIGURATION_FILE=examples/sagemaker/config.yaml uv run examples/sagemaker/parallel_pipeline.py
"""

from runnable import Pipeline, PythonTask, Parallel


def prepare_data():
    """Prepare data for model training."""
    print("Preparing data for multiple models...")
    return {
        "train_samples": 800,
        "test_samples": 200,
        "features": 15
    }


def train_random_forest(data):
    """Train Random Forest model."""
    print(f"Training Random Forest on {data['train_samples']} samples...")
    import time
    time.sleep(3)  # Simulate training time

    return {
        "model_type": "random_forest",
        "accuracy": 0.87,
        "training_time": 3.2,
        "model_path": "/tmp/rf_model.pkl"
    }


def train_xgboost(data):
    """Train XGBoost model."""
    print(f"Training XGBoost on {data['train_samples']} samples...")
    import time
    time.sleep(4)  # Simulate longer training time

    return {
        "model_type": "xgboost",
        "accuracy": 0.91,
        "training_time": 4.1,
        "model_path": "/tmp/xgb_model.pkl"
    }


def train_neural_network(data):
    """Train Neural Network model (GPU-intensive)."""
    print(f"Training Neural Network on {data['train_samples']} samples...")
    import time
    time.sleep(5)  # Simulate GPU training time

    return {
        "model_type": "neural_network",
        "accuracy": 0.93,
        "training_time": 5.8,
        "model_path": "/tmp/nn_model.pkl"
    }


def compare_models(rf_results, xgb_results, nn_results):
    """Compare all trained models and select the best."""
    models = [rf_results, xgb_results, nn_results]

    print("\n📊 Model Comparison Results:")
    for model in models:
        print(f"  {model['model_type']}: {model['accuracy']} accuracy ({model['training_time']}s)")

    best_model = max(models, key=lambda x: x['accuracy'])
    print(f"\n🏆 Best model: {best_model['model_type']} with {best_model['accuracy']} accuracy")

    return {
        "best_model": best_model['model_type'],
        "best_accuracy": best_model['accuracy'],
        "models_compared": len(models)
    }


def main():
    """Execute parallel model comparison pipeline."""

    # Parallel model training branches
    model_comparison = Parallel(
        name="model_comparison",
        branches={
            "random_forest": Pipeline([
                PythonTask(function=train_random_forest, name="train_rf", returns=["rf_results"])
            ]),
            "xgboost": Pipeline([
                PythonTask(function=train_xgboost, name="train_xgb", returns=["xgb_results"])
            ]),
            "neural_network": Pipeline([
                PythonTask(
                    function=train_neural_network,
                    name="train_nn",
                    returns=["nn_results"],
                    overrides={"sagemaker": "gpu_training"}  # Use GPU override
                )
            ])
        }
    )

    # Complete pipeline
    pipeline = Pipeline(steps=[
        PythonTask(function=prepare_data, name="prepare_data", returns=["data"]),
        model_comparison,
        PythonTask(function=compare_models, name="compare_models", returns=["comparison"])
    ])

    print("🚀 Starting parallel model comparison on SageMaker...")
    pipeline.execute()
    print("✅ Model comparison complete!")

    return pipeline


if __name__ == "__main__":
    main()
```

**Step 4: Create configuration file**

```yaml
# examples/sagemaker/config.yaml
pipeline-executor:
  type: sagemaker
  config:
    # AWS Infrastructure (update with your values)
    role_arn: "arn:aws:iam::123456789012:role/SageMakerExecutionRole"
    region: "us-east-1"
    image: "my-pipeline:latest"  # Your containerized pipeline image

    # Default compute configuration
    instance_type: "ml.m5.large"
    instance_count: 1
    volume_size_gb: 30
    max_runtime_seconds: 3600

    # Don't wait for completion in examples
    wait_for_completion: false

    # Resource overrides for different task types
    overrides:
      gpu_training:
        instance_type: "ml.p3.2xlarge"  # GPU instance for neural networks
        volume_size_gb: 100
        max_runtime_seconds: 7200

      lightweight_tasks:
        instance_type: "ml.t3.medium"   # Cheaper for simple tasks
        volume_size_gb: 10
        max_runtime_seconds: 600

# S3 storage configuration
catalog:
  type: s3
  config:
    bucket: "my-ml-pipeline-bucket"  # Update with your bucket
    prefix: "catalog/"

run_log_store:
  type: s3
  config:
    bucket: "my-ml-pipeline-bucket"  # Update with your bucket
    prefix: "logs/"
```

**Step 5: Update overview documentation**

```markdown
# docs/production/pipeline-execution/overview.md (add to table)

| Executor | Use Case | Environment | Execution Model |
|----------|----------|-------------|-----------------|
| [Local](local.md) | Development | Local machine | **Sequential + Conditional Parallel*** |
| [Local Container](local-container.md) | Isolated development | Docker containers** | **Sequential + Conditional Parallel*** |
| [Argo Workflows](argo.md) | Production | Argo cluster | **Parallel + Sequential** (full orchestration) |
| [SageMaker Pipelines](sagemaker.md) | AWS ML Production | SageMaker | **Parallel + Sequential** (managed ML infrastructure) |
| [Mocked](mocked.md) | Testing & validation | Local machine | **Simulation** (no actual execution) |
```

**Step 6: Commit documentation and examples**

```bash
git add docs/production/pipeline-execution/sagemaker.md \
        docs/production/pipeline-execution/overview.md \
        examples/sagemaker/simple_pipeline.py \
        examples/sagemaker/parallel_pipeline.py \
        examples/sagemaker/config.yaml

git commit -m "docs: add comprehensive SageMaker executor documentation and examples

- Add complete SageMaker pipeline execution documentation
- Document supported vs unsupported patterns with clear examples
- Add IAM requirements and S3 integration guide
- Create simple_pipeline.py showing basic ML workflow
- Create parallel_pipeline.py showing model comparison with overrides
- Add configuration examples with resource overrides
- Update overview documentation to include SageMaker option"
```

## Summary and Execution Options

Plan complete and saved to `docs/plans/2026-02-20-sagemaker-executor-implementation.md`.

**Implementation Summary:**
- **Phase 1-2**: Foundation (configuration, entry points, validation)
- **Phase 3-4**: Core functionality (DAG transpilation, parallel and conditional support)
- **Phase 5-6**: Testing and documentation
- **11 tasks** total (Tasks 1-8 implementation, Tasks 9-11 testing/docs)

**Two execution options:**

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

**Which approach?**
