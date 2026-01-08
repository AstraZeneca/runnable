# Branch-Aware Tasks Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make tasks branch-aware using partitioned storage, eliminating the `reduced` flag complexity and enabling natural parameter scoping.

**Architecture:** Tasks receive `internal_branch_name` from TaskNode and use it to access scoped parameters from partitioned storage. Map nodes use discovery-based aggregation instead of tracking branch returns. This eliminates all `reduced` flag logic.

**Tech Stack:** Python, Pydantic models, pytest for testing, existing runnable framework patterns

---

## Task 1: Remove reduced Flag from Parameter Classes

**Files:**
- Modify: `runnable/datastore.py:63-137`
- Test: `tests/runnable/test_datastore.py`

**Step 1: Write failing test for parameter classes without reduced flag**

```python
def test_json_parameter_without_reduced_flag():
    """Test JsonParameter can be created without reduced field."""
    param = JsonParameter(kind="json", value={"test": "data"})
    # Should not have reduced attribute
    assert not hasattr(param, 'reduced')

def test_metric_parameter_without_reduced_flag():
    """Test MetricParameter can be created without reduced field."""
    param = MetricParameter(kind="metric", value=42)
    assert not hasattr(param, 'reduced')

def test_object_parameter_without_reduced_flag():
    """Test ObjectParameter can be created without reduced field."""
    param = ObjectParameter(kind="object", value="test_object")
    assert not hasattr(param, 'reduced')
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/runnable/test_datastore.py::test_json_parameter_without_reduced_flag -v`
Expected: FAIL with "AssertionError: assert not True"

**Step 3: Remove reduced field from JsonParameter**

In `runnable/datastore.py`:

```python
class JsonParameter(BaseModel):
    kind: Literal["json"]
    value: JSONType
    # reduced: bool = True  # REMOVED

    @computed_field  # type: ignore
    @property
    def description(self) -> JSONType:
        # truncate value if its longer than 10 chars
        return (
            self.value
            if not isinstance(self.value, str) or len(self.value) <= 10
            else f"{self.value[:10]}..."
        )

    def get_value(self) -> JSONType:
        return self.value
```

**Step 4: Remove reduced field from MetricParameter**

```python
class MetricParameter(BaseModel):
    kind: Literal["metric"]
    value: JSONType
    # reduced: bool = True  # REMOVED

    @computed_field  # type: ignore
    @property
    def description(self) -> JSONType:
        # truncate value if its longer than 10 chars
        return (
            self.value
            if not isinstance(self.value, str) or len(self.value) <= 10
            else f"{self.value[:10]}..."
        )

    def get_value(self) -> JSONType:
        return self.value
```

**Step 5: Remove reduced field from ObjectParameter**

```python
class ObjectParameter(BaseModel):
    kind: Literal["object"]
    value: str  # The name of the pickled object
    # reduced: bool = True  # REMOVED

    @computed_field  # type: ignore
    @property
    def description(self) -> str:
        current_context = context.get_run_context()
        if current_context and current_context.object_serialisation:
            return f"Pickled object stored in catalog as: {self.value}"

        return f"Object stored in memory as: {self.value}"

    @property
    def file_name(self) -> str:
        current_context = context.get_run_context()
        if current_context is None:
            raise RuntimeError("No run context available")
        return f"{self.value}{current_context.pickler.extension}"

    def get_value(self) -> Any:
        current_context = context.get_run_context()
        if current_context is None:
            raise RuntimeError("No run context available")

        # If there was no serialisation, return the object from the return objects
        if not current_context.object_serialisation:
            return current_context.return_objects[self.value]

        # If the object was serialised, get it from the catalog
        catalog_handler = current_context.catalog
        catalog_handler.get(name=self.file_name)
        obj = current_context.pickler.load(path=self.file_name)
        os.remove(self.file_name)  # Remove after loading
        return obj

    def put_object(self, data: Any) -> None:
        current_context = context.get_run_context()
        if current_context is None:
            raise RuntimeError("No run context available")

        if not current_context.object_serialisation:
            current_context.return_objects[self.value] = data
            return

        # If the object was serialised, put it in the catalog
        current_context.pickler.dump(data=data, path=self.file_name)

        catalog_handler = current_context.catalog
        catalog_handler.put(name=self.file_name)
        os.remove(self.file_name)  # Remove after loading
```

**Step 6: Run test to verify it passes**

Run: `uv run pytest tests/runnable/test_datastore.py::test_json_parameter_without_reduced_flag -v`
Expected: PASS

**Step 7: Commit**

```bash
git add runnable/datastore.py tests/runnable/test_datastore.py
git commit -m "feat: remove reduced flag from Parameter classes"
```

## Task 2: Add Branch Awareness to BaseTaskType

**Files:**
- Modify: `runnable/tasks.py:126-276`
- Test: `tests/runnable/test_tasks_branch_aware.py`

**Step 1: Write failing test for branch-aware BaseTaskType**

Create `tests/runnable/test_tasks_branch_aware.py`:

```python
import pytest
from unittest.mock import Mock, patch
from runnable.tasks import BaseTaskType
from runnable.datastore import JsonParameter


class TestBranchAwareTask(BaseTaskType):
    task_type: str = "test"

    def execute(self):
        return {"test": "result"}


def test_base_task_type_has_internal_branch_name_field():
    """Test BaseTaskType accepts internal_branch_name parameter."""
    task = TestBranchAwareTask(internal_branch_name="test.branch")
    assert task.internal_branch_name == "test.branch"


def test_base_task_type_internal_branch_name_defaults_to_none():
    """Test internal_branch_name defaults to None."""
    task = TestBranchAwareTask()
    assert task.internal_branch_name is None


@patch('runnable.context.get_run_context')
def test_get_scoped_parameters_uses_branch_context(mock_get_context):
    """Test that get_scoped_parameters uses internal_branch_name."""
    # Setup mock context
    mock_context = Mock()
    mock_run_log_store = Mock()
    mock_context.run_log_store = mock_run_log_store
    mock_context.run_id = "test_run"
    mock_get_context.return_value = mock_context

    # Test with branch context
    task = TestBranchAwareTask(internal_branch_name="map.iteration_1")
    task._get_scoped_parameters()

    mock_run_log_store.get_parameters.assert_called_once_with(
        run_id="test_run",
        internal_branch_name="map.iteration_1"
    )


@patch('runnable.context.get_run_context')
def test_set_scoped_parameters_uses_branch_context(mock_get_context):
    """Test that set_scoped_parameters uses internal_branch_name."""
    # Setup mock context
    mock_context = Mock()
    mock_run_log_store = Mock()
    mock_context.run_log_store = mock_run_log_store
    mock_context.run_id = "test_run"
    mock_get_context.return_value = mock_context

    # Test with branch context
    task = TestBranchAwareTask(internal_branch_name="map.iteration_1")
    params = {"result": JsonParameter(kind="json", value="test")}
    task._set_scoped_parameters(params)

    mock_run_log_store.set_parameters.assert_called_once_with(
        parameters=params,
        run_id="test_run",
        internal_branch_name="map.iteration_1"
    )
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/runnable/test_tasks_branch_aware.py -v`
Expected: FAIL with "AttributeError: 'TestBranchAwareTask' object has no attribute 'internal_branch_name'"

**Step 3: Add internal_branch_name field to BaseTaskType**

In `runnable/tasks.py`:

```python
class BaseTaskType(BaseModel):
    """A base task class which does the execution of command defined by the user."""

    task_type: str = Field(serialization_alias="command_type")
    internal_branch_name: Optional[str] = Field(
        default=None,
        description="Branch context for partitioned parameter storage"
    )
    secrets: List[str] = Field(
        default_factory=list
    )  # A list of secrets to expose by secrets manager
    returns: List[TaskReturns] = Field(
        default_factory=list, alias="returns"
    )  # The return values of the task

    model_config = ConfigDict(extra="forbid")
```

**Step 4: Add helper methods for scoped parameter access**

```python
    def _get_scoped_parameters(self) -> Dict[str, Parameter]:
        """Get parameters from appropriate partition based on branch context."""
        return self._context.run_log_store.get_parameters(
            run_id=self._context.run_id,
            internal_branch_name=self.internal_branch_name
        )

    def _set_scoped_parameters(self, parameters: Dict[str, Parameter]) -> None:
        """Set parameters to appropriate partition based on branch context."""
        self._context.run_log_store.set_parameters(
            parameters=parameters,
            run_id=self._context.run_id,
            internal_branch_name=self.internal_branch_name
        )
```

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/runnable/test_tasks_branch_aware.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add runnable/tasks.py tests/runnable/test_tasks_branch_aware.py
git commit -m "feat: add branch awareness to BaseTaskType"
```

## Task 3: Replace resolve_unreduced_parameters with Scoped Access

**Files:**
- Modify: `runnable/tasks.py:254-276`
- Modify: `runnable/tasks.py:execute_command method around line 400`
- Test: `tests/runnable/test_tasks_branch_aware.py`

**Step 1: Write test for eliminating resolve_unreduced_parameters**

Add to `tests/runnable/test_tasks_branch_aware.py`:

```python
@patch('runnable.context.get_run_context')
def test_execute_command_uses_scoped_parameters_directly(mock_get_context):
    """Test execute_command gets parameters from scoped partition directly."""
    # Setup mock context
    mock_context = Mock()
    mock_run_log_store = Mock()
    mock_context.run_log_store = mock_run_log_store
    mock_context.run_id = "test_run"
    mock_get_context.return_value = mock_context

    # Mock scoped parameters
    scoped_params = {
        "input": JsonParameter(kind="json", value="test_input")
    }
    mock_run_log_store.get_parameters.return_value = scoped_params

    # Test task with branch context
    task = TestBranchAwareTask(internal_branch_name="map.iteration_1")

    # Should call get_parameters with branch context, not resolve_unreduced_parameters
    with patch.object(task, '_get_scoped_parameters', return_value=scoped_params) as mock_get_scoped:
        # Simulate part of execute_command that gets parameters
        params = task._get_scoped_parameters()

    mock_get_scoped.assert_called_once()
    assert params == scoped_params
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/runnable/test_tasks_branch_aware.py::test_execute_command_uses_scoped_parameters_directly -v`
Expected: PASS (this is testing the new interface we already added)

**Step 3: Remove resolve_unreduced_parameters method**

In `runnable/tasks.py`, remove the entire method:

```python
    # TODO: Bring in loop variable
    def resolve_unreduced_parameters(
        self,
        iter_variable: Optional[IterableParameterModel] = None,
    ):
        """Resolve the unreduced parameters."""
        # DELETE THIS ENTIRE METHOD - NO LONGER NEEDED
```

**Step 4: Update execute_command to use scoped parameters**

Find the execute_command method around line 400 and replace parameter resolution:

```python
def execute_command(
    self,
    iter_variable: Optional[IterableParameterModel] = None,
) -> StepAttempt:
    """The function to execute the command.

    And map_variable is sent in as an argument into the function.

    Args:
        iter_variable (dict, optional): If the command is part of map node, the value of map. Defaults to None.
    """

    # OLD CODE TO REPLACE:
    # params = self.resolve_unreduced_parameters(iter_variable=iter_variable)

    # NEW CODE:
    params = self._get_scoped_parameters()

    logger.info(f"Parameters available for the execution: {params}")

    task_console.log("Parameters available for the execution:")
    task_console.log(params)

    logger.debug(f"Resolved parameters: {params}")

    if not allow_complex:
        params = {
            key: value
            for key, value in params.items()
            if isinstance(value, JsonParameter)
            or isinstance(value, MetricParameter)
        }

    parameters_in = copy.deepcopy(params)
    # ... rest of method unchanged
```

**Step 5: Run existing tests to verify no regression**

Run: `uv run pytest tests/runnable/test_tasks.py -v`
Expected: Some tests may fail due to removed method - will fix in next step

**Step 6: Commit**

```bash
git add runnable/tasks.py
git commit -m "feat: replace resolve_unreduced_parameters with scoped parameter access"
```

## Task 4: Update TaskNode to Pass Branch Context

**Files:**
- Modify: `extensions/nodes/task.py:30-42`
- Test: `tests/extensions/nodes/test_task_branch_aware.py`

**Step 1: Write test for TaskNode passing internal_branch_name**

Create `tests/extensions/nodes/test_task_branch_aware.py`:

```python
import pytest
from unittest.mock import Mock, patch
from extensions.nodes.task import TaskNode


def test_parse_from_config_passes_internal_branch_name():
    """Test TaskNode.parse_from_config passes internal_branch_name to task."""

    config = {
        "name": "test_task",
        "command_type": "python",
        "command": "test_function"
    }

    # Mock create_task to capture the task_config
    with patch('extensions.nodes.task.create_task') as mock_create_task:
        mock_task = Mock()
        mock_create_task.return_value = mock_task

        # Create TaskNode with internal_branch_name
        task_node = TaskNode.parse_from_config(config)
        task_node.internal_branch_name = "test.branch"

        # Re-parse to test the internal_branch_name passing
        task_node = TaskNode.parse_from_config(config)

        # Verify create_task was called
        mock_create_task.assert_called_once()
        called_config = mock_create_task.call_args[0][0]

        # The config should not yet include internal_branch_name
        # (this test will guide us to add it)
        assert "internal_branch_name" not in called_config


def test_task_node_inherits_internal_branch_name_from_base_node():
    """Test TaskNode has internal_branch_name from BaseNode."""
    config = {
        "name": "test_task",
        "command_type": "python",
        "command": "test_function"
    }

    with patch('extensions.nodes.task.create_task') as mock_create_task:
        mock_task = Mock()
        mock_create_task.return_value = mock_task

        task_node = TaskNode.parse_from_config(config)

        # TaskNode should have internal_branch_name attribute (inherited from BaseNode)
        assert hasattr(task_node, 'internal_branch_name')
        assert task_node.internal_branch_name is None  # Default value
```

**Step 2: Run test to verify current behavior**

Run: `uv run pytest tests/extensions/nodes/test_task_branch_aware.py -v`
Expected: PASS (establishes baseline)

**Step 3: Write test for the desired behavior**

Add to the test file:

```python
def test_parse_from_config_includes_internal_branch_name_in_task_config():
    """Test parse_from_config passes internal_branch_name to create_task."""

    config = {
        "name": "test_task",
        "command_type": "python",
        "command": "test_function"
    }

    with patch('extensions.nodes.task.create_task') as mock_create_task:
        mock_task = Mock()
        mock_task.internal_branch_name = None  # Will be set by create_task
        mock_create_task.return_value = mock_task

        # Create TaskNode and set branch context
        task_node = TaskNode.parse_from_config(config)
        task_node.internal_branch_name = "map.iteration_1"

        # Parse again to see if branch context gets passed
        # (We need to modify parse_from_config to do this)
        task_node_with_branch = TaskNode.parse_from_config({
            **config,
            "internal_branch_name": "map.iteration_1"
        })

        # Verify the task config includes internal_branch_name
        called_config = mock_create_task.call_args[0][0]
        assert called_config["internal_branch_name"] == "map.iteration_1"
```

**Step 4: Run test to verify it fails**

Run: `uv run pytest tests/extensions/nodes/test_task_branch_aware.py::test_parse_from_config_includes_internal_branch_name_in_task_config -v`
Expected: FAIL - internal_branch_name not in config

**Step 5: Modify TaskNode.parse_from_config to pass internal_branch_name**

In `extensions/nodes/task.py`:

```python
@classmethod
def parse_from_config(cls, config: Dict[str, Any]) -> "TaskNode":
    # separate task config from node config
    task_config = {
        k: v for k, v in config.items() if k not in TaskNode.model_fields.keys()
    }
    node_config = {
        k: v for k, v in config.items() if k in TaskNode.model_fields.keys()
    }

    # Pass internal_branch_name to task if present in config
    if "internal_branch_name" in config:
        task_config["internal_branch_name"] = config["internal_branch_name"]

    executable = create_task(task_config)
    return cls(executable=executable, **node_config, **task_config)
```

**Step 6: Run test to verify it passes**

Run: `uv run pytest tests/extensions/nodes/test_task_branch_aware.py -v`
Expected: PASS

**Step 7: Commit**

```bash
git add extensions/nodes/task.py tests/extensions/nodes/test_task_branch_aware.py
git commit -m "feat: TaskNode passes internal_branch_name to tasks"
```

## Task 5: Update Map Node for Discovery-Based Aggregation

**Files:**
- Modify: `extensions/nodes/map.py:190-207` (fan_out changes)
- Modify: `extensions/nodes/map.py:440-469` (fan_in changes)
- Test: `tests/extensions/nodes/test_map_discovery.py`

**Step 1: Write test for discovery-based map aggregation**

Create `tests/extensions/nodes/test_map_discovery.py`:

```python
import pytest
from unittest.mock import Mock, patch
from extensions.nodes.map import MapNode
from runnable.datastore import JsonParameter, BranchLog


@patch('runnable.context.get_run_context')
def test_fan_in_discovers_parameters_from_branch_partitions(mock_get_context):
    """Test fan_in discovers and aggregates parameters from branch partitions."""

    # Setup mock context
    mock_context = Mock()
    mock_run_log_store = Mock()
    mock_context.run_log_store = mock_run_log_store
    mock_context.run_id = "test_run"
    mock_get_context.return_value = mock_context

    # Mock branch parameters
    branch_1_params = {
        "result": JsonParameter(kind="json", value="value_1"),
        "metric": JsonParameter(kind="json", value=10)
    }
    branch_2_params = {
        "result": JsonParameter(kind="json", value="value_2"),
        "metric": JsonParameter(kind="json", value=20)
    }

    def get_parameters_side_effect(run_id, internal_branch_name):
        if internal_branch_name == "map_node.0":
            return branch_1_params
        elif internal_branch_name == "map_node.1":
            return branch_2_params
        return {}

    mock_run_log_store.get_parameters.side_effect = get_parameters_side_effect

    # Create map node
    map_node = MapNode(
        name="map_node",
        internal_name="map_node",
        internal_branch_name=None,  # Root context
        iterate_on="items",
        iterate_as="item"
    )
    map_node._context = mock_context

    # Mock the iteration values
    with patch.object(map_node, '_get_map_iteration_values', return_value=["0", "1"]):
        # Call fan_in
        map_node.fan_in(reducer_f=list)  # list aggregation

    # Verify parameters were fetched from branch partitions
    assert mock_run_log_store.get_parameters.call_count == 2
    mock_run_log_store.get_parameters.assert_any_call(
        run_id="test_run", internal_branch_name="map_node.0"
    )
    mock_run_log_store.get_parameters.assert_any_call(
        run_id="test_run", internal_branch_name="map_node.1"
    )

    # Verify aggregated parameters were set to parent partition
    mock_run_log_store.set_parameters.assert_called_once()
    call_args = mock_run_log_store.set_parameters.call_args

    assert call_args[1]["run_id"] == "test_run"
    assert call_args[1]["internal_branch_name"] is None  # Parent context

    # Check aggregated parameters
    aggregated_params = call_args[1]["parameters"]
    assert "result" in aggregated_params
    assert "metric" in aggregated_params
    assert aggregated_params["result"].value == ["value_1", "value_2"]
    assert aggregated_params["metric"].value == [10, 20]


def test_fan_out_creates_branch_partitions_only():
    """Test fan_out only creates branch logs without parameter tracking."""

    # This test ensures we don't track branch_returns anymore
    map_node = MapNode(
        name="map_node",
        internal_name="map_node",
        iterate_on="items",
        iterate_as="item"
    )

    # Should not have branch_returns attribute or should be empty
    assert not hasattr(map_node, 'branch_returns') or len(map_node.branch_returns) == 0
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/extensions/nodes/test_map_discovery.py -v`
Expected: FAIL - method doesn't exist or behaves differently

**Step 3: Remove branch_returns tracking from MapNode**

In `extensions/nodes/map.py`, find and remove/modify the fan_out method around line 190:

```python
def fan_out(self, iter_variable: Optional[IterableParameterModel] = None):
    """Create branch logs for map iterations without parameter tracking."""

    # OLD CODE REMOVED - no more branch_returns tracking:
    # raw_parameters = {}
    # if iter_variable and iter_variable.map_variable:
    #     for _, v in iter_variable.map_variable.items():
    #         for branch_return in self.branch_returns:
    #             param_name, param_type = branch_return
    #             raw_parameters[f"{v.value}_{param_name}"] = param_type.model_copy()
    # else:
    #     for branch_return in self.branch_returns:
    #         param_name, param_type = branch_return
    #         raw_parameters[f"{param_name}"] = param_type.model_copy()

    # NEW CODE - just create branch logs like ParallelNode:
    for branch_name in self._get_branch_names(iter_variable):
        branch_log = self._context.run_log_store.create_branch_log(
            internal_branch_name=branch_name
        )
        self._context.run_log_store.add_branch_log(branch_log, self._context.run_id)

    # NO MORE: self._context.run_log_store.set_parameters(...)
```

**Step 4: Implement discovery-based fan_in method**

Replace the fan_in method around line 440:

```python
def fan_in(self, reducer_f=None):
    """Aggregate parameters from branch partitions using discovery."""

    if reducer_f is None:
        reducer_f = list  # Default to list aggregation

    # Discover all parameters from branch partitions
    branch_names = self._get_branch_names()
    all_param_names = set()
    branch_data = {}

    for branch_name in branch_names:
        try:
            branch_params = self._context.run_log_store.get_parameters(
                run_id=self._context.run_id,
                internal_branch_name=branch_name
            )
            branch_data[branch_name] = branch_params
            all_param_names.update(branch_params.keys())
        except exceptions.EntityNotFoundError:
            # Branch has no parameters - skip
            continue

    # Aggregate each parameter across all branches
    aggregated_params = {}
    for param_name in all_param_names:
        values_to_aggregate = []
        for branch_name, branch_params in branch_data.items():
            if param_name in branch_params:
                values_to_aggregate.append(branch_params[param_name].value)

        # Apply reducer function
        if values_to_aggregate:
            aggregated_value = reducer_f(*values_to_aggregate) if len(values_to_aggregate) > 1 else values_to_aggregate[0]
            aggregated_params[param_name] = JsonParameter(
                kind="json",
                value=aggregated_value
            )

    # Store aggregated parameters in parent partition
    if aggregated_params:
        self._context.run_log_store.set_parameters(
            parameters=aggregated_params,
            run_id=self._context.run_id,
            internal_branch_name=self.internal_branch_name
        )
```

**Step 5: Add helper method for getting branch names**

```python
def _get_branch_names(self, iter_variable: Optional[IterableParameterModel] = None) -> List[str]:
    """Get list of branch names for map iterations."""
    branch_names = []

    # This logic should match existing map iteration logic
    if iter_variable and iter_variable.map_variable:
        for _, v in iter_variable.map_variable.items():
            branch_names.append(f"{self.internal_name}.{v.value}")
    else:
        # Need to derive from iterate_on configuration
        # This would need the actual iteration values
        pass

    return branch_names
```

**Step 6: Run test to verify it passes**

Run: `uv run pytest tests/extensions/nodes/test_map_discovery.py -v`
Expected: PASS

**Step 7: Commit**

```bash
git add extensions/nodes/map.py tests/extensions/nodes/test_map_discovery.py
git commit -m "feat: implement discovery-based parameter aggregation in Map node"
```

## Task 6: Clean Up Parameter Naming Logic

**Files:**
- Modify: `runnable/tasks.py:485-491` (output parameter naming)
- Modify: `runnable/tasks.py:735-740` (similar pattern)
- Modify: `runnable/tasks.py:1000-1005` (similar pattern)
- Modify: `runnable/tasks.py:1210-1215` (similar pattern)
- Test: `tests/runnable/test_tasks_branch_aware.py`

**Step 1: Write test for clean parameter naming**

Add to `tests/runnable/test_tasks_branch_aware.py`:

```python
def test_task_output_parameters_use_clean_names():
    """Test tasks don't prefix parameter names when using partitioned storage."""

    # With partitioned storage, parameter names should be clean
    # No more "iteration_1_result" - just "result" in the right partition

    task = TestBranchAwareTask(internal_branch_name="map_node.iteration_1")

    # Mock task return
    with patch.object(task, 'execute', return_value="test_result"):
        # Mock task returns configuration
        task.returns = [Mock(name="result")]

        # When task processes output parameters, should use clean names
        # regardless of iter_variable presence
        iter_variable = Mock()
        iter_variable.map_variable = {"0": Mock(value="iteration_1")}

        # The parameter name should be clean, not prefixed
        # This will guide us to remove the prefixing logic
        output_params = task._process_output_parameters(
            outputs={"result": "test_result"},
            iter_variable=iter_variable
        )

        # Should have clean parameter name
        assert "result" in output_params
        assert "iteration_1_result" not in output_params
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/runnable/test_tasks_branch_aware.py::test_task_output_parameters_use_clean_names -v`
Expected: FAIL - current code still prefixes names

**Step 3: Remove parameter name prefixing in execute_command**

In `runnable/tasks.py` around line 485:

```python
# OLD CODE TO REPLACE:
# # TODO: Handle loop variable
# param_name = task_return.name
# if iter_variable and iter_variable.map_variable:
#     for _, v in iter_variable.map_variable.items():
#         param_name = f"{v.value}_{param_name}"
#
# output_parameters[param_name] = output_parameter

# NEW CODE:
# Clean parameter names - partition handles scoping
param_name = task_return.name
output_parameters[param_name] = output_parameter
```

**Step 4: Remove similar prefixing logic at line ~738**

```python
# OLD CODE TO REPLACE:
# param_name = task_return.name
# if iter_variable and iter_variable.map_variable:
#     for _, v in iter_variable.map_variable.items():
#         param_name = f"{v.value}_{param_name}"

# NEW CODE:
param_name = task_return.name
```

**Step 5: Remove similar prefixing logic at line ~1001**

```python
# OLD CODE TO REPLACE:
# param_name = task_return.name
# if iter_variable and iter_variable.map_variable:
#     for _, v in iter_variable.map_variable.items():
#         param_name = f"{v.value}_{param_name}"

# NEW CODE:
param_name = task_return.name
```

**Step 6: Remove similar prefixing logic at line ~1213**

```python
# OLD CODE TO REPLACE:
# param_name = task_return.name
# if iter_variable and iter_variable.map_variable:
#     for _, v in iter_variable.map_variable.items():
#         param_name = f"{v.value}_{param_name}"

# NEW CODE:
param_name = task_return.name
```

**Step 7: Run test to verify it passes**

Run: `uv run pytest tests/runnable/test_tasks_branch_aware.py::test_task_output_parameters_use_clean_names -v`
Expected: PASS

**Step 8: Commit**

```bash
git add runnable/tasks.py tests/runnable/test_tasks_branch_aware.py
git commit -m "feat: remove parameter name prefixing - partitioned storage handles scoping"
```

## Task 7: Clean Up Legacy reduced Flag Usage

**Step 1: Search for remaining reduced flag usage**

Run: `grep -r "\.reduced" --include="*.py" .`
Expected: Find remaining references to clean up

**Step 2: Remove reduced references from tasks.py**

Find and remove any remaining `.reduced` checks around line 687:

```python
# OLD CODE TO REMOVE:
# unprocessed_params = [
#     k for k, v in copy_params.items() if not v.reduced
# ]

# NEW CODE - no filtering needed:
# All parameters are naturally scoped, no filtering required
```

**Step 3: Remove reduced references from map.py**

Find and remove any remaining `param.reduced = True` assignments around line 458:

```python
# OLD CODE TO REMOVE:
# params[param_name].reduced = True

# NEW CODE - nothing needed, parameters are naturally scoped
```

**Step 4: Verify no reduced references remain**

Run: `grep -r "reduced" --include="*.py" . | grep -v test | grep -v docs`
Expected: Only test files and docs should reference reduced

**Step 5: Run all tests to verify no regressions**

Run: `uv run pytest tests/runnable/ tests/extensions/nodes/ -v`
Expected: All tests pass

**Step 6: Commit**

```bash
git add .
git commit -m "feat: remove all legacy reduced flag usage"
```

## Task 7: Integration Testing

**Files:**
- Create: `tests/integration/test_branch_aware_execution.py`
- Test complete workflow

**Step 1: Write comprehensive integration test**

Create `tests/integration/test_branch_aware_execution.py`:

```python
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch

from runnable import Pipeline
from runnable.tasks import PythonTask
from runnable.datastore import JsonParameter
from extensions.run_log_store.partitioned_fs import FileSystemPartitionedRunLogStore


def simple_task_function(input_param):
    """Simple function for testing."""
    return f"processed_{input_param}"


def test_branch_aware_task_execution_with_partitioned_storage():
    """Test complete workflow: tasks get scoped parameters from partitioned storage."""

    with tempfile.TemporaryDirectory() as temp_dir:
        # Setup partitioned run log store
        run_log_store = FileSystemPartitionedRunLogStore()
        run_log_store.log_folder = temp_dir

        # Create task with branch context
        task = PythonTask(
            function=simple_task_function,
            name="process_task",
            internal_branch_name="map_node.iteration_1"
        )

        # Setup run context
        run_id = "test_branch_aware_run"

        with patch('runnable.context.get_run_context') as mock_get_context:
            mock_context = mock_get_context.return_value
            mock_context.run_log_store = run_log_store
            mock_context.run_id = run_id

            # Create run log
            run_log_store.create_run_log(run_id)

            # Set parameters in branch partition
            branch_params = {
                "input_param": JsonParameter(kind="json", value="test_input")
            }
            run_log_store.set_parameters(
                run_id=run_id,
                parameters=branch_params,
                internal_branch_name="map_node.iteration_1"
            )

            # Execute task - should get parameters from branch partition
            result = task.execute_command()

            # Verify task got scoped parameters
            # (This tests the full integration)
            assert result.status == "SUCCESS"

            # Verify output parameters were stored in branch partition
            output_params = run_log_store.get_parameters(
                run_id=run_id,
                internal_branch_name="map_node.iteration_1"
            )

            # Should contain both input and output parameters
            assert "input_param" in output_params
            # Check for output parameters based on task implementation


def test_map_node_discovery_aggregation_integration():
    """Test map node discovers and aggregates parameters correctly."""

    # This would test the full map workflow:
    # 1. Map node creates branch partitions
    # 2. Tasks execute in branch contexts
    # 3. Map node aggregates via discovery
    # 4. Next tasks see aggregated parameters

    pass  # Implement based on actual map node structure
```

**Step 2: Run integration test**

Run: `uv run pytest tests/integration/test_branch_aware_execution.py -v`
Expected: Tests reveal any integration issues

**Step 3: Fix any integration issues found**

Based on test results, fix any issues in the integration between components.

**Step 4: Verify complete test suite passes**

Run: `uv run pytest tests/ -v`
Expected: All tests pass

**Step 5: Final commit**

```bash
git add tests/integration/test_branch_aware_execution.py
git commit -m "test: add comprehensive integration tests for branch-aware execution"
```

---

Plan complete and saved to `docs/plans/2026-01-08-branch-aware-tasks.md`. Two execution options:

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

Which approach?
