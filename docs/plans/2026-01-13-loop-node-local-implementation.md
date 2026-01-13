# Loop Node Local Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement loop node for local execution only, with proper placeholder resolution following existing map patterns.

**Architecture:** Follow existing composite node patterns, refactor placeholder resolution to support both map and loop variables.

**Tech Stack:** Python, Pydantic (local execution only, no Argo)

---

## Task 1: Add Loop Placeholder Constant

**Files:**
- Modify: `runnable/defaults.py`
- Test: `tests/runnable/test_defaults.py`

**Step 1: Write the failing test**

```python
def test_loop_placeholder_constant():
    """Test LOOP_PLACEHOLDER constant exists."""
    from runnable.defaults import LOOP_PLACEHOLDER

    assert LOOP_PLACEHOLDER == "loop_variable_placeholder"
    assert isinstance(LOOP_PLACEHOLDER, str)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/runnable/test_defaults.py::test_loop_placeholder_constant -v`
Expected: FAIL with "cannot import name 'LOOP_PLACEHOLDER'"

**Step 3: Write minimal implementation**

Add to `runnable/defaults.py`:

```python
# Add after MAP_PLACEHOLDER
LOOP_PLACEHOLDER = "loop_variable_placeholder"
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/runnable/test_defaults.py::test_loop_placeholder_constant -v`
Expected: PASS

**Step 5: Commit**

```bash
git add runnable/defaults.py tests/runnable/test_defaults.py
git commit -m "feat: add LOOP_PLACEHOLDER constant

Add loop placeholder constant following MAP_PLACEHOLDER pattern
for iteration index replacement in loop node names"
```

## Task 2: Refactor Placeholder Resolution

**Files:**
- Modify: `runnable/nodes.py`
- Test: `tests/runnable/test_nodes.py`

**Step 1: Write the failing test**

```python
def test_resolve_iter_placeholders_map_only():
    """Test resolving map placeholders works as before."""
    from runnable.nodes import BaseNode
    from runnable.defaults import IterableParameterModel, MapVariableModel, MAP_PLACEHOLDER
    from collections import OrderedDict

    iter_var = IterableParameterModel()
    iter_var.map_variable = OrderedDict({
        "chunk": MapVariableModel(value="item_a")
    })

    name = f"step.{MAP_PLACEHOLDER}.task"
    result = BaseNode._resolve_iter_placeholders(name, iter_var)

    assert result == "step.item_a.task"


def test_resolve_iter_placeholders_loop_only():
    """Test resolving loop placeholders."""
    from runnable.nodes import BaseNode
    from runnable.defaults import IterableParameterModel, LoopIndexModel, LOOP_PLACEHOLDER

    iter_var = IterableParameterModel()
    iter_var.loop_variable = [LoopIndexModel(value=2)]

    name = f"loop.{LOOP_PLACEHOLDER}.task"
    result = BaseNode._resolve_iter_placeholders(name, iter_var)

    assert result == "loop.2.task"


def test_resolve_iter_placeholders_nested():
    """Test resolving nested map and loop placeholders."""
    from runnable.nodes import BaseNode
    from runnable.defaults import (
        IterableParameterModel, MapVariableModel, LoopIndexModel,
        MAP_PLACEHOLDER, LOOP_PLACEHOLDER
    )
    from collections import OrderedDict

    iter_var = IterableParameterModel()
    iter_var.map_variable = OrderedDict({
        "chunk": MapVariableModel(value="item_a")
    })
    iter_var.loop_variable = [
        LoopIndexModel(value=1),  # outer loop
        LoopIndexModel(value=3)   # inner loop
    ]

    name = f"map.{MAP_PLACEHOLDER}.loop.{LOOP_PLACEHOLDER}.inner_loop.{LOOP_PLACEHOLDER}.task"
    result = BaseNode._resolve_iter_placeholders(name, iter_var)

    assert result == "map.item_a.loop.1.inner_loop.3.task"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/runnable/test_nodes.py::test_resolve_iter_placeholders_map_only -v`
Expected: FAIL - method doesn't exist

**Step 3: Write minimal implementation**

Replace `_resolve_map_placeholders` method in `runnable/nodes.py`:

```python
@classmethod
def _resolve_iter_placeholders(
    cls,
    name: str,
    iter_variable: Optional[IterableParameterModel] = None,
) -> str:
    """
    Resolve iteration placeholders (map and loop) in node names.

    Replaces MAP_PLACEHOLDER with map variable values and LOOP_PLACEHOLDER
    with loop iteration indices in order.

    Args:
        name: The name containing placeholders
        iter_variable: Iteration variables (map and loop)

    Returns:
        str: Name with placeholders resolved
    """
    if not iter_variable:
        return name

    resolved_name = name

    # Resolve map placeholders
    if iter_variable.map_variable:
        for _, value in iter_variable.map_variable.items():
            resolved_name = resolved_name.replace(
                defaults.MAP_PLACEHOLDER, str(value.value), 1
            )

    # Resolve loop placeholders
    if iter_variable.loop_variable:
        for loop_index in iter_variable.loop_variable:
            resolved_name = resolved_name.replace(
                defaults.LOOP_PLACEHOLDER, str(loop_index.value), 1
            )

    return resolved_name

# Keep old method for backward compatibility
@classmethod
def _resolve_map_placeholders(
    cls,
    name: str,
    iter_variable: Optional[IterableParameterModel] = None,
) -> str:
    """Deprecated: Use _resolve_iter_placeholders instead."""
    return cls._resolve_iter_placeholders(name, iter_variable)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/runnable/test_nodes.py -k "test_resolve_iter_placeholders" -v`
Expected: PASS

**Step 5: Commit**

```bash
git add runnable/nodes.py tests/runnable/test_nodes.py
git commit -m "feat: refactor placeholder resolution for map and loop

- Rename _resolve_map_placeholders to _resolve_iter_placeholders
- Add support for LOOP_PLACEHOLDER resolution
- Handle nested map and loop placeholders correctly
- Keep old method for backward compatibility"
```

## Task 3: Core LoopNode Class

**Files:**
- Create: `extensions/nodes/loop.py`
- Test: `tests/extensions/nodes/test_loop.py`

**Step 1: Write the failing test**

```python
import pytest
from extensions.nodes.loop import LoopNode
from runnable.graph import Graph, create_graph


def test_loop_node_creation():
    """Test basic LoopNode creation and attributes."""
    branch_config = {
        "start_at": "dummy_step",
        "steps": {
            "dummy_step": {
                "type": "success"
            }
        }
    }
    branch = create_graph(branch_config, internal_branch_name="test.branch")

    loop = LoopNode(
        name="test_loop",
        branch=branch,
        max_iterations=5,
        break_on="should_stop",
        index_as="iteration"
    )

    assert loop.name == "test_loop"
    assert loop.node_type == "loop"
    assert loop.max_iterations == 5
    assert loop.break_on == "should_stop"
    assert loop.index_as == "iteration"
    assert loop.branch == branch


def test_loop_node_branch_name_generation():
    """Test loop node generates correct branch names using placeholders."""
    from runnable.defaults import IterableParameterModel, LoopIndexModel, LOOP_PLACEHOLDER

    loop = LoopNode(
        name="test_loop",
        branch=Graph(),  # minimal mock
        max_iterations=3,
        break_on="done",
        index_as="idx"
    )
    loop.internal_name = "test_loop"

    # Should use LOOP_PLACEHOLDER in branch name template
    base_template = f"test_loop.{LOOP_PLACEHOLDER}"

    # Mock iter_variable for iteration 2
    iter_var = IterableParameterModel()
    iter_var.loop_variable = [LoopIndexModel(value=2)]

    result = loop._get_iteration_branch_name(iter_var)
    assert result == "test_loop.2"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/extensions/nodes/test_loop.py::test_loop_node_creation -v`
Expected: FAIL with "No module named 'extensions.nodes.loop'"

**Step 3: Write minimal implementation**

```python
import logging
import os
from typing import Any, Optional

from pydantic import Field, field_validator

from runnable import defaults
from runnable.defaults import IterableParameterModel, LoopIndexModel, LOOP_PLACEHOLDER
from runnable.graph import Graph
from runnable.nodes import CompositeNode

logger = logging.getLogger(defaults.LOGGER_NAME)


class LoopNode(CompositeNode):
    """
    A loop node that iterates over a branch until a break condition is met.

    The branch executes repeatedly until either:
    - parameters[break_on] == True
    - max_iterations is reached

    Each iteration gets its own branch log using LOOP_PLACEHOLDER pattern.
    """

    node_type: str = Field(default="loop", serialization_alias="type")

    # The sub-graph to execute repeatedly
    branch: Graph

    # Maximum iterations (safety limit)
    max_iterations: int

    # Boolean parameter name - when True, loop exits
    break_on: str

    # Environment variable name for iteration index (no prefix)
    index_as: str

    @field_validator("break_on", mode="after")
    @classmethod
    def check_break_on(cls, break_on: str) -> str:
        """Validate that the break_on parameter name is alphanumeric."""
        if not break_on.isalnum():
            raise ValueError(f"Parameter '{break_on}' must be alphanumeric.")
        return break_on

    @field_validator("index_as", mode="after")
    @classmethod
    def check_index_as(cls, index_as: str) -> str:
        """Validate that the index_as variable name is alphanumeric."""
        if not index_as.isalnum():
            raise ValueError(f"Variable '{index_as}' must be alphanumeric.")
        return index_as

    def get_summary(self) -> dict[str, Any]:
        summary = {
            "name": self.name,
            "type": self.node_type,
            "branch": self.branch.get_summary(),
            "max_iterations": self.max_iterations,
            "break_on": self.break_on,
            "index_as": self.index_as,
        }
        return summary

    def _get_iteration_branch_name(
        self,
        iter_variable: Optional[IterableParameterModel] = None
    ) -> str:
        """Get branch name for current iteration using placeholder resolution."""
        # Create branch name template with loop placeholder
        branch_template = f"{self.internal_name}.{LOOP_PLACEHOLDER}"

        # Resolve using the refactored method
        return self._resolve_iter_placeholders(branch_template, iter_variable)

    def fan_out(self, iter_variable: Optional[IterableParameterModel] = None):
        """Create branch log and set up parameters - implementation in next task."""
        pass

    def execute_as_graph(self, iter_variable: Optional[IterableParameterModel] = None):
        """Execute the loop locally - implementation in next task."""
        pass

    def fan_in(
        self,
        iter_variable: Optional[IterableParameterModel] = None
    ) -> bool:
        """Check conditions and return should_exit flag - implementation in next task."""
        return True
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/extensions/nodes/test_loop.py::test_loop_node_creation -v`
Expected: PASS

**Step 5: Commit**

```bash
git add extensions/nodes/loop.py tests/extensions/nodes/test_loop.py
git commit -m "feat: add basic LoopNode class with placeholder support

- Create LoopNode class following composite node patterns
- Use LOOP_PLACEHOLDER for branch name generation
- Add validation for break_on and index_as parameters
- Implement _get_iteration_branch_name using refactored placeholder resolution"
```

## Task 4: Loop Node Parameter and Branch Management

**Files:**
- Modify: `extensions/nodes/loop.py`
- Test: `tests/extensions/nodes/test_loop.py`

**Step 1: Write the failing test**

```python
def test_get_break_condition_value():
    """Test reading break condition from parameters."""
    from runnable.datastore import Parameter
    from unittest.mock import Mock

    # Mock context and run_log_store
    mock_context = Mock()
    mock_run_log_store = Mock()
    mock_context.run_log_store = mock_run_log_store
    mock_context.run_id = "test-run-123"

    # Set up parameters
    parameters = {
        "should_stop": Parameter.create_unknown("should_stop", False, "json")
    }
    mock_run_log_store.get_parameters.return_value = parameters

    loop = LoopNode(
        name="test_loop",
        branch=Mock(),
        max_iterations=5,
        break_on="should_stop",
        index_as="iteration"
    )
    loop._context = mock_context

    result = loop.get_break_condition_value()
    assert result is False
    mock_run_log_store.get_parameters.assert_called_with(run_id="test-run-123")


def test_create_iteration_branch_log():
    """Test creating branch logs with proper iteration naming."""
    from runnable.defaults import IterableParameterModel, LoopIndexModel
    from unittest.mock import Mock

    mock_context = Mock()
    mock_run_log_store = Mock()
    mock_context.run_log_store = mock_run_log_store
    mock_context.run_id = "test-run-456"

    # Mock branch log creation
    mock_branch_log = Mock()
    mock_run_log_store.create_branch_log.return_value = mock_branch_log

    loop = LoopNode(
        name="test_loop",
        branch=Mock(),
        max_iterations=3,
        break_on="done",
        index_as="idx"
    )
    loop._context = mock_context
    loop.internal_name = "test_loop"

    # Create iter_variable for iteration 1
    iter_var = IterableParameterModel()
    iter_var.loop_variable = [LoopIndexModel(value=1)]

    loop._create_iteration_branch_log(iter_var)

    # Should create branch log with resolved name
    expected_name = "test_loop.1"
    mock_run_log_store.create_branch_log.assert_called_with(expected_name)
    mock_run_log_store.add_branch_log.assert_called_with(mock_branch_log, "test-run-456")
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/extensions/nodes/test_loop.py::test_get_break_condition_value -v`
Expected: FAIL - methods don't exist

**Step 3: Write minimal implementation**

Add to `LoopNode` class in `extensions/nodes/loop.py`:

```python
from runnable.datastore import Parameter

def get_break_condition_value(self) -> bool:
    """Get the break condition parameter value."""
    parameters: dict[str, Parameter] = self._context.run_log_store.get_parameters(
        run_id=self._context.run_id
    )

    if self.break_on not in parameters:
        return False  # Default to continue if parameter doesn't exist

    condition_value = parameters[self.break_on].get_value()

    if not isinstance(condition_value, bool):
        raise ValueError(
            f"Break condition '{self.break_on}' must be boolean, "
            f"got {type(condition_value).__name__}"
        )

    return condition_value

def _create_iteration_branch_log(
    self,
    iter_variable: Optional[IterableParameterModel] = None
):
    """Create branch log for the current iteration."""
    branch_name = self._get_iteration_branch_name(iter_variable)

    try:
        branch_log = self._context.run_log_store.get_branch_log(
            branch_name, self._context.run_id
        )
        logger.debug(f"Branch log already exists for {branch_name}")
    except Exception:  # BranchLogNotFoundError
        branch_log = self._context.run_log_store.create_branch_log(branch_name)
        logger.debug(f"Branch log created for {branch_name}")

    branch_log.status = defaults.PROCESSING
    self._context.run_log_store.add_branch_log(branch_log, self._context.run_id)
    return branch_log

def _build_iteration_iter_variable(
    self,
    parent_iter_variable: Optional[IterableParameterModel],
    iteration: int
) -> IterableParameterModel:
    """Build iter_variable for current iteration."""
    if parent_iter_variable:
        iter_var = parent_iter_variable.model_copy(deep=True)
    else:
        iter_var = IterableParameterModel()

    # Initialize loop_variable if None
    if iter_var.loop_variable is None:
        iter_var.loop_variable = []

    # Add current iteration index
    iter_var.loop_variable.append(LoopIndexModel(value=iteration))

    return iter_var
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/extensions/nodes/test_loop.py -k "test_get_break_condition_value or test_create_iteration_branch_log" -v`
Expected: PASS

**Step 5: Commit**

```bash
git add extensions/nodes/loop.py tests/extensions/nodes/test_loop.py
git commit -m "feat: add loop node parameter and branch log management

- Add get_break_condition_value() method with proper validation
- Add _create_iteration_branch_log() using placeholder resolution
- Add _build_iteration_iter_variable() for iteration context
- Proper error handling for missing/invalid parameters"
```

## Task 5: Fan-out Implementation

**Files:**
- Modify: `extensions/nodes/loop.py`
- Test: `tests/extensions/nodes/test_loop.py`

**Step 1: Write the failing test**

```python
def test_fan_out_initial_iteration():
    """Test fan_out creates branch log and copies parent parameters."""
    from unittest.mock import Mock, patch

    mock_context = Mock()
    mock_run_log_store = Mock()
    mock_context.run_log_store = mock_run_log_store
    mock_context.run_id = "test-run-123"

    # Mock parent parameters
    parent_params = {"param1": Mock()}
    mock_run_log_store.get_parameters.return_value = parent_params

    loop = LoopNode(
        name="test_loop",
        branch=Mock(),
        max_iterations=5,
        break_on="should_stop",
        index_as="iteration"
    )
    loop._context = mock_context
    loop.internal_name = "test_loop"
    loop.internal_branch_name = "root"

    # Create iter_variable for iteration 0
    from runnable.defaults import IterableParameterModel, LoopIndexModel
    iter_var = IterableParameterModel()
    iter_var.loop_variable = [LoopIndexModel(value=0)]

    with patch.object(loop, '_create_iteration_branch_log') as mock_create_branch:
        loop.fan_out(iter_var)

    # Should create branch log
    mock_create_branch.assert_called_once_with(iter_var)

    # Should copy parent parameters to iteration branch
    mock_run_log_store.set_parameters.assert_called_once_with(
        parameters=parent_params,
        run_id="test-run-123",
        internal_branch_name="test_loop.0"
    )


def test_fan_out_subsequent_iteration():
    """Test fan_out copies from previous iteration."""
    from unittest.mock import Mock, patch
    from runnable.defaults import IterableParameterModel, LoopIndexModel

    mock_context = Mock()
    mock_run_log_store = Mock()
    mock_context.run_log_store = mock_run_log_store
    mock_context.run_id = "test-run-123"

    # Mock previous iteration parameters
    prev_params = {"param1": Mock(), "result": Mock()}
    mock_run_log_store.get_parameters.return_value = prev_params

    loop = LoopNode(
        name="test_loop",
        branch=Mock(),
        max_iterations=5,
        break_on="should_stop",
        index_as="iteration"
    )
    loop._context = mock_context
    loop.internal_name = "test_loop"

    # Create iter_variable for iteration 2
    iter_var = IterableParameterModel()
    iter_var.loop_variable = [LoopIndexModel(value=2)]

    with patch.object(loop, '_create_iteration_branch_log') as mock_create_branch:
        loop.fan_out(iter_var)

    # Should get parameters from iteration 1
    prev_iter_var = IterableParameterModel()
    prev_iter_var.loop_variable = [LoopIndexModel(value=1)]
    expected_prev_name = "test_loop.1"

    mock_run_log_store.get_parameters.assert_called_with(
        run_id="test-run-123",
        internal_branch_name=expected_prev_name
    )

    # Should copy to iteration 2 branch
    mock_run_log_store.set_parameters.assert_called_once_with(
        parameters=prev_params,
        run_id="test-run-123",
        internal_branch_name="test_loop.2"
    )
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/extensions/nodes/test_loop.py::test_fan_out_initial_iteration -v`
Expected: FAIL - fan_out not implemented

**Step 3: Write minimal implementation**

Replace the `fan_out` method in `extensions/nodes/loop.py`:

```python
def fan_out(
    self,
    iter_variable: Optional[IterableParameterModel] = None
):
    """
    Create branch log for current iteration and copy parameters.

    For iteration 0: copy from parent scope
    For iteration N: copy from previous iteration (N-1) scope
    """
    # Create branch log for current iteration
    self._create_iteration_branch_log(iter_variable)

    # Determine current iteration from iter_variable
    current_iteration = 0
    if iter_variable and iter_variable.loop_variable:
        current_iteration = iter_variable.loop_variable[-1].value

    # Determine source of parameters
    if current_iteration == 0:
        # Copy from parent scope
        source_branch_name = self.internal_branch_name
    else:
        # Copy from previous iteration
        prev_iter_var = iter_variable.model_copy(deep=True) if iter_variable else IterableParameterModel()
        if prev_iter_var.loop_variable is None:
            prev_iter_var.loop_variable = []
        # Replace last loop index with previous iteration
        prev_iter_var.loop_variable[-1] = LoopIndexModel(value=current_iteration - 1)
        source_branch_name = self._get_iteration_branch_name(prev_iter_var)

    # Get source parameters
    source_params = self._context.run_log_store.get_parameters(
        run_id=self._context.run_id,
        internal_branch_name=source_branch_name
    )

    # Copy to current iteration branch
    target_branch_name = self._get_iteration_branch_name(iter_variable)
    self._context.run_log_store.set_parameters(
        parameters=source_params,
        run_id=self._context.run_id,
        internal_branch_name=target_branch_name
    )
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/extensions/nodes/test_loop.py -k "test_fan_out" -v`
Expected: PASS

**Step 5: Commit**

```bash
git add extensions/nodes/loop.py tests/extensions/nodes/test_loop.py
git commit -m "feat: implement LoopNode fan_out method

- Create branch logs for current iteration
- Copy parameters from parent (iteration 0) or previous iteration
- Use proper placeholder resolution for branch names
- Support both initial and subsequent iterations"
```

## Task 6: Fan-in Implementation

**Files:**
- Modify: `extensions/nodes/loop.py`
- Test: `tests/extensions/nodes/test_loop.py`

**Step 1: Write the failing test**

```python
def test_fan_in_should_continue():
    """Test fan_in returns False when break condition not met."""
    from unittest.mock import Mock
    from runnable.datastore import Parameter
    from runnable.defaults import IterableParameterModel, LoopIndexModel

    mock_context = Mock()
    mock_run_log_store = Mock()
    mock_context.run_log_store = mock_run_log_store
    mock_context.run_id = "test-run-123"

    # Break condition is False, should continue
    parameters = {
        "should_stop": Parameter.create_unknown("should_stop", False, "json")
    }
    mock_run_log_store.get_parameters.return_value = parameters

    loop = LoopNode(
        name="test_loop",
        branch=Mock(),
        max_iterations=5,
        break_on="should_stop",
        index_as="iteration"
    )
    loop._context = mock_context

    # Iteration 1 (0-indexed), not at max yet
    iter_var = IterableParameterModel()
    iter_var.loop_variable = [LoopIndexModel(value=1)]

    should_exit = loop.fan_in(iter_var)

    assert should_exit is False  # Should continue looping


def test_fan_in_should_exit_break_condition():
    """Test fan_in returns True when break condition is met."""
    from unittest.mock import Mock, patch
    from runnable.datastore import Parameter
    from runnable.defaults import IterableParameterModel, LoopIndexModel

    mock_context = Mock()
    mock_run_log_store = Mock()
    mock_context.run_log_store = mock_run_log_store
    mock_context.run_id = "test-run-123"

    # Break condition is True, should exit
    parameters = {
        "should_stop": Parameter.create_unknown("should_stop", True, "json")
    }
    mock_run_log_store.get_parameters.return_value = parameters

    loop = LoopNode(
        name="test_loop",
        branch=Mock(),
        max_iterations=5,
        break_on="should_stop",
        index_as="iteration"
    )
    loop._context = mock_context

    iter_var = IterableParameterModel()
    iter_var.loop_variable = [LoopIndexModel(value=2)]

    with patch.object(loop, '_rollback_parameters_to_parent') as mock_rollback, \
         patch.object(loop, '_set_final_step_status') as mock_set_status:

        should_exit = loop.fan_in(iter_var)

        assert should_exit is True
        mock_rollback.assert_called_once_with(iter_var)
        mock_set_status.assert_called_once_with(iter_var)


def test_fan_in_should_exit_max_iterations():
    """Test fan_in returns True when max iterations reached."""
    from unittest.mock import Mock, patch
    from runnable.datastore import Parameter
    from runnable.defaults import IterableParameterModel, LoopIndexModel

    mock_context = Mock()
    mock_run_log_store = Mock()
    mock_context.run_log_store = mock_run_log_store
    mock_context.run_id = "test-run-123"

    # Break condition is False but max iterations reached
    parameters = {
        "should_stop": Parameter.create_unknown("should_stop", False, "json")
    }
    mock_run_log_store.get_parameters.return_value = parameters

    loop = LoopNode(
        name="test_loop",
        branch=Mock(),
        max_iterations=3,  # 0, 1, 2 (3 iterations)
        break_on="should_stop",
        index_as="iteration"
    )
    loop._context = mock_context

    # Iteration 2 (0-indexed) = 3rd iteration = max reached
    iter_var = IterableParameterModel()
    iter_var.loop_variable = [LoopIndexModel(value=2)]

    with patch.object(loop, '_rollback_parameters_to_parent'), \
         patch.object(loop, '_set_final_step_status'):

        should_exit = loop.fan_in(iter_var)

        assert should_exit is True  # Should exit due to max iterations
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/extensions/nodes/test_loop.py -k "test_fan_in_should" -v`
Expected: FAIL - fan_in not implemented properly

**Step 3: Write minimal implementation**

Replace the `fan_in` method in `extensions/nodes/loop.py`:

```python
def fan_in(
    self,
    iter_variable: Optional[IterableParameterModel] = None
) -> bool:
    """
    Check termination conditions and handle loop completion.

    Returns:
        bool: True if loop should exit, False if should continue
    """
    # Get current iteration from iter_variable
    current_iteration = 0
    if iter_variable and iter_variable.loop_variable:
        current_iteration = iter_variable.loop_variable[-1].value

    # Check break condition
    break_condition_met = False
    try:
        break_condition_met = self.get_break_condition_value()
    except (KeyError, ValueError):
        # If break parameter doesn't exist or invalid, continue
        break_condition_met = False

    # Check max iterations (0-indexed, so iteration N means N+1 total iterations)
    max_iterations_reached = current_iteration >= (self.max_iterations - 1)

    should_exit = break_condition_met or max_iterations_reached

    if should_exit:
        # Roll back parameters to parent and set status on exit
        self._rollback_parameters_to_parent(iter_variable)
        self._set_final_step_status(iter_variable)

    return should_exit

def _rollback_parameters_to_parent(
    self,
    iter_variable: Optional[IterableParameterModel] = None
):
    """Copy parameters from current iteration back to parent scope."""
    current_branch_name = self._get_iteration_branch_name(iter_variable)

    current_params = self._context.run_log_store.get_parameters(
        run_id=self._context.run_id,
        internal_branch_name=current_branch_name
    )

    # Copy back to parent
    self._context.run_log_store.set_parameters(
        parameters=current_params,
        run_id=self._context.run_id,
        internal_branch_name=self.internal_branch_name
    )

def _set_final_step_status(
    self,
    iter_variable: Optional[IterableParameterModel] = None
):
    """Set the loop node's final status based on branch execution."""
    effective_internal_name = self._resolve_iter_placeholders(
        self.internal_name, iter_variable=iter_variable
    )

    step_log = self._context.run_log_store.get_step_log(
        effective_internal_name, self._context.run_id
    )

    # Check current iteration branch status
    current_branch_name = self._get_iteration_branch_name(iter_variable)
    try:
        current_branch_log = self._context.run_log_store.get_branch_log(
            current_branch_name, self._context.run_id
        )

        if current_branch_log.status == defaults.SUCCESS:
            step_log.status = defaults.SUCCESS
        else:
            step_log.status = defaults.FAIL

    except Exception:
        # If branch log not found, mark as failed
        step_log.status = defaults.FAIL

    self._context.run_log_store.add_step_log(step_log, self._context.run_id)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/extensions/nodes/test_loop.py -k "test_fan_in" -v`
Expected: PASS

**Step 5: Commit**

```bash
git add extensions/nodes/loop.py tests/extensions/nodes/test_loop.py
git commit -m "feat: implement LoopNode fan_in method

- Check break condition and max iterations correctly
- Roll back parameters to parent on loop exit
- Set final step status based on branch execution
- Extract current iteration from iter_variable properly"
```

## Task 7: Local Loop Execution

**Files:**
- Modify: `extensions/nodes/loop.py`
- Test: `tests/extensions/nodes/test_loop.py`

**Step 1: Write the failing test**

```python
def test_execute_as_graph_single_iteration():
    """Test loop executes once then exits on break condition."""
    from unittest.mock import Mock, patch
    import os

    mock_context = Mock()
    mock_pipeline_executor = Mock()
    mock_context.pipeline_executor = mock_pipeline_executor

    branch = Mock()

    loop = LoopNode(
        name="test_loop",
        branch=branch,
        max_iterations=5,
        break_on="should_stop",
        index_as="iteration"
    )
    loop._context = mock_context

    with patch.object(loop, 'fan_out') as mock_fan_out, \
         patch.object(loop, 'fan_in') as mock_fan_in, \
         patch.dict(os.environ, {}, clear=True):

        # First iteration should exit
        mock_fan_in.return_value = True

        loop.execute_as_graph()

        # Should call fan_out for iteration 0
        mock_fan_out.assert_called_once()

        # Should execute branch once
        mock_pipeline_executor.execute_graph.assert_called_once()

        # Should call fan_in for iteration 0
        mock_fan_in.assert_called_once()

        # Should set environment variable
        assert os.environ.get("iteration") == "0"


def test_execute_as_graph_multiple_iterations():
    """Test loop executes multiple times before break condition."""
    from unittest.mock import Mock, patch, call
    import os

    mock_context = Mock()
    mock_pipeline_executor = Mock()
    mock_context.pipeline_executor = mock_pipeline_executor

    branch = Mock()

    loop = LoopNode(
        name="test_loop",
        branch=branch,
        max_iterations=5,
        break_on="should_stop",
        index_as="attempt"
    )
    loop._context = mock_context

    with patch.object(loop, 'fan_out') as mock_fan_out, \
         patch.object(loop, 'fan_in') as mock_fan_in, \
         patch.dict(os.environ, {}, clear=True):

        # Return False twice, then True (3 iterations total)
        mock_fan_in.side_effect = [False, False, True]

        loop.execute_as_graph()

        # Should call fan_out 3 times
        assert mock_fan_out.call_count == 3

        # Should execute branch 3 times
        assert mock_pipeline_executor.execute_graph.call_count == 3

        # Should call fan_in 3 times
        assert mock_fan_in.call_count == 3

        # Final env var should be "2"
        assert os.environ.get("attempt") == "2"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/extensions/nodes/test_loop.py::test_execute_as_graph_single_iteration -v`
Expected: FAIL - execute_as_graph not implemented

**Step 3: Write minimal implementation**

Replace the `execute_as_graph` method in `extensions/nodes/loop.py`:

```python
def execute_as_graph(
    self,
    iter_variable: Optional[IterableParameterModel] = None
):
    """Execute the loop locally until break condition or max iterations."""
    iteration = 0

    while True:
        # Set iteration index environment variable
        os.environ[self.index_as] = str(iteration)

        # Build iter_variable for current iteration
        current_iter_variable = self._build_iteration_iter_variable(
            iter_variable, iteration
        )

        # Set up branch log and parameters for this iteration
        self.fan_out(current_iter_variable)

        # Execute the branch
        self._context.pipeline_executor.execute_graph(
            self.branch,
            iter_variable=current_iter_variable
        )

        # Check if we should exit
        should_exit = self.fan_in(current_iter_variable)

        if should_exit:
            break

        iteration += 1
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/extensions/nodes/test_loop.py -k "test_execute_as_graph" -v`
Expected: PASS

**Step 5: Commit**

```bash
git add extensions/nodes/loop.py tests/extensions/nodes/test_loop.py
git commit -m "feat: implement LoopNode local execution

- Add execute_as_graph method for local loop execution
- Set iteration index as environment variable
- Build proper iter_variable with loop index for each iteration
- Support both single and multiple iteration scenarios"
```

## Task 8: SDK Interface and Entry Point

**Files:**
- Modify: `runnable/sdk.py`
- Modify: `runnable/__init__.py`
- Modify: `pyproject.toml`
- Test: `tests/runnable/test_sdk.py`

**Step 1: Write the failing test**

Add to `tests/runnable/test_sdk.py`:

```python
def test_loop_sdk_interface():
    """Test Loop class creates LoopNode correctly."""
    from runnable import Pipeline, PythonTask, Loop
    from extensions.nodes.loop import LoopNode
    from runnable.returns import json

    def dummy_func():
        return {"result": True, "should_stop": True}

    task = PythonTask(
        name="dummy_task",
        function=dummy_func,
        returns=[json("result"), json("should_stop")]
    )

    branch_pipeline = Pipeline(steps=[task])

    loop = Loop(
        name="test_loop",
        branch=branch_pipeline,
        max_iterations=3,
        break_on="should_stop",
        index_as="iteration_num"
    )

    # Should create LoopNode
    assert isinstance(loop, LoopNode)
    assert loop.name == "test_loop"
    assert loop.max_iterations == 3
    assert loop.break_on == "should_stop"
    assert loop.index_as == "iteration_num"
    assert loop.branch == branch_pipeline._get_dag()
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/runnable/test_sdk.py::test_loop_sdk_interface -v`
Expected: FAIL with "cannot import name 'Loop'"

**Step 3: Write minimal implementation**

Add to `runnable/sdk.py`:

```python
# Add import at top with other node imports
from extensions.nodes.loop import LoopNode

# Add Loop class after other composite node classes
class Loop(LoopNode):
    """
    SDK interface for creating loop nodes.

    A loop node iterates over a branch pipeline until a break condition
    is met or max iterations is reached.

    Args:
        name: Name of the loop node
        branch: Pipeline to execute repeatedly
        max_iterations: Maximum number of iterations (safety limit)
        break_on: Name of boolean parameter that triggers loop exit when True
        index_as: Name of environment variable for iteration index (0, 1, 2...)
    """

    def __init__(
        self,
        name: str,
        branch: "Pipeline",
        max_iterations: int,
        break_on: str,
        index_as: str,
        **kwargs
    ):
        # Convert Pipeline to Graph
        branch_dag = branch._get_dag()

        super().__init__(
            name=name,
            branch=branch_dag,
            max_iterations=max_iterations,
            break_on=break_on,
            index_as=index_as,
            **kwargs
        )
```

**Step 4: Update exports**

Add to `runnable/__init__.py`:

```python
from runnable.sdk import Loop  # Add this import
```

**Step 5: Register entry point**

Add to the `[project.entry-points."nodes"]` section in `pyproject.toml`:

```toml
loop = "extensions.nodes.loop:LoopNode"
```

**Step 6: Run test to verify it passes**

Run: `uv run pytest tests/runnable/test_sdk.py::test_loop_sdk_interface -v`
Expected: PASS

**Step 7: Commit**

```bash
git add runnable/sdk.py runnable/__init__.py pyproject.toml tests/runnable/test_sdk.py
git commit -m "feat: add Loop SDK interface and entry point

- Add Loop class to sdk.py for user-friendly loop creation
- Register loop node entry point in pyproject.toml
- Export Loop from runnable package
- Add SDK integration test with proper imports"
```

## Task 9: Integration Tests and Examples

**Files:**
- Create: `tests/extensions/nodes/test_loop_integration.py`
- Create: `examples/05-loops/simple_retry_loop.py`

**Step 1: Create integration test**

```python
"""Integration tests for loop node functionality."""

import pytest
import os
from unittest.mock import Mock, patch

def test_loop_integration_with_parameters():
    """Test loop node with actual parameter flow."""
    from runnable import Pipeline, PythonTask, Loop
    from runnable.returns import json
    from runnable.datastore import Parameter

    call_count = 0

    def increment_and_check():
        nonlocal call_count
        call_count += 1
        iteration = int(os.environ.get("loop_idx", "0"))

        return {
            "count": call_count,
            "iteration": iteration,
            "should_stop": call_count >= 3
        }

    task = PythonTask(
        name="counter",
        function=increment_and_check,
        returns=[json("count"), json("iteration"), json("should_stop")]
    )

    branch = Pipeline(steps=[task])

    loop = Loop(
        name="counter_loop",
        branch=branch,
        max_iterations=10,  # High limit, should exit on condition
        break_on="should_stop",
        index_as="loop_idx"
    )

    # Test the iteration iter_variable building
    from runnable.defaults import IterableParameterModel, LoopIndexModel

    iter_var = loop._build_iteration_iter_variable(None, 2)

    assert iter_var.loop_variable is not None
    assert len(iter_var.loop_variable) == 1
    assert iter_var.loop_variable[0].value == 2


def test_loop_branch_naming():
    """Test branch names are generated correctly."""
    from extensions.nodes.loop import LoopNode
    from runnable.defaults import IterableParameterModel, LoopIndexModel
    from runnable.graph import Graph

    loop = LoopNode(
        name="test_loop",
        branch=Graph(),
        max_iterations=3,
        break_on="done",
        index_as="idx"
    )
    loop.internal_name = "parent.test_loop"

    # Test iteration 0
    iter_var_0 = IterableParameterModel()
    iter_var_0.loop_variable = [LoopIndexModel(value=0)]

    name_0 = loop._get_iteration_branch_name(iter_var_0)
    assert name_0 == "parent.test_loop.0"

    # Test iteration 5
    iter_var_5 = IterableParameterModel()
    iter_var_5.loop_variable = [LoopIndexModel(value=5)]

    name_5 = loop._get_iteration_branch_name(iter_var_5)
    assert name_5 == "parent.test_loop.5"


def test_loop_max_iterations():
    """Test loop respects max_iterations limit."""
    from extensions.nodes.loop import LoopNode
    from runnable.defaults import IterableParameterModel, LoopIndexModel
    from runnable.graph import Graph
    from unittest.mock import Mock

    mock_context = Mock()
    mock_run_log_store = Mock()
    mock_context.run_log_store = mock_run_log_store
    mock_context.run_id = "test"

    # Break condition never met
    from runnable.datastore import Parameter
    parameters = {
        "never_stop": Parameter.create_unknown("never_stop", False, "json")
    }
    mock_run_log_store.get_parameters.return_value = parameters

    loop = LoopNode(
        name="limited_loop",
        branch=Graph(),
        max_iterations=2,  # Should stop after 2 iterations (0, 1)
        break_on="never_stop",
        index_as="iteration"
    )
    loop._context = mock_context

    # Test iteration 1 (should continue)
    iter_var_1 = IterableParameterModel()
    iter_var_1.loop_variable = [LoopIndexModel(value=1)]

    should_exit_1 = loop.fan_in(iter_var_1)
    assert should_exit_1 is True  # max_iterations=2 means 0,1 are allowed; 1 is last

    # Test iteration 0 (should continue)
    iter_var_0 = IterableParameterModel()
    iter_var_0.loop_variable = [LoopIndexModel(value=0)]

    with patch.object(loop, '_rollback_parameters_to_parent'), \
         patch.object(loop, '_set_final_step_status'):
        should_exit_0 = loop.fan_in(iter_var_0)

    assert should_exit_0 is False  # Should continue to iteration 1
```

**Step 2: Create example**

```python
"""Simple retry loop example demonstrating loop node usage."""

import os
import random
from runnable import Pipeline, PythonTask, Loop
from runnable.returns import json


def unreliable_operation():
    """Simulates an operation that might fail."""
    attempt = int(os.environ.get("attempt", "0"))
    print(f"Attempt {attempt}: Calling unreliable service...")

    # Simulate 70% failure rate
    if random.random() < 0.7:
        print(f"Attempt {attempt}: Failed!")
        return {
            "success": False,
            "should_stop": False  # Continue trying
        }
    else:
        print(f"Attempt {attempt}: Success!")
        return {
            "success": True,
            "should_stop": True  # Stop on success
        }


if __name__ == "__main__":
    # Create the retry task
    retry_task = PythonTask(
        name="call_service",
        function=unreliable_operation,
        returns=[json("success"), json("should_stop")]
    )

    # Wrap in a pipeline (required for loop branches)
    retry_branch = Pipeline(steps=[retry_task])

    # Create retry loop
    retry_loop = Loop(
        name="service_retry",
        branch=retry_branch,
        max_iterations=5,  # Try up to 5 times
        break_on="should_stop",  # Stop when this parameter is True
        index_as="attempt"  # Available as 'attempt' env var (0, 1, 2...)
    )

    # Create main pipeline
    pipeline = Pipeline(steps=[retry_loop])

    print("Starting retry loop...")
    print("This will try up to 5 times or until success")
    print()

    # Execute
    try:
        result = pipeline.execute()
        print(f"\nPipeline completed successfully!")
        print(f"Final result: {result}")
    except Exception as e:
        print(f"\nPipeline failed: {e}")
```

**Step 3: Run tests**

Run: `uv run pytest tests/extensions/nodes/test_loop_integration.py -v`
Expected: PASS

**Step 4: Test example (optional)**

```bash
mkdir -p examples/05-loops
# Then run: uv run examples/05-loops/simple_retry_loop.py
```

**Step 5: Commit**

```bash
git add tests/extensions/nodes/test_loop_integration.py examples/05-loops/simple_retry_loop.py
git commit -m "feat: add loop node integration tests and example

- Integration tests for parameter flow and branch naming
- Test max_iterations termination correctly
- Simple retry example demonstrating loop usage
- Tests cover iter_variable building and placeholder resolution"
```

## Task 10: Final Testing

**Files:**
- Run comprehensive tests

**Step 1: Run all loop tests**

```bash
uv run pytest tests/extensions/nodes/test_loop.py -v
uv run pytest tests/extensions/nodes/test_loop_integration.py -v
uv run pytest tests/runnable/test_sdk.py -k loop -v
uv run pytest tests/runnable/test_nodes.py -k "iter_placeholder" -v
```

**Step 2: Run full test suite to ensure no regressions**

```bash
uv run pytest tests/ -x
```

**Step 3: Fix any failing tests**

Fix issues as they arise, committing each fix separately.

**Step 4: Final validation commit**

```bash
git add .
git commit -m "feat: finalize loop node local implementation

- All tests passing
- Proper placeholder resolution for map and loop variables
- Local execution fully working
- Example validated
- Ready for use in local pipelines"
```

---

This plan focuses only on local execution and gets the placeholder resolution pattern correct. The key insight is refactoring `_resolve_map_placeholders` to `_resolve_iter_placeholders` to handle both map and loop variables properly, following the existing patterns in the codebase.
