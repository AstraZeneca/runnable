# Attempt Number Cleanup Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Clean up attempt number logic by extracting calculation into proper methods and removing deprecated step_attempt_number property

**Architecture:** Replace the property-based approach with explicit attempt number calculation based on existing step logs in the run log store

**Tech Stack:** Python, pytest for testing

---

### Task 1: Extract Attempt Number Calculation Method

**Files:**
- Modify: `extensions/pipeline_executor/__init__.py:224-238`
- Test: `tests/extensions/test_pipeline_executor.py`

**Step 1: Write the failing test**

```python
def test_calculate_attempt_number_first_attempt(mock_context, generic_executor):
    """Test calculating attempt number for first execution"""
    from runnable import exceptions

    mock_context.run_log_store.get_step_log.side_effect = exceptions.StepLogNotFoundError("Not found")
    node = MagicMock()
    node._get_step_log_name.return_value = "test_step"

    attempt_num = generic_executor._calculate_attempt_number(node, None)

    assert attempt_num == 1

def test_calculate_attempt_number_with_existing_attempts(mock_context, generic_executor):
    """Test calculating attempt number when previous attempts exist"""
    from runnable.datastore import StepLog, StepAttempt

    # Mock existing step log with 2 attempts
    mock_step_log = StepLog(
        name="test_step",
        internal_name="test_step",
        attempts=[
            StepAttempt(attempt_number=1, status="FAILED"),
            StepAttempt(attempt_number=2, status="FAILED")
        ]
    )
    mock_context.run_log_store.get_step_log.return_value = mock_step_log

    node = MagicMock()
    node._get_step_log_name.return_value = "test_step"

    attempt_num = generic_executor._calculate_attempt_number(node, None)

    assert attempt_num == 3
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/extensions/test_pipeline_executor.py::test_calculate_attempt_number_first_attempt -v`
Expected: FAIL with "AttributeError: '_calculate_attempt_number' method not found"

**Step 3: Implement the minimal method**

Add to `GenericPipelineExecutor` class in `extensions/pipeline_executor/__init__.py`:

```python
def _calculate_attempt_number(self, node: BaseNode, map_variable: dict = None) -> int:
    """
    Calculate the attempt number for a node based on existing attempts in the run log.

    Args:
        node: The node to calculate attempt number for
        map_variable: Optional map variable if node is in a map state

    Returns:
        int: The attempt number (starting from 1)
    """
    step_log_name = node._get_step_log_name(map_variable)

    try:
        existing_step_log = self._context.run_log_store.get_step_log(
            step_log_name, self._context.run_id
        )
        # If step log exists, increment attempt number based on existing attempts
        return len(existing_step_log.attempts) + 1
    except exceptions.StepLogNotFoundError:
        # This is the first attempt, use attempt number 1
        return 1
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/extensions/test_pipeline_executor.py::test_calculate_attempt_number_first_attempt -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/extensions/test_pipeline_executor.py extensions/pipeline_executor/__init__.py
git commit -m "feat: add _calculate_attempt_number method to pipeline executor"
```

### Task 2: Update execute_node Method to Use New Logic

**Files:**
- Modify: `extensions/pipeline_executor/__init__.py:224-246`

**Step 1: Write test for execute_node using new logic**

Add to existing test file:

```python
def test_execute_node_uses_calculated_attempt_number(mock_context, generic_executor):
    """Test that execute_node uses the calculated attempt number"""
    import os
    from runnable import defaults

    node = MagicMock()
    node._get_step_log_name.return_value = "test_step"
    node.execute.return_value = MagicMock()

    # Mock the attempt calculation method
    generic_executor._calculate_attempt_number = MagicMock(return_value=3)

    generic_executor.execute_node(node)

    # Verify attempt number was calculated and used
    generic_executor._calculate_attempt_number.assert_called_once_with(node, None)
    node.execute.assert_called_once_with(
        map_variable=None,
        attempt_number=3,
        mock=False
    )
    # Verify environment variable was set
    assert os.environ[defaults.ATTEMPT_NUMBER] == "3"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/extensions/test_pipeline_executor.py::test_execute_node_uses_calculated_attempt_number -v`
Expected: FAIL with assertion error on environment variable or method calls

**Step 3: Update execute_node method**

Replace the inline logic (lines 224-246) in `execute_node` method:

```python
def execute_node(self, node: BaseNode, map_variable: dict = None, mock: bool = False):
    """
    Executes a given node.

    Args:
        node (Node): The node to execute
        map_variable (dict, optional): If the node is of a map state, map_variable is the value of the iterable.
                    Defaults to None.
    """
    # Calculate attempt number based on existing attempts in run log
    current_attempt_number = self._calculate_attempt_number(node, map_variable)

    # Set the environment variable for this attempt
    os.environ[defaults.ATTEMPT_NUMBER] = str(current_attempt_number)

    logger.info(
        f"Trying to execute node: {node.internal_name}, attempt : {current_attempt_number}"
    )

    self._context_node = node
    self._context_attempt_number = current_attempt_number

    # ... rest of method unchanged ...

    step_log = node.execute(
        map_variable=map_variable,
        attempt_number=current_attempt_number,
        mock=mock,
    )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/extensions/test_pipeline_executor.py::test_execute_node_uses_calculated_attempt_number -v`
Expected: PASS

**Step 5: Commit**

```bash
git add extensions/pipeline_executor/__init__.py
git commit -m "refactor: use extracted attempt calculation logic in execute_node"
```

### Task 3: Remove step_attempt_number Property from BaseExecutor

**Files:**
- Modify: `runnable/executor.py:67-85`
- Test: `tests/runnable/test_executor.py`

**Step 1: Check current usage of the property**

Run: `grep -r "step_attempt_number" --include="*.py" .`
Expected: Find all references to remove or update

**Step 2: Write test for property removal**

Add to test file (if not already covered):

```python
def test_base_executor_no_step_attempt_number():
    """Test that BaseExecutor no longer has step_attempt_number property"""
    from runnable.executor import BaseExecutor

    # This should not exist anymore
    assert not hasattr(BaseExecutor, 'step_attempt_number')
```

**Step 3: Run test to verify current state**

Run: `pytest tests/runnable/test_executor.py::test_base_executor_no_step_attempt_number -v`
Expected: FAIL because property still exists

**Step 4: Remove the property and TODO**

Remove lines 67-85 from `runnable/executor.py`:

```python
# Remove this entire section:
# TODO: we might have to remove this in favor of a correct implementation in the pipeline executor
@property
def step_attempt_number(self) -> int:
    """
    The attempt number of the current step.
    Orchestrators should use this step to submit multiple attempts of the job.

    Returns:
        int: The attempt number of the current step.
    """
    run_id = BaseExecutor._get_run_id_from_environment()

    # TODO: Maybe this should be part of another class so it is mockable
    runnable_object = FailureMode()

    return runnable_object.step_attempt_number or 1
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/runnable/test_executor.py::test_base_executor_no_step_attempt_number -v`
Expected: PASS

**Step 6: Commit**

```bash
git add runnable/executor.py tests/runnable/test_executor.py
git commit -m "refactor: remove deprecated step_attempt_number property from BaseExecutor"
```

### Task 4: Update Job Executor to Remove step_attempt_number Usage

**Files:**
- Modify: `extensions/job_executor/__init__.py:112-125`

**Step 1: Check the current implementation**

Review the step_attempt_number property in job executor to understand its usage

**Step 2: Write test for job executor without property**

```python
def test_job_executor_no_step_attempt_number():
    """Test that job executor doesn't use step_attempt_number property"""
    from extensions.job_executor import BaseJobExecutor

    # Should not have this property anymore
    assert not hasattr(BaseJobExecutor, 'step_attempt_number')
```

**Step 3: Run test to verify current state**

Run: `pytest tests/extensions/test_job_executor.py::test_job_executor_no_step_attempt_number -v`
Expected: FAIL because property still exists

**Step 4: Remove the property from job executor**

Remove the step_attempt_number property from `BaseJobExecutor` class:

```python
# Remove this property entirely from BaseJobExecutor
```

**Step 5: Update any references in job executor**

If job executor needs attempt numbers, update it to receive them as parameters rather than using the property.

**Step 6: Run test to verify it passes**

Run: `pytest tests/extensions/test_job_executor.py::test_job_executor_no_step_attempt_number -v`
Expected: PASS

**Step 7: Commit**

```bash
git add extensions/job_executor/__init__.py tests/extensions/test_job_executor.py
git commit -m "refactor: remove step_attempt_number property from job executor"
```

### Task 5: Integration Testing and Cleanup

**Files:**
- Test: Run full test suite
- Modify: Remove any remaining TODOs

**Step 1: Remove resolved TODOs**

Remove the completed TODO comments from `extensions/pipeline_executor/__init__.py`:

```python
# Remove these lines:
# TODO: Move this to a different function,
# TODO: Remove the property step_attempt_number and use this logic
```

**Step 2: Run full test suite**

Run: `pytest tests/ -v`
Expected: All tests pass

**Step 3: Test with example pipeline**

Run: `uv run examples/01-tasks/python_tasks.py`
Expected: Pipeline executes successfully with correct attempt numbers in logs

**Step 4: Verify attempt number environment variable**

Add a simple test to verify the attempt number environment variable is set correctly during execution.

**Step 5: Final commit**

```bash
git add .
git commit -m "cleanup: remove resolved TODOs for attempt number refactoring"
```

### Task 6: Documentation Update

**Files:**
- Create: `docs/architecture/attempt-number-handling.md`

**Step 1: Document the new approach**

Create documentation explaining how attempt numbers are now calculated:

```markdown
# Attempt Number Handling

## Overview

Attempt numbers are calculated dynamically based on existing attempts in the run log store.

## Implementation

The `GenericPipelineExecutor` calculates attempt numbers by:
1. Querying the run log store for existing attempts
2. Counting the number of previous attempts
3. Setting the next attempt number as count + 1

This approach ensures accurate attempt tracking without maintaining state in executor properties.
```

**Step 2: Commit documentation**

```bash
git add docs/architecture/attempt-number-handling.md
git commit -m "docs: add attempt number handling documentation"
```
