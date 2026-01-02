# Contextvars Run Context Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace global run_context with contextvars to fix concurrency issues when multiple pipelines run simultaneously (e.g., FastAPI async endpoints).

**Architecture:** Replace the global `run_context` variable in `context.py` with Python's `contextvars` module for proper request-level isolation. Update all references throughout the codebase to use context-aware getter functions. This ensures each execution context (async task, thread, etc.) maintains its own isolated run context.

**Tech Stack:** Python contextvars, Pydantic, existing runnable framework

---

### Task 1: Write Tests for Context Isolation

**Files:**
- Create: `tests/runnable/test_context_isolation.py`

**Step 1: Write the failing test**

```python
import asyncio
import pytest
from runnable.context import PipelineContext, get_run_context, set_run_context


def test_context_isolation_sync():
    """Test that different contexts don't interfere in sync execution."""
    # Create first context
    context1 = PipelineContext(
        pipeline_definition_file="test1.py",
        run_id="test-run-1",
        catalog={"type": "file-system"},
        secrets={"type": "env-secrets"},
        pickler={"type": "pickle"},
        run_log_store={"type": "memory"},
        pipeline_executor={"type": "local"}
    )

    # Create second context
    context2 = PipelineContext(
        pipeline_definition_file="test2.py",
        run_id="test-run-2",
        catalog={"type": "file-system"},
        secrets={"type": "env-secrets"},
        pickler={"type": "pickle"},
        run_log_store={"type": "memory"},
        pipeline_executor={"type": "local"}
    )

    # Set context1, verify it's returned
    set_run_context(context1)
    assert get_run_context().run_id == "test-run-1"

    # Set context2, verify it overwrites (current behavior)
    set_run_context(context2)
    assert get_run_context().run_id == "test-run-2"


@pytest.mark.asyncio
async def test_context_isolation_async():
    """Test that async tasks maintain separate contexts."""
    results = []

    async def async_task(context_id: str):
        context = PipelineContext(
            pipeline_definition_file=f"test{context_id}.py",
            run_id=f"test-run-{context_id}",
            catalog={"type": "file-system"},
            secrets={"type": "env-secrets"},
            pickler={"type": "pickle"},
            run_log_store={"type": "memory"},
            pipeline_executor={"type": "local"}
        )

        # Each task should maintain its own context
        current_context = get_run_context()
        assert current_context is not None
        assert current_context.run_id == f"test-run-{context_id}"

        # Simulate some async work
        await asyncio.sleep(0.1)

        # Context should still be correct after await
        current_context = get_run_context()
        assert current_context.run_id == f"test-run-{context_id}"
        results.append(context_id)

    # Run multiple async tasks concurrently
    await asyncio.gather(
        async_task("1"),
        async_task("2"),
        async_task("3")
    )

    assert len(results) == 3
    assert set(results) == {"1", "2", "3"}
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/runnable/test_context_isolation.py -v`
Expected: FAIL with "ImportError: cannot import name 'get_run_context' from 'runnable.context'"

**Step 3: Commit failing test**

```bash
git add tests/runnable/test_context_isolation.py
git commit -m "test: add context isolation tests (failing)"
```

### Task 2: Implement Contextvars in context.py

**Files:**
- Modify: `runnable/context.py:1-10` (imports)
- Modify: `runnable/context.py:526-527` (global variable)
- Modify: `runnable/context.py:276-278` (model_post_init)

**Step 1: Add contextvars import**

```python
import contextvars
import hashlib
import importlib
import json
import logging
import os
import sys
from datetime import datetime
from enum import Enum
from functools import cached_property, partial
from typing import Annotated, Any, Callable, Dict, Optional, TYPE_CHECKING
```

**Step 2: Replace global variable with contextvars**

Replace:
```python
run_context: PipelineContext | JobContext = None  # type: ignore
```

With:
```python
# Context variable for thread/async-safe run context storage
if TYPE_CHECKING:
    from typing import Union
    RunnableContextType = Union['PipelineContext', 'JobContext']
else:
    RunnableContextType = Any

_run_context_var: contextvars.ContextVar[Optional[RunnableContextType]] = contextvars.ContextVar(
    'run_context',
    default=None
)

def get_run_context() -> Optional[RunnableContextType]:
    """Get the current run context for this execution context."""
    return _run_context_var.get()

def set_run_context(context: RunnableContextType) -> None:
    """Set the run context for this execution context."""
    _run_context_var.set(context)

# Backward compatibility property (deprecated)
@property
def run_context() -> Optional[RunnableContextType]:
    """Deprecated: Use get_run_context() instead."""
    return get_run_context()
```

**Step 3: Update model_post_init**

Replace:
```python
    def model_post_init(self, __context: Any) -> None:
        os.environ[defaults.ENV_RUN_ID] = self.run_id

        if self.configuration_file:
            os.environ[defaults.RUNNABLE_CONFIGURATION_FILE] = self.configuration_file
        if self.tag:
            os.environ[defaults.RUNNABLE_RUN_TAG] = self.tag

        global run_context
        if not run_context:
            run_context = self  # type: ignore
```

With:
```python
    def model_post_init(self, __context: Any) -> None:
        os.environ[defaults.ENV_RUN_ID] = self.run_id

        if self.configuration_file:
            os.environ[defaults.RUNNABLE_CONFIGURATION_FILE] = self.configuration_file
        if self.tag:
            os.environ[defaults.RUNNABLE_RUN_TAG] = self.tag

        # Set the context using contextvars for proper isolation
        set_run_context(self)
```

**Step 4: Run tests to verify contextvars basics work**

Run: `uv run pytest tests/runnable/test_context_isolation.py::test_context_isolation_sync -v`
Expected: PASS

**Step 5: Commit contextvars implementation**

```bash
git add runnable/context.py
git commit -m "feat: implement contextvars for run_context isolation"
```

### Task 3: Update tasks.py to use get_run_context()

**Files:**
- Modify: `runnable/tasks.py:141` (_context property)
- Modify: `runnable/tasks.py:146` (set_secrets_as_env_variables)
- Modify: `runnable/tasks.py:570` (notebook task)
- Modify: `runnable/tasks.py:716` (shell task)

**Step 1: Write failing test for tasks context usage**

Add to `tests/runnable/test_context_isolation.py`:

```python
def test_task_uses_correct_context():
    """Test that tasks access the correct context."""
    from runnable.tasks import BaseTaskType
    from runnable.context import PipelineContext, set_run_context

    # Create a mock task
    class MockTask(BaseTaskType):
        command: str = "test"
        secrets: list = []

    task = MockTask()

    # Create and set context
    context = PipelineContext(
        pipeline_definition_file="test.py",
        run_id="task-test-run",
        catalog={"type": "file-system"},
        secrets={"type": "env-secrets"},
        pickler={"type": "pickle"},
        run_log_store={"type": "memory"},
        pipeline_executor={"type": "local"}
    )

    # Task should access the correct context
    task_context = task._context
    assert task_context is not None
    assert task_context.run_id == "task-test-run"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/runnable/test_context_isolation.py::test_task_uses_correct_context -v`
Expected: FAIL (tasks still use old context access)

**Step 3: Update tasks.py context access**

Replace:
```python
    @property
    def _context(self):
        return context.run_context
```

With:
```python
    @property
    def _context(self):
        current_context = context.get_run_context()
        if current_context is None:
            raise RuntimeError("No run context available in current execution context")
        return current_context
```

**Step 4: Update secrets handling**

Replace:
```python
    def set_secrets_as_env_variables(self):
        # Preparing the environment for the task execution
        for key in self.secrets:
            secret_value = context.run_context.secrets.get(key)
            os.environ[key] = secret_value
```

With:
```python
    def set_secrets_as_env_variables(self):
        # Preparing the environment for the task execution
        current_context = context.get_run_context()
        if current_context is None:
            raise RuntimeError("No run context available for secrets")

        for key in self.secrets:
            secret_value = current_context.secrets.get(key)
            os.environ[key] = secret_value
```

**Step 5: Update notebook task context usage**

Find line around 570 and replace:
```python
                context.run_context.catalog.put(name=notebook_output_path)
```

With:
```python
                current_context = context.get_run_context()
                if current_context is None:
                    raise RuntimeError("No run context available for catalog operations")
                current_context.catalog.put(name=notebook_output_path)
```

**Step 6: Update shell task secrets**

Find line around 716 and replace:
```python
        if self.secrets:
            for key in self.secrets:
                secret_value = context.run_context.secrets.get(key)
                subprocess_env[key] = secret_value
```

With:
```python
        if self.secrets:
            current_context = context.get_run_context()
            if current_context is None:
                raise RuntimeError("No run context available for secrets")

            for key in self.secrets:
                secret_value = current_context.secrets.get(key)
                subprocess_env[key] = secret_value
```

**Step 7: Run task tests**

Run: `uv run pytest tests/runnable/test_context_isolation.py::test_task_uses_correct_context -v`
Expected: PASS

**Step 8: Commit tasks.py updates**

```bash
git add runnable/tasks.py
git commit -m "feat: update tasks.py to use contextvars run_context"
```

### Task 4: Update executor.py to use get_run_context()

**Files:**
- Modify: `runnable/executor.py:45` (_context property)

**Step 1: Write test for executor context usage**

Add to `tests/runnable/test_context_isolation.py`:

```python
def test_executor_uses_correct_context():
    """Test that executors access the correct context."""
    from runnable.executor import BaseJobExecutor
    from runnable.context import PipelineContext, set_run_context

    # Create mock executor
    class MockExecutor(BaseJobExecutor):
        def submit_job(self, job, catalog_settings=None):
            pass

    executor = MockExecutor()

    # Create and set context
    context = PipelineContext(
        pipeline_definition_file="test.py",
        run_id="executor-test-run",
        catalog={"type": "file-system"},
        secrets={"type": "env-secrets"},
        pickler={"type": "pickle"},
        run_log_store={"type": "memory"},
        pipeline_executor={"type": "local"}
    )

    # Executor should access correct context
    executor_context = executor._context
    assert executor_context is not None
    assert executor_context.run_id == "executor-test-run"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/runnable/test_context_isolation.py::test_executor_uses_correct_context -v`
Expected: FAIL

**Step 3: Update executor.py**

Replace:
```python
    @property
    def _context(self):
        return context.run_context
```

With:
```python
    @property
    def _context(self):
        current_context = context.get_run_context()
        if current_context is None:
            raise RuntimeError("No run context available in current execution context")
        return current_context
```

**Step 4: Run executor test**

Run: `uv run pytest tests/runnable/test_context_isolation.py::test_executor_uses_correct_context -v`
Expected: PASS

**Step 5: Commit executor.py updates**

```bash
git add runnable/executor.py
git commit -m "feat: update executor.py to use contextvars run_context"
```

### Task 5: Update datastore.py to use get_run_context()

**Files:**
- Modify: `runnable/datastore.py:101,108,112-132,405,571` (all run_context references)

**Step 1: Write test for datastore context usage**

Add to `tests/runnable/test_context_isolation.py`:

```python
def test_datastore_uses_correct_context():
    """Test that datastore components access the correct context."""
    from runnable.datastore import PickledParameter
    from runnable.context import PipelineContext, set_run_context

    # Create and set context
    context = PipelineContext(
        pipeline_definition_file="test.py",
        run_id="datastore-test-run",
        catalog={"type": "file-system"},
        secrets={"type": "env-secrets"},
        pickler={"type": "pickle"},
        run_log_store={"type": "memory"},
        pipeline_executor={"type": "local"}
    )

    # Create pickled parameter
    param = PickledParameter(value="test_param")

    # Should access correct context for serialization settings
    assert param.description.startswith("Pickled object")
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/runnable/test_context_isolation.py::test_datastore_uses_correct_context -v`
Expected: FAIL

**Step 3: Update datastore.py context access patterns**

Update all `context.run_context` references to use `context.get_run_context()`:

```python
    @property
    def description(self) -> str:
        current_context = context.get_run_context()
        if current_context and current_context.object_serialisation:
            return f"Pickled object stored in catalog as: {self.value}"
        return f"Object parameter: {self.value}"

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

**Step 4: Update other datastore context references**

Find and update remaining references around lines 405 and 571:

```python
        summary: Dict[str, Any] = {}

        current_context = context.get_run_context()
        if current_context is None:
            raise RuntimeError("No run context available")

        summary["Unique execution id"] = self.run_id
```

And:

```python
    @property
    def _context(self):
        current_context = context.get_run_context()
        if current_context is None:
            raise RuntimeError("No run context available")
        return current_context
```

**Step 5: Run datastore test**

Run: `uv run pytest tests/runnable/test_context_isolation.py::test_datastore_uses_correct_context -v`
Expected: PASS

**Step 6: Commit datastore.py updates**

```bash
git add runnable/datastore.py
git commit -m "feat: update datastore.py to use contextvars run_context"
```

### Task 6: Update Remaining Files

**Files:**
- Modify: All files found in Grep search for `run_context`

**Step 1: Find all remaining run_context references**

Run: `grep -r "context.run_context" runnable/ extensions/ --exclude-dir=__pycache__ | grep -v ".pyc"`

**Step 2: Update each file systematically**

For each file found, replace `context.run_context` with:
```python
current_context = context.get_run_context()
if current_context is None:
    raise RuntimeError("No run context available")
# Then use current_context instead of context.run_context
```

**Step 3: Update import statements where needed**

Add imports for the new functions:
```python
from runnable.context import get_run_context
```

**Step 4: Run targeted tests for each updated file**

Run: `uv run pytest tests/runnable/ -v -k "test_[filename]"`

**Step 5: Commit each file update**

```bash
git add [filename]
git commit -m "feat: update [filename] to use contextvars run_context"
```

### Task 7: Run Full Async Context Test

**Files:**
- Test: `tests/runnable/test_context_isolation.py`

**Step 1: Run the async isolation test**

Run: `uv run pytest tests/runnable/test_context_isolation.py::test_context_isolation_async -v`
Expected: PASS - multiple async tasks should maintain separate contexts

**Step 2: Run all context isolation tests**

Run: `uv run pytest tests/runnable/test_context_isolation.py -v`
Expected: All tests PASS

**Step 3: Commit final test verification**

```bash
git add tests/runnable/test_context_isolation.py
git commit -m "test: verify context isolation works correctly"
```

### Task 8: Integration Testing

**Files:**
- Test: Run existing test suite to ensure no regressions

**Step 1: Run core runnable tests**

Run: `uv run pytest tests/runnable/ -v`
Expected: All existing tests still PASS

**Step 2: Run task-specific tests**

Run: `uv run pytest tests/runnable/test_tasks.py -v`
Expected: All task tests PASS with new context system

**Step 3: Run executor tests**

Run: `uv run pytest tests/runnable/test_executor.py -v`
Expected: All executor tests PASS

**Step 4: Run integration tests**

Run: `uv run pytest tests/test_pipeline_examples.py -v -k "python_tasks"`
Expected: Pipeline execution works with contextvars

**Step 5: Final commit**

```bash
git add .
git commit -m "feat: complete contextvars implementation for run_context

- Replace global run_context with contextvars for thread/async safety
- Fix concurrency issues when multiple pipelines run simultaneously
- Maintain backward compatibility where possible
- Add comprehensive context isolation tests"
```

### Task 9: Documentation Update

**Files:**
- Create: `docs/architecture/context-isolation.md`

**Step 1: Document the context isolation architecture**

```markdown
# Context Isolation in Runnable

## Problem Solved

Previously, `run_context` was a global variable that caused issues when multiple pipelines ran concurrently (e.g., in FastAPI endpoints). All pipelines would share the same context, leading to:

- Data leakage between pipelines
- Incorrect run IDs in logs
- Configuration mix-ups
- Resource conflicts

## Solution

Replaced the global variable with Python's `contextvars` module, providing:

- **Request isolation**: Each execution context maintains its own run context
- **Async safety**: Contexts automatically propagate through async/await chains
- **Thread safety**: Works correctly with thread pools and concurrent execution
- **Explicit error handling**: Clear errors when no context is available

## Usage

```python
from runnable.context import get_run_context, set_run_context

# Get current context (returns None if no context)
current_context = get_run_context()

# Set context (automatically isolated per request/task)
set_run_context(my_context)

# Context automatically propagates through async chains
async def my_async_function():
    context = get_run_context()  # Same context as caller
    await some_other_async_function()
```

## Migration Notes

- Old global `context.run_context` still works but is deprecated
- New code should use `get_run_context()` instead
- Error handling is now explicit - functions raise `RuntimeError` if no context available
- No changes needed for FastAPI or async usage - isolation happens automatically
```

**Step 2: Commit documentation**

```bash
git add docs/architecture/context-isolation.md
git commit -m "docs: add context isolation architecture documentation"
```

---

## Verification Checklist

- [ ] All tests pass: `uv run pytest`
- [ ] Context isolation works in async scenarios
- [ ] Multiple concurrent pipeline executions don't interfere
- [ ] Existing functionality preserved
- [ ] Error handling is clear and explicit
- [ ] Documentation explains the architecture

## Key Files Modified

- `runnable/context.py` - Core contextvars implementation
- `runnable/tasks.py` - Task context access
- `runnable/executor.py` - Executor context access
- `runnable/datastore.py` - Datastore context access
- All other files with `context.run_context` references
- `tests/runnable/test_context_isolation.py` - New isolation tests
