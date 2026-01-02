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

- The global `context.run_context` variable has been removed
- New code must use `get_run_context()` to access the context
- Error handling is now explicit - functions raise `RuntimeError` if no context available
- No changes needed for FastAPI or async usage - isolation happens automatically

## Implementation Details

The implementation uses Python's `contextvars.ContextVar`:

```python
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
```

This ensures each async task, thread, or request maintains its own isolated context.
