# GenericPartitioned RunLogStore Design

## Overview

Refactor GenericChunked RunLogStore into GenericPartitionedRunLogStore with full partitioning support. Each branch gets completely separate storage for all data types (parameters, step logs, branch logs) using a hierarchical storage structure.

## Architecture

### Hierarchical Storage Structure

```
/run_id/                          # Root partition
├── parameters/                   # Root parameters
├── step_logs/                   # Root step logs
├── branch_logs/                 # Root branch logs
└── branches/
    └── branch_a/                # Branch partition
        ├── parameters/          # Branch-specific parameters
        ├── step_logs/          # Branch-specific step logs
        ├── branch_logs/        # Branch-specific branch logs
        └── branches/
            └── nested_branch_b/ # Nested branch partition
                ├── parameters/
                ├── step_logs/
                └── branch_logs/
```

### Branch Name Parsing

- `internal_branch_name = "branch_a"` → `/run_id/branches/branch_a/`
- `internal_branch_name = "branch_a.nested_branch_b"` → `/run_id/branches/branch_a/branches/nested_branch_b/`
- `internal_branch_name = None` or `""` → `/run_id/` (root)

### Routing Logic

```python
def _get_branch_path(self, internal_branch_name: Optional[str]) -> Optional[str]:
    """Convert internal_branch_name to storage path. Returns None for root."""
    if not internal_branch_name or "." not in internal_branch_name:
        return None  # Root partition
    return internal_branch_name  # Use full path for nested branches
```

## Method Structure

### Updated Public Interface

Methods updated to include `internal_branch_name` parameter:

```python
def get_step_log(self, internal_name: str, run_id: str, internal_branch_name: Optional[str] = None) -> StepLog
def add_step_log(self, step_log: StepLog, run_id: str, internal_branch_name: Optional[str] = None)
def get_branch_log(self, internal_branch_name: str, run_id: str, parent_branch_name: Optional[str] = None) -> Union[BranchLog, RunLog]
def add_branch_log(self, branch_log: Union[BranchLog, RunLog], run_id: str, parent_branch_name: Optional[str] = None)
```

Existing methods maintain compatibility:
```python
def get_parameters(self, run_id: str, internal_branch_name: Optional[str] = None) -> Dict[str, Parameter]
def set_parameters(self, run_id: str, parameters: Dict[str, Parameter], internal_branch_name: Optional[str] = None)
```

### New Abstract Storage Methods

#### Root Partition Storage
```python
@abstractmethod
def _store_root_parameters(self, run_id: str, parameters: Dict[str, Parameter]) -> None
def _retrieve_root_parameters(self, run_id: str) -> Dict[str, Parameter]

def _store_root_step_log(self, run_id: str, step_log: StepLog) -> None
def _retrieve_root_step_log(self, run_id: str, step_name: str) -> StepLog

def _store_root_branch_log(self, run_id: str, branch_log: BranchLog) -> None
def _retrieve_root_branch_log(self, run_id: str, branch_name: str) -> BranchLog
```

#### Branch Partition Storage
```python
@abstractmethod
def _store_branch_parameters(self, run_id: str, internal_branch_name: str, parameters: Dict[str, Parameter]) -> None
def _retrieve_branch_parameters(self, run_id: str, internal_branch_name: str) -> Dict[str, Parameter]

def _store_branch_step_log(self, run_id: str, internal_branch_name: str, step_log: StepLog) -> None
def _retrieve_branch_step_log(self, run_id: str, internal_branch_name: str, step_name: str) -> StepLog

def _store_branch_branch_log(self, run_id: str, internal_branch_name: str, branch_log: BranchLog) -> None
def _retrieve_branch_branch_log(self, run_id: str, internal_branch_name: str, branch_name: str) -> BranchLog
```

## Data Flow and Compilation

### Execution and Storage Flow

```python
# During branch creation - parameter inheritance
create branch_a → copy root parameters to /branches/branch_a/
create nested_branch_b → copy branch_a parameters to /branches/branch_a/branches/nested_branch_b/

# During execution - data stored in partitions
nested_branch_b.step_x → store in /branches/branch_a/branches/nested_branch_b/
branch_a.step_y → store in /branches/branch_a/
root.step_z → store in /root/

# Parameter reduction flow (nested → parent → root)
nested_branch_b completes → parameters reduced to branch_a partition
branch_a completes → parameters reduced to root partition
```

### Parameter Inheritance During Branch Creation

```python
def copy_parameters_to_branch(self, run_id: str, source_branch: Optional[str], target_branch: str):
    """
    Copy parameters from source to target branch during branch creation.
    This ensures nested branch steps have access to parent context parameters.

    Args:
        run_id: The run ID
        source_branch: Source branch name (None for root parameters)
        target_branch: Target branch name to inherit parameters
    """
    if source_branch is None:
        # Copy from root to new branch
        source_params = self._retrieve_root_parameters(run_id)
    else:
        # Copy from parent branch to nested branch
        source_params = self._retrieve_branch_parameters(run_id, source_branch)

    # Deep copy parameters to target branch for isolation
    self._store_branch_parameters(run_id, target_branch, source_params.copy())
```

### Internal Compilation Methods

```python
def compile_branch_results(self, run_id: str, branch_name: str):
    """Compile nested branch results up to parent branch."""
    # Get all nested branch parameters
    # Reduce/merge them according to reduction rules
    # Store reduced parameters in parent branch partition

def finalize_run_log(self, run_id: str) -> RunLog:
    """Final compilation - build complete RunLog from all partitions."""
    # Start with root partition
    # Recursively compile all branch partitions
    # Build hierarchical RunLog structure

def reduce_parameters_to_parent(self, run_id: str, child_branch: str, parent_branch: Optional[str]):
    """Handle parameter x that child_branch produces and parent needs."""
    child_params = self._retrieve_branch_parameters(run_id, child_branch)

    if parent_branch is None:
        # Reduce to root
        self._store_root_parameters(run_id, reduced_params)
    else:
        # Reduce to parent branch
        self._store_branch_parameters(run_id, parent_branch, reduced_params)
```

## Implementation Approach

### Remove Chunked Internals

Remove these from GenericPartitioned:
- `store()` method and its complex naming patterns
- `retrieve()` method and matching logic
- `orderly_retrieve()` method
- `LogTypes` and `ModelTypes` enums
- `naming_pattern()` method
- `get_matches()` method template system

### Replace with Transparent Partitioning

```python
def get_parameters(self, run_id: str, internal_branch_name: Optional[str] = None) -> Dict[str, Parameter]:
    """Domain calls this - partitioning handled internally."""
    if internal_branch_name is None:
        return self._retrieve_root_parameters(run_id)
    else:
        return self._retrieve_branch_parameters(run_id, internal_branch_name)

def get_branch_log(self, internal_branch_name: str, run_id: str) -> Union[BranchLog, RunLog]:
    """Domain calls this - automatic compilation if needed."""
    if not internal_branch_name:
        # Return root log with all compiled branches
        return self._compile_full_run_log(run_id)
    else:
        # Return specific branch log with nested branches compiled
        return self._compile_branch_log(run_id, internal_branch_name)
```

## Key Benefits

- **Clean separation of concerns**: Each execution context has isolated storage
- **Natural hierarchy**: Matches DAG execution flow
- **Parameter reduction support**: Easy compilation from nested → parent → root
- **Backward compatibility**: `internal_branch_name` defaults to `None`, existing calls route to root partition
- **Storage agnostic**: Abstract methods let concrete implementations use their own path conventions
- **Transparent partitioning**: Domain code unchanged, partitioning is implementation detail

## Backward Compatibility

Since `internal_branch_name` defaults to `None` in all updated method signatures, existing domain calls like:
- `get_parameters(run_id)` automatically route to root partition
- `get_step_log(internal_name, run_id)` automatically route to root partition

This maintains full compatibility with existing code while enabling branch-specific functionality when needed.
