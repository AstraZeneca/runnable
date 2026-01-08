# GenericPartitioned RunLogStore Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor GenericPartitionedRunLogStore from chunked approach to full partitioned storage with hierarchical branch support.

**Architecture:** Replace GenericChunked inheritance with direct BaseRunLogStore implementation, adding explicit abstract methods for root/branch storage and transparent parameter inheritance.

**Tech Stack:** Python, abstract methods, hierarchical storage patterns

---

## Task 1: Set Up Test Infrastructure

**Files:**
- Create: `tests/extensions/run_log_store/test_generic_partitioned.py`
- Modify: `extensions/run_log_store/generic_partitioned.py:1-121`

**Step 1: Write failing test for basic abstract method structure**

```python
# tests/extensions/run_log_store/test_generic_partitioned.py
import pytest
from unittest.mock import Mock
from runnable.datastore import Parameter, RunLog, StepLog, BranchLog
from extensions.run_log_store.generic_partitioned import GenericPartitionedRunLogStore

class ConcretePartitionedStore(GenericPartitionedRunLogStore):
    """Test implementation of GenericPartitionedRunLogStore"""

    def get_summary(self):
        return {"service_name": "test-partitioned"}

    def create_run_log(self, run_id: str, dag_hash: str = "", use_cached: bool = False,
                       tag: str = "", original_run_id: str = "", status: str = "CREATED"):
        return RunLog(run_id=run_id, dag_hash=dag_hash, tag=tag, status=status)

    def get_run_log_by_id(self, run_id: str, full: bool = False) -> RunLog:
        return RunLog(run_id=run_id)

    def put_run_log(self, run_log: RunLog):
        pass

    # These should be implemented as abstract methods
    def _store_root_parameters(self, run_id: str, parameters: dict):
        self._root_params = {run_id: parameters}

    def _retrieve_root_parameters(self, run_id: str) -> dict:
        return self._root_params.get(run_id, {})

def test_concrete_store_instantiation():
    """Test that ConcretePartitionedStore can be instantiated"""
    store = ConcretePartitionedStore()
    assert store.service_name == "generic-partitioned"
    assert store.supports_parallel_writes == True

def test_abstract_method_structure():
    """Test that abstract methods are properly defined"""
    # This will fail initially as methods don't exist yet
    store = ConcretePartitionedStore()

    # Test that these methods exist (will fail until implemented)
    assert hasattr(store, '_store_root_parameters')
    assert hasattr(store, '_retrieve_root_parameters')
    assert hasattr(store, '_store_branch_parameters')
    assert hasattr(store, '_retrieve_branch_parameters')
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/extensions/run_log_store/test_generic_partitioned.py::test_abstract_method_structure -v`
Expected: FAIL with AttributeError for missing branch methods

**Step 3: Add abstract method signatures to GenericPartitioned**

```python
# extensions/run_log_store/generic_partitioned.py
import json
import logging
from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union

from runnable import defaults
from runnable.datastore import BaseRunLogStore, Parameter, RunLog, StepLog, BranchLog

logger = logging.getLogger(defaults.LOGGER_NAME)


class GenericPartitionedRunLogStore(BaseRunLogStore):
    """
    A generic implementation of a RunLogStore that supports partitioned parameter storage.

    This store enables branch-specific parameter isolation by storing parameters
    in separate partitions/files per branch execution context.
    """

    service_name: str = "generic-partitioned"
    supports_parallel_writes: bool = True

    # Root partition abstract methods
    @abstractmethod
    def _store_root_parameters(self, run_id: str, parameters: Dict[str, Parameter]) -> None:
        """Store parameters in root partition."""
        ...

    @abstractmethod
    def _retrieve_root_parameters(self, run_id: str) -> Dict[str, Parameter]:
        """Retrieve parameters from root partition."""
        ...

    @abstractmethod
    def _store_root_step_log(self, run_id: str, step_log: StepLog) -> None:
        """Store step log in root partition."""
        ...

    @abstractmethod
    def _retrieve_root_step_log(self, run_id: str, step_name: str) -> StepLog:
        """Retrieve step log from root partition."""
        ...

    @abstractmethod
    def _store_root_branch_log(self, run_id: str, branch_log: BranchLog) -> None:
        """Store branch log in root partition."""
        ...

    @abstractmethod
    def _retrieve_root_branch_log(self, run_id: str, branch_name: str) -> BranchLog:
        """Retrieve branch log from root partition."""
        ...

    # Branch partition abstract methods
    @abstractmethod
    def _store_branch_parameters(self, run_id: str, internal_branch_name: str, parameters: Dict[str, Parameter]) -> None:
        """Store parameters in branch partition."""
        ...

    @abstractmethod
    def _retrieve_branch_parameters(self, run_id: str, internal_branch_name: str) -> Dict[str, Parameter]:
        """Retrieve parameters from branch partition."""
        ...

    @abstractmethod
    def _store_branch_step_log(self, run_id: str, internal_branch_name: str, step_log: StepLog) -> None:
        """Store step log in branch partition."""
        ...

    @abstractmethod
    def _retrieve_branch_step_log(self, run_id: str, internal_branch_name: str, step_name: str) -> StepLog:
        """Retrieve step log from branch partition."""
        ...

    @abstractmethod
    def _store_branch_branch_log(self, run_id: str, internal_branch_name: str, branch_log: BranchLog) -> None:
        """Store branch log in branch partition."""
        ...

    @abstractmethod
    def _retrieve_branch_branch_log(self, run_id: str, internal_branch_name: str, branch_name: str) -> BranchLog:
        """Retrieve branch log from branch partition."""
        ...
```

**Step 4: Update test to implement missing abstract methods**

```python
# tests/extensions/run_log_store/test_generic_partitioned.py (update ConcretePartitionedStore)
class ConcretePartitionedStore(GenericPartitionedRunLogStore):
    """Test implementation of GenericPartitionedRunLogStore"""

    def __init__(self):
        super().__init__()
        self._root_params = {}
        self._root_steps = {}
        self._root_branches = {}
        self._branch_params = {}
        self._branch_steps = {}
        self._branch_branches = {}

    def get_summary(self):
        return {"service_name": "test-partitioned"}

    def create_run_log(self, run_id: str, dag_hash: str = "", use_cached: bool = False,
                       tag: str = "", original_run_id: str = "", status: str = "CREATED"):
        return RunLog(run_id=run_id, dag_hash=dag_hash, tag=tag, status=status)

    def get_run_log_by_id(self, run_id: str, full: bool = False) -> RunLog:
        return RunLog(run_id=run_id)

    def put_run_log(self, run_log: RunLog):
        pass

    # Root partition implementations
    def _store_root_parameters(self, run_id: str, parameters: Dict[str, Parameter]) -> None:
        self._root_params[run_id] = parameters

    def _retrieve_root_parameters(self, run_id: str) -> Dict[str, Parameter]:
        return self._root_params.get(run_id, {})

    def _store_root_step_log(self, run_id: str, step_log: StepLog) -> None:
        key = f"{run_id}:{step_log.internal_name}"
        self._root_steps[key] = step_log

    def _retrieve_root_step_log(self, run_id: str, step_name: str) -> StepLog:
        key = f"{run_id}:{step_name}"
        return self._root_steps[key]

    def _store_root_branch_log(self, run_id: str, branch_log: BranchLog) -> None:
        key = f"{run_id}:{branch_log.internal_name}"
        self._root_branches[key] = branch_log

    def _retrieve_root_branch_log(self, run_id: str, branch_name: str) -> BranchLog:
        key = f"{run_id}:{branch_name}"
        return self._root_branches[key]

    # Branch partition implementations
    def _store_branch_parameters(self, run_id: str, internal_branch_name: str, parameters: Dict[str, Parameter]) -> None:
        key = f"{run_id}:{internal_branch_name}"
        self._branch_params[key] = parameters

    def _retrieve_branch_parameters(self, run_id: str, internal_branch_name: str) -> Dict[str, Parameter]:
        key = f"{run_id}:{internal_branch_name}"
        return self._branch_params.get(key, {})

    def _store_branch_step_log(self, run_id: str, internal_branch_name: str, step_log: StepLog) -> None:
        key = f"{run_id}:{internal_branch_name}:{step_log.internal_name}"
        self._branch_steps[key] = step_log

    def _retrieve_branch_step_log(self, run_id: str, internal_branch_name: str, step_name: str) -> StepLog:
        key = f"{run_id}:{internal_branch_name}:{step_name}"
        return self._branch_steps[key]

    def _store_branch_branch_log(self, run_id: str, internal_branch_name: str, branch_log: BranchLog) -> None:
        key = f"{run_id}:{internal_branch_name}:{branch_log.internal_name}"
        self._branch_branches[key] = branch_log

    def _retrieve_branch_branch_log(self, run_id: str, internal_branch_name: str, branch_name: str) -> BranchLog:
        key = f"{run_id}:{internal_branch_name}:{branch_name}"
        return self._branch_branches[key]
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/extensions/run_log_store/test_generic_partitioned.py::test_abstract_method_structure -v`
Expected: PASS

**Step 6: Commit**

```bash
git add extensions/run_log_store/generic_partitioned.py tests/extensions/run_log_store/test_generic_partitioned.py
git commit -m "feat: add abstract method structure for partitioned storage"
```

---

## Task 2: Implement Parameter Management

**Files:**
- Modify: `extensions/run_log_store/generic_partitioned.py:90-121`
- Modify: `tests/extensions/run_log_store/test_generic_partitioned.py`

**Step 1: Write failing test for parameter inheritance**

```python
# Add to test_generic_partitioned.py
from runnable.datastore import JsonParameter

def test_copy_parameters_to_branch():
    """Test parameter inheritance during branch creation"""
    store = ConcretePartitionedStore()
    run_id = "test_run"

    # Set up root parameters
    root_params = {
        "param1": JsonParameter(kind="json", value={"key": "value"})
    }
    store._store_root_parameters(run_id, root_params)

    # Copy to branch - this method doesn't exist yet
    store.copy_parameters_to_branch(run_id, None, "branch1")

    # Verify branch has inherited parameters
    branch_params = store._retrieve_branch_parameters(run_id, "branch1")
    assert "param1" in branch_params
    assert branch_params["param1"].value == {"key": "value"}

def test_get_parameters_routing():
    """Test get_parameters routes correctly based on internal_branch_name"""
    store = ConcretePartitionedStore()
    run_id = "test_run"

    # Set up different parameters for root and branch
    root_params = {"root_param": JsonParameter(kind="json", value="root_value")}
    branch_params = {"branch_param": JsonParameter(kind="json", value="branch_value")}

    store._store_root_parameters(run_id, root_params)
    store._store_branch_parameters(run_id, "branch1", branch_params)

    # Test routing - these methods don't exist yet
    root_result = store.get_parameters(run_id, None)
    branch_result = store.get_parameters(run_id, "branch1")

    assert "root_param" in root_result
    assert "branch_param" in branch_result

def test_set_parameters_routing():
    """Test set_parameters routes correctly based on internal_branch_name"""
    store = ConcretePartitionedStore()
    run_id = "test_run"

    root_params = {"new_root": JsonParameter(kind="json", value="new_root_value")}
    branch_params = {"new_branch": JsonParameter(kind="json", value="new_branch_value")}

    # These methods don't exist yet
    store.set_parameters(run_id, root_params, None)
    store.set_parameters(run_id, branch_params, "branch1")

    # Verify storage
    assert store._retrieve_root_parameters(run_id)["new_root"].value == "new_root_value"
    assert store._retrieve_branch_parameters(run_id, "branch1")["new_branch"].value == "new_branch_value"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/extensions/run_log_store/test_generic_partitioned.py::test_copy_parameters_to_branch -v`
Expected: FAIL with AttributeError for missing method

**Step 3: Implement parameter management methods**

```python
# Add to extensions/run_log_store/generic_partitioned.py after abstract methods

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

def get_parameters(self, run_id: str, internal_branch_name: Optional[str] = None) -> Dict[str, Parameter]:
    """
    Get parameters with branch-specific support.

    Args:
        run_id: The run ID
        internal_branch_name: If provided, get parameters for specific branch

    Returns:
        Dictionary of parameters
    """
    if internal_branch_name is None or "." not in internal_branch_name:
        # Get main run parameters
        return self._retrieve_root_parameters(run_id)
    else:
        # Get branch-specific parameters
        return self._retrieve_branch_parameters(run_id, internal_branch_name)

def set_parameters(self, run_id: str, parameters: Dict[str, Parameter], internal_branch_name: Optional[str] = None):
    """
    Set parameters with branch-specific support.

    Args:
        run_id: The run ID
        parameters: Parameters to set
        internal_branch_name: If provided, set parameters for specific branch
    """
    if internal_branch_name is None or "." not in internal_branch_name:
        # Set main run parameters
        self._store_root_parameters(run_id, parameters)
    else:
        # Set branch-specific parameters
        self._store_branch_parameters(run_id, internal_branch_name, parameters)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/extensions/run_log_store/test_generic_partitioned.py::test_copy_parameters_to_branch -v`
Expected: PASS

**Step 5: Commit**

```bash
git add extensions/run_log_store/generic_partitioned.py tests/extensions/run_log_store/test_generic_partitioned.py
git commit -m "feat: implement parameter inheritance and routing logic"
```

---

## Task 3: Implement Step Log Management

**Files:**
- Modify: `extensions/run_log_store/generic_partitioned.py`
- Modify: `tests/extensions/run_log_store/test_generic_partitioned.py`

**Step 1: Write failing test for step log management**

```python
# Add to test_generic_partitioned.py
from runnable import defaults

def test_get_step_log_routing():
    """Test get_step_log routes correctly based on internal_branch_name"""
    store = ConcretePartitionedStore()
    run_id = "test_run"

    # Create step logs for root and branch
    root_step = StepLog(name="root_step", internal_name="root_step", status=defaults.CREATED)
    branch_step = StepLog(name="branch_step", internal_name="branch1.branch_step", status=defaults.CREATED)

    store._store_root_step_log(run_id, root_step)
    store._store_branch_step_log(run_id, "branch1", branch_step)

    # Test routing - these methods don't exist yet with internal_branch_name parameter
    root_result = store.get_step_log("root_step", run_id, None)
    branch_result = store.get_step_log("branch1.branch_step", run_id, "branch1")

    assert root_result.name == "root_step"
    assert branch_result.name == "branch_step"

def test_add_step_log_routing():
    """Test add_step_log routes correctly based on internal_branch_name"""
    store = ConcretePartitionedStore()
    run_id = "test_run"

    root_step = StepLog(name="new_root_step", internal_name="new_root_step", status=defaults.CREATED)
    branch_step = StepLog(name="new_branch_step", internal_name="branch1.new_branch_step", status=defaults.CREATED)

    # These methods don't exist yet with internal_branch_name parameter
    store.add_step_log(root_step, run_id, None)
    store.add_step_log(branch_step, run_id, "branch1")

    # Verify storage
    assert store._retrieve_root_step_log(run_id, "new_root_step").name == "new_root_step"
    assert store._retrieve_branch_step_log(run_id, "branch1", "branch1.new_branch_step").name == "new_branch_step"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/extensions/run_log_store/test_generic_partitioned.py::test_get_step_log_routing -v`
Expected: FAIL with TypeError for missing internal_branch_name parameter

**Step 3: Implement step log management methods**

```python
# Add to extensions/run_log_store/generic_partitioned.py after parameter methods

def get_step_log(self, internal_name: str, run_id: str, internal_branch_name: Optional[str] = None) -> StepLog:
    """
    Get a step log from the datastore for run_id and the internal naming of the step log.

    Args:
        internal_name: The internal name of the step log
        run_id: The run_id of the run
        internal_branch_name: If provided, get from specific branch partition

    Returns:
        StepLog: The step log object for the step
    """
    if internal_branch_name is None or "." not in internal_branch_name:
        # Get from root partition
        return self._retrieve_root_step_log(run_id, internal_name)
    else:
        # Get from branch partition
        return self._retrieve_branch_step_log(run_id, internal_branch_name, internal_name)

def add_step_log(self, step_log: StepLog, run_id: str, internal_branch_name: Optional[str] = None):
    """
    Add the step log in the run log as identified by the run_id in the datastore.

    Args:
        step_log: The Step log to add to the database
        run_id: The run id of the run
        internal_branch_name: If provided, store in specific branch partition
    """
    if internal_branch_name is None or "." not in internal_branch_name:
        # Store in root partition
        self._store_root_step_log(run_id, step_log)
    else:
        # Store in branch partition
        self._store_branch_step_log(run_id, internal_branch_name, step_log)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/extensions/run_log_store/test_generic_partitioned.py::test_get_step_log_routing -v`
Expected: PASS

**Step 5: Commit**

```bash
git add extensions/run_log_store/generic_partitioned.py tests/extensions/run_log_store/test_generic_partitioned.py
git commit -m "feat: implement step log management with partitioned routing"
```

---

## Task 4: Implement Branch Log Management

**Files:**
- Modify: `extensions/run_log_store/generic_partitioned.py`
- Modify: `tests/extensions/run_log_store/test_generic_partitioned.py`

**Step 1: Write failing test for branch log management**

```python
# Add to test_generic_partitioned.py

def test_get_branch_log_routing():
    """Test get_branch_log routes correctly"""
    store = ConcretePartitionedStore()
    run_id = "test_run"

    # Test root log (empty internal_branch_name)
    root_log = store.get_run_log_by_id(run_id)

    # Create branch log
    branch_log = BranchLog(internal_name="branch1", status=defaults.CREATED)
    store._store_branch_branch_log(run_id, "parent_branch", branch_log)

    # Test routing - method signature needs to be updated
    root_result = store.get_branch_log("", run_id, None)  # Empty string for root
    branch_result = store.get_branch_log("branch1", run_id, "parent_branch")

    assert isinstance(root_result, RunLog)
    assert isinstance(branch_result, BranchLog)
    assert branch_result.internal_name == "branch1"

def test_add_branch_log_routing():
    """Test add_branch_log routes correctly"""
    store = ConcretePartitionedStore()
    run_id = "test_run"

    # Test adding root log (RunLog instance)
    run_log = RunLog(run_id=run_id, status=defaults.CREATED)
    branch_log = BranchLog(internal_name="new_branch", status=defaults.CREATED)

    # These methods need signature updates
    store.add_branch_log(run_log, run_id, None)  # Root log
    store.add_branch_log(branch_log, run_id, "parent_branch")  # Branch log

    # Verify storage - root log goes through put_run_log, branch log goes to partition
    retrieved_branch = store._retrieve_branch_branch_log(run_id, "parent_branch", "new_branch")
    assert retrieved_branch.internal_name == "new_branch"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/extensions/run_log_store/test_generic_partitioned.py::test_get_branch_log_routing -v`
Expected: FAIL with TypeError for missing parent_branch_name parameter

**Step 3: Implement branch log management methods**

```python
# Add to extensions/run_log_store/generic_partitioned.py after step log methods

def get_branch_log(self, internal_branch_name: str, run_id: str, parent_branch_name: Optional[str] = None) -> Union[BranchLog, RunLog]:
    """
    Returns the branch log by the internal branch name for the run id.

    If the internal branch name is empty, returns the run log.

    Args:
        internal_branch_name: The internal branch name to retrieve
        run_id: The run id of interest
        parent_branch_name: The parent branch containing this branch (None for root)

    Returns:
        BranchLog or RunLog: The branch log or the run log as requested
    """
    if not internal_branch_name:
        # Return root run log
        return self.get_run_log_by_id(run_id=run_id)

    if parent_branch_name is None or "." not in parent_branch_name:
        # Retrieve from root partition
        return self._retrieve_root_branch_log(run_id, internal_branch_name)
    else:
        # Retrieve from parent branch partition
        return self._retrieve_branch_branch_log(run_id, parent_branch_name, internal_branch_name)

def add_branch_log(self, branch_log: Union[BranchLog, RunLog], run_id: str, parent_branch_name: Optional[str] = None):
    """
    Add the branch log to the appropriate partition.

    Args:
        branch_log: The branch log/run log to add to the database
        run_id: The run id to which the branch/run log is added
        parent_branch_name: The parent branch containing this branch (None for root)
    """
    if not isinstance(branch_log, BranchLog):
        # This is a RunLog, store it directly
        self.put_run_log(branch_log)
        return

    internal_branch_name = branch_log.internal_name

    if parent_branch_name is None or "." not in parent_branch_name:
        # Store in root partition
        self._store_root_branch_log(run_id, branch_log)
    else:
        # Store in parent branch partition
        self._store_branch_branch_log(run_id, parent_branch_name, branch_log)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/extensions/run_log_store/test_generic_partitioned.py::test_get_branch_log_routing -v`
Expected: PASS

**Step 5: Commit**

```bash
git add extensions/run_log_store/generic_partitioned.py tests/extensions/run_log_store/test_generic_partitioned.py
git commit -m "feat: implement branch log management with partitioned routing"
```

---

## Task 5: Remove Chunked Method Stubs

**Files:**
- Modify: `extensions/run_log_store/generic_partitioned.py:52-121`

**Step 1: Write test to ensure chunked methods are removed**

```python
# Add to test_generic_partitioned.py

def test_chunked_methods_removed():
    """Verify that chunked-style methods are no longer present"""
    store = ConcretePartitionedStore()

    # These methods should not exist in partitioned implementation
    chunked_methods = [
        '_get_main_parameters',
        '_set_main_parameters',
        '_get_branch_parameters',
        '_set_branch_parameters'
    ]

    for method_name in chunked_methods:
        # These should not exist anymore
        assert not hasattr(store, method_name), f"Method {method_name} should be removed from partitioned implementation"
```

**Step 2: Run test to verify current state**

Run: `pytest tests/extensions/run_log_store/test_generic_partitioned.py::test_chunked_methods_removed -v`
Expected: FAIL because methods still exist

**Step 3: Remove chunked method stubs from GenericPartitioned**

```python
# Remove these methods from extensions/run_log_store/generic_partitioned.py:

# DELETE this entire section (lines ~87-121):
# @abstractmethod
# def _get_main_parameters(self, run_id: str) -> Dict[str, Parameter]:
#     """Get main run parameters."""
#     ...

# @abstractmethod
# def _set_main_parameters(self, run_id: str, parameters: Dict[str, Parameter]):
#     """Set main run parameters."""
#     ...

# @abstractmethod
# def _get_branch_parameters(self, run_id: str, internal_branch_name: str) -> Dict[str, Parameter]:
#     """Get parameters for a specific branch."""
#     ...

# @abstractmethod
# def _set_branch_parameters(self, run_id: str, internal_branch_name: str, parameters: Dict[str, Parameter]):
#     """Set parameters for a specific branch."""
#     ...

# def copy_parameters_to_branch(self, run_id: str, source_branch: Optional[str], target_branch: str):
#     # ... existing implementation was moved earlier
```

**Step 4: Clean up the existing parameter methods section**

The existing parameter methods (get_parameters, set_parameters) should be replaced by our new implementations.

**Step 5: Run test to verify it passes**

Run: `pytest tests/extensions/run_log_store/test_generic_partitioned.py::test_chunked_methods_removed -v`
Expected: PASS

**Step 6: Run all tests to verify nothing broke**

Run: `pytest tests/extensions/run_log_store/test_generic_partitioned.py -v`
Expected: All tests PASS

**Step 7: Commit**

```bash
git add extensions/run_log_store/generic_partitioned.py tests/extensions/run_log_store/test_generic_partitioned.py
git commit -m "refactor: remove chunked method stubs from partitioned implementation"
```

---

## Task 6: Add Comprehensive Integration Tests

**Files:**
- Modify: `tests/extensions/run_log_store/test_generic_partitioned.py`

**Step 1: Write integration test for complete workflow**

```python
# Add to test_generic_partitioned.py

def test_complete_partitioned_workflow():
    """Integration test for complete partitioned workflow"""
    store = ConcretePartitionedStore()
    run_id = "integration_test_run"

    # 1. Create root parameters
    root_params = {
        "global_config": JsonParameter(kind="json", value={"env": "test"}),
        "shared_data": JsonParameter(kind="json", value=[1, 2, 3])
    }
    store.set_parameters(run_id, root_params, None)

    # 2. Create branch and inherit parameters
    store.copy_parameters_to_branch(run_id, None, "branch1")

    # 3. Verify branch inherited parameters
    branch_params = store.get_parameters(run_id, "branch1")
    assert "global_config" in branch_params
    assert branch_params["global_config"].value == {"env": "test"}

    # 4. Add branch-specific parameters
    branch_specific = {
        "branch_config": JsonParameter(kind="json", value={"branch_id": 1})
    }
    current_branch_params = store.get_parameters(run_id, "branch1")
    current_branch_params.update(branch_specific)
    store.set_parameters(run_id, current_branch_params, "branch1")

    # 5. Create nested branch
    store.copy_parameters_to_branch(run_id, "branch1", "branch1.nested")
    nested_params = store.get_parameters(run_id, "branch1.nested")
    assert "global_config" in nested_params  # Inherited from root via branch1
    assert "branch_config" in nested_params  # Inherited from branch1

    # 6. Add step logs to different partitions
    root_step = StepLog(name="root_step", internal_name="root_step", status=defaults.CREATED)
    branch_step = StepLog(name="branch_step", internal_name="branch1.branch_step", status=defaults.CREATED)
    nested_step = StepLog(name="nested_step", internal_name="branch1.nested.step", status=defaults.CREATED)

    store.add_step_log(root_step, run_id, None)
    store.add_step_log(branch_step, run_id, "branch1")
    store.add_step_log(nested_step, run_id, "branch1.nested")

    # 7. Verify step logs are in correct partitions
    retrieved_root = store.get_step_log("root_step", run_id, None)
    retrieved_branch = store.get_step_log("branch1.branch_step", run_id, "branch1")
    retrieved_nested = store.get_step_log("branch1.nested.step", run_id, "branch1.nested")

    assert retrieved_root.name == "root_step"
    assert retrieved_branch.name == "branch_step"
    assert retrieved_nested.name == "nested_step"

    # 8. Add branch logs
    branch_log = BranchLog(internal_name="branch1", status=defaults.SUCCESS)
    nested_branch_log = BranchLog(internal_name="nested", status=defaults.SUCCESS)

    store.add_branch_log(branch_log, run_id, None)  # branch1 stored in root
    store.add_branch_log(nested_branch_log, run_id, "branch1")  # nested stored in branch1

    # 9. Verify branch logs
    retrieved_branch_log = store.get_branch_log("branch1", run_id, None)
    retrieved_nested_log = store.get_branch_log("nested", run_id, "branch1")

    assert retrieved_branch_log.internal_name == "branch1"
    assert retrieved_nested_log.internal_name == "nested"

def test_parameter_isolation():
    """Test that branch parameters are properly isolated"""
    store = ConcretePartitionedStore()
    run_id = "isolation_test"

    # Set different parameters in root and branches
    root_params = {"param": JsonParameter(kind="json", value="root_value")}
    branch1_params = {"param": JsonParameter(kind="json", value="branch1_value")}
    branch2_params = {"param": JsonParameter(kind="json", value="branch2_value")}

    store.set_parameters(run_id, root_params, None)
    store.set_parameters(run_id, branch1_params, "branch1")
    store.set_parameters(run_id, branch2_params, "branch2")

    # Verify isolation
    root_result = store.get_parameters(run_id, None)
    branch1_result = store.get_parameters(run_id, "branch1")
    branch2_result = store.get_parameters(run_id, "branch2")

    assert root_result["param"].value == "root_value"
    assert branch1_result["param"].value == "branch1_value"
    assert branch2_result["param"].value == "branch2_value"

    # Changing one should not affect others
    branch1_updated = {"param": JsonParameter(kind="json", value="branch1_updated")}
    store.set_parameters(run_id, branch1_updated, "branch1")

    # Others should be unchanged
    assert store.get_parameters(run_id, None)["param"].value == "root_value"
    assert store.get_parameters(run_id, "branch2")["param"].value == "branch2_value"
    assert store.get_parameters(run_id, "branch1")["param"].value == "branch1_updated"
```

**Step 2: Run integration tests**

Run: `pytest tests/extensions/run_log_store/test_generic_partitioned.py::test_complete_partitioned_workflow -v`
Expected: PASS

**Step 3: Run all tests to verify complete functionality**

Run: `pytest tests/extensions/run_log_store/test_generic_partitioned.py -v`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add tests/extensions/run_log_store/test_generic_partitioned.py
git commit -m "test: add comprehensive integration tests for partitioned storage"
```

---

## Task 7: Final Validation and Cleanup

**Files:**
- Modify: `extensions/run_log_store/generic_partitioned.py`

**Step 1: Add proper docstring and final review**

```python
# Update the class docstring in extensions/run_log_store/generic_partitioned.py

class GenericPartitionedRunLogStore(BaseRunLogStore):
    """
    A generic implementation of a RunLogStore that supports full partitioned storage.

    This implementation provides complete partitioning of all data types (parameters,
    step logs, branch logs) with hierarchical storage structure. Each branch execution
    context gets isolated storage while supporting parameter inheritance from parent contexts.

    Key Features:
    - Full data partitioning per branch execution context
    - Hierarchical storage: root -> branches -> nested branches
    - Parameter inheritance during branch creation
    - Transparent routing based on internal_branch_name
    - Backward compatibility with existing domain code

    Storage Structure:
        /run_id/                    # Root partition
        ├── parameters/             # Root parameters
        ├── step_logs/              # Root step logs
        ├── branch_logs/            # Root branch logs
        └── branches/
            └── branch_a/           # Branch partition
                ├── parameters/     # Branch-specific parameters
                ├── step_logs/      # Branch-specific step logs
                ├── branch_logs/    # Branch-specific branch logs
                └── branches/       # Nested branches...

    Abstract Methods:
        Concrete implementations must provide storage methods for:
        - Root partition: _store_root_*, _retrieve_root_*
        - Branch partition: _store_branch_*, _retrieve_branch_*

        Each method handles one data type (parameters, step_logs, branch_logs)
        in one partition type (root or branch).
    """
```

**Step 2: Run final comprehensive test**

Run: `pytest tests/extensions/run_log_store/test_generic_partitioned.py -v --tb=short`
Expected: All tests PASS with clean output

**Step 3: Run linting and formatting**

Run: `ruff check extensions/run_log_store/generic_partitioned.py --fix`
Run: `ruff format extensions/run_log_store/generic_partitioned.py`
Expected: Clean output, no issues

**Step 4: Final commit**

```bash
git add extensions/run_log_store/generic_partitioned.py
git commit -m "docs: add comprehensive docstring for GenericPartitioned implementation

Completes the refactoring from GenericChunked to GenericPartitioned with:
- Full partitioned storage for all data types
- Hierarchical storage structure
- Parameter inheritance during branch creation
- Transparent routing maintaining backward compatibility
- Abstract methods for concrete implementations"
```

**Step 5: Verify git history is clean**

Run: `git log --oneline -10`
Expected: Clean commit history showing implementation progression

---

## Implementation Complete

The GenericPartitioned RunLogStore refactoring is now complete with:

✅ **Full partitioned storage architecture**
✅ **Abstract methods for root and branch storage**
✅ **Parameter inheritance during branch creation**
✅ **Transparent routing maintaining backward compatibility**
✅ **Comprehensive test coverage**
✅ **Clean separation from chunked approach**

**Next Steps:**
- Concrete implementations (FileSystem, S3, etc.) can now implement the abstract methods
- Domain code requires no changes due to backward-compatible method signatures
- Integration tests can validate end-to-end partitioned workflow
