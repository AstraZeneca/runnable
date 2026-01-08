import pytest
from unittest.mock import Mock
from typing import Dict, Union
from runnable.datastore import Parameter, RunLog, StepLog, BranchLog, JsonParameter
from runnable import defaults
from extensions.run_log_store.generic_partitioned import GenericPartitionedRunLogStore

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


# Test step log management

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


# Test branch log management

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


# Integration tests

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
