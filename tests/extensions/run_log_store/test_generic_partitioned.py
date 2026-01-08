import pytest
from unittest.mock import Mock
from typing import Dict
from runnable.datastore import Parameter, RunLog, StepLog, BranchLog
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
