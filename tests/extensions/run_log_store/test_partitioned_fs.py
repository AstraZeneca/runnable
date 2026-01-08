import tempfile
import pytest
from pathlib import Path

from runnable.datastore import JsonParameter, StepLog, BranchLog
from runnable import defaults
from extensions.run_log_store.partitioned_fs import FileSystemPartitionedRunLogStore


@pytest.fixture
def temp_fs_store():
    """Create a temporary filesystem store for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        store = FileSystemPartitionedRunLogStore()
        store.log_folder = temp_dir
        yield store


def test_concrete_fs_store_instantiation(temp_fs_store):
    """Test that FileSystemPartitionedRunLogStore can be instantiated."""
    assert temp_fs_store.service_name == "partitioned-fs"
    assert temp_fs_store.supports_parallel_writes == True


def test_fs_parameter_storage_and_retrieval(temp_fs_store):
    """Test parameter storage and retrieval in file system."""
    run_id = "test_run"

    # Test root parameters
    root_params = {
        "config": JsonParameter(kind="json", value={"env": "test"}),
        "data": JsonParameter(kind="json", value=[1, 2, 3])
    }

    temp_fs_store.set_parameters(run_id, root_params, None)
    retrieved_root = temp_fs_store.get_parameters(run_id, None)

    assert "config" in retrieved_root
    assert retrieved_root["config"].value == {"env": "test"}
    assert retrieved_root["data"].value == [1, 2, 3]

    # Test branch parameters
    branch_params = {
        "branch_config": JsonParameter(kind="json", value={"branch_id": 1})
    }

    temp_fs_store.set_parameters(run_id, branch_params, "branch1")
    retrieved_branch = temp_fs_store.get_parameters(run_id, "branch1")

    assert "branch_config" in retrieved_branch
    assert retrieved_branch["branch_config"].value == {"branch_id": 1}


def test_fs_parameter_inheritance(temp_fs_store):
    """Test parameter inheritance works with file system."""
    run_id = "test_run"

    # Set up root parameters
    root_params = {
        "shared": JsonParameter(kind="json", value={"shared_data": "root"})
    }
    temp_fs_store.set_parameters(run_id, root_params, None)

    # Copy to branch
    temp_fs_store.copy_parameters_to_branch(run_id, None, "branch1")

    # Verify inheritance
    branch_params = temp_fs_store.get_parameters(run_id, "branch1")
    assert "shared" in branch_params
    assert branch_params["shared"].value == {"shared_data": "root"}


def test_fs_hierarchical_storage_structure(temp_fs_store):
    """Test that the hierarchical folder structure is created correctly."""
    run_id = "test_run"

    # Create root data
    root_params = {"root_param": JsonParameter(kind="json", value="root_value")}
    temp_fs_store.set_parameters(run_id, root_params, None)

    root_step = StepLog(name="root_step", internal_name="root_step", status=defaults.CREATED)
    temp_fs_store.add_step_log(root_step, run_id, None)

    root_branch = BranchLog(internal_name="branch1", status=defaults.CREATED)
    temp_fs_store.add_branch_log(root_branch, run_id, None)

    # Create branch data
    branch_params = {"branch_param": JsonParameter(kind="json", value="branch_value")}
    temp_fs_store.set_parameters(run_id, branch_params, "branch1")

    branch_step = StepLog(name="branch_step", internal_name="branch1.step", status=defaults.CREATED)
    temp_fs_store.add_step_log(branch_step, run_id, "branch1")

    nested_branch = BranchLog(internal_name="nested", status=defaults.CREATED)
    temp_fs_store.add_branch_log(nested_branch, run_id, "branch1")

    # Verify folder structure
    base_path = Path(temp_fs_store.log_folder) / run_id

    # Root structure
    assert (base_path / "parameters").exists()
    assert (base_path / "steps").exists()
    assert (base_path / "branches").exists()

    # Branch structure
    branch_path = base_path / "branch_partitions" / "branch1"
    assert (branch_path / "parameters").exists()
    assert (branch_path / "steps").exists()
    assert (branch_path / "branches").exists()

    # Verify files exist
    assert (base_path / "parameters" / "root_param.json").exists()
    assert (base_path / "steps" / "root_step.json").exists()
    assert (base_path / "branches" / "branch1.json").exists()

    assert (branch_path / "parameters" / "branch_param.json").exists()
    assert (branch_path / "steps" / "branch1.step.json").exists()
    assert (branch_path / "branches" / "nested.json").exists()


def test_fs_nested_branches(temp_fs_store):
    """Test nested branch structure works correctly."""
    run_id = "test_run"

    # Create nested branch data
    nested_params = {"nested_param": JsonParameter(kind="json", value="nested_value")}
    temp_fs_store.set_parameters(run_id, nested_params, "branch1.nested")

    nested_step = StepLog(name="nested_step", internal_name="branch1.nested.step", status=defaults.CREATED)
    temp_fs_store.add_step_log(nested_step, run_id, "branch1.nested")

    # Verify nested structure
    base_path = Path(temp_fs_store.log_folder) / run_id
    nested_path = base_path / "branch_partitions" / "branch1" / "branch_partitions" / "nested"

    assert (nested_path / "parameters").exists()
    assert (nested_path / "steps").exists()
    assert (nested_path / "parameters" / "nested_param.json").exists()
    assert (nested_path / "steps" / "branch1.nested.step.json").exists()

    # Verify retrieval
    retrieved_params = temp_fs_store.get_parameters(run_id, "branch1.nested")
    assert "nested_param" in retrieved_params
    assert retrieved_params["nested_param"].value == "nested_value"

    retrieved_step = temp_fs_store.get_step_log("branch1.nested.step", run_id, "branch1.nested")
    assert retrieved_step.name == "nested_step"


def test_fs_run_log_storage(temp_fs_store):
    """Test basic run log storage."""
    run_id = "test_run"

    # Create and store run log
    run_log = temp_fs_store.create_run_log(run_id, status=defaults.CREATED)
    assert run_log.run_id == run_id

    # Verify storage
    base_path = Path(temp_fs_store.log_folder) / run_id
    assert (base_path / "run_log.json").exists()

    # Verify retrieval
    retrieved_run_log = temp_fs_store.get_run_log_by_id(run_id)
    assert retrieved_run_log.run_id == run_id
    assert retrieved_run_log.status == defaults.CREATED
