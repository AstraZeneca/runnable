import pytest
from extensions.run_log_store.chunked_fs import ChunkedFileSystemRunLogStore
from runnable.datastore import JsonParameter, StepLog, BranchLog


@pytest.fixture
def chunked_store(tmp_path):
    """Create a chunked store with temp directory"""
    store = ChunkedFileSystemRunLogStore(log_folder=str(tmp_path))
    return store


def test_chunked_store_only_has_runlog_and_branchlog_types(chunked_store):
    """Test that chunked store only uses RUN_LOG and BRANCH_LOG types"""
    log_types = [lt.name for lt in chunked_store.LogTypes]
    assert "RUN_LOG" in log_types
    assert "BRANCH_LOG" in log_types
    # These should no longer exist
    assert "STEP_LOG" not in log_types
    assert "PARAMETER" not in log_types


def test_add_step_log_stores_within_runlog(chunked_store):
    """Test that add_step_log stores step within RunLog, not separate file"""
    import os

    run_log = chunked_store.create_run_log(run_id="test_run")

    step = chunked_store.create_step_log(name="step1", internal_name="step1")
    chunked_store.add_step_log(step, "test_run")

    # Verify step is in run_log
    retrieved_run_log = chunked_store.get_run_log_by_id("test_run", full=False)
    assert "step1" in retrieved_run_log.steps

    # Verify no separate StepLog file was created
    run_folder = os.path.join(chunked_store.log_folder, "test_run")
    files = os.listdir(run_folder) if os.path.exists(run_folder) else []
    step_files = [f for f in files if f.startswith("StepLog-")]
    assert len(step_files) == 0


def test_add_step_log_stores_within_branch(chunked_store):
    """Test that add_step_log stores step within BranchLog when in branch"""
    run_log = chunked_store.create_run_log(run_id="test_run")

    # Create parent step and branch
    parent_step = chunked_store.create_step_log(name="parent", internal_name="parent")
    chunked_store.add_step_log(parent_step, "test_run")

    branch = chunked_store.create_branch_log(internal_branch_name="parent.branch1")
    chunked_store.add_branch_log(branch, "test_run")

    # Add step to branch
    nested_step = chunked_store.create_step_log(
        name="nested", internal_name="parent.branch1.nested"
    )
    chunked_store.add_step_log(nested_step, "test_run")

    # Verify step is in branch
    retrieved_branch = chunked_store.get_branch_log("parent.branch1", "test_run")
    assert "parent.branch1.nested" in retrieved_branch.steps
