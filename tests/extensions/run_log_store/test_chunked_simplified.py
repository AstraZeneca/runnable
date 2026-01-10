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
