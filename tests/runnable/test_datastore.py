from unittest.mock import Mock, patch

import pytest

from runnable import exceptions
from runnable.datastore import (
    BranchLog,
    BufferRunLogstore,
    DataCatalog,
    JsonParameter,
    ObjectParameter,
    RunLog,
    StepLog,
)


@pytest.fixture
def mock_context():
    """Fixture to create a mock run context"""
    mock_ctx = Mock()
    mock_ctx.object_serialisation = False
    mock_ctx.pickler = Mock()
    mock_ctx.pickler.extension = ".pkl"
    mock_ctx.return_objects = {}
    mock_ctx.catalog = Mock()
    return mock_ctx


def test_object_parameter_init():
    """Test basic initialization of ObjectParameter"""
    obj_param = ObjectParameter(kind="object", value="test_obj")
    assert obj_param.kind == "object"
    assert obj_param.value == "test_obj"
    assert obj_param.reduced is True


@pytest.mark.parametrize(
    "serialisation,expected",
    [
        (True, "Pickled object stored in catalog as: test_obj"),
        (False, "Object stored in memory as: test_obj"),
    ],
)
def test_object_parameter_description(mock_context, serialisation, expected):
    """Test description property under different serialization settings"""
    with patch("runnable.datastore.context.get_run_context") as mock_get_ctx:
        mock_context.object_serialisation = serialisation
        mock_get_ctx.return_value = mock_context

        obj_param = ObjectParameter(kind="object", value="test_obj")
        assert obj_param.description == expected


def test_object_parameter_file_name(mock_context):
    """Test file_name property"""
    with patch("runnable.datastore.context.get_run_context") as mock_get_ctx:
        mock_get_ctx.return_value = mock_context

        obj_param = ObjectParameter(kind="object", value="test_obj")
        assert obj_param.file_name == "test_obj.pkl"


def test_get_value_without_serialisation(mock_context):
    """Test get_value when object serialisation is disabled"""
    with patch("runnable.datastore.context.get_run_context") as mock_get_ctx:
        mock_context.object_serialisation = False
        mock_context.return_objects = {"test_obj": "test_value"}
        mock_get_ctx.return_value = mock_context

        obj_param = ObjectParameter(kind="object", value="test_obj")
        assert obj_param.get_value() == "test_value"


def test_put_object_without_serialisation(mock_context):
    """Test put_object when object serialisation is disabled"""
    with patch("runnable.datastore.context.get_run_context") as mock_get_ctx:
        mock_context.object_serialisation = False
        mock_get_ctx.return_value = mock_context

        obj_param = ObjectParameter(kind="object", value="test_obj")
        obj_param.put_object("test_value")

        assert mock_context.return_objects["test_obj"] == "test_value"


def test_get_data_catalogs_by_stage_invalid_stage():
    """Test that invalid stage raises exception"""
    step_log = StepLog(name="test_step", internal_name="test_step")

    with pytest.raises(Exception) as exc_info:
        step_log.get_data_catalogs_by_stage(stage="invalid")
    assert str(exc_info.value) == "Stage should be in get or put"


def test_get_data_catalogs_by_stage_empty():
    """Test with empty data catalogs"""
    step_log = StepLog(name="test_step", internal_name="test_step")

    catalogs = step_log.get_data_catalogs_by_stage(stage="put")
    assert len(catalogs) == 0


def test_get_data_catalogs_by_stage_put():
    """Test get_data_catalogs_by_stage with 'put' stage"""
    step_log = StepLog(name="test_step", internal_name="test_step")

    # Add test catalogs
    catalogs = [
        DataCatalog(name="data1", stage="put"),
        DataCatalog(name="data2", stage="get"),
        DataCatalog(name="data3", stage="put"),
    ]
    step_log.data_catalog.extend(catalogs)

    put_catalogs = step_log.get_data_catalogs_by_stage(stage="put")
    assert len(put_catalogs) == 2
    assert all(c.stage == "put" for c in put_catalogs)
    assert {c.name for c in put_catalogs} == {"data1", "data3"}


def test_get_data_catalogs_by_stage_get():
    """Test get_data_catalogs_by_stage with 'get' stage"""
    step_log = StepLog(name="test_step", internal_name="test_step")

    # Add test catalogs
    catalogs = [
        DataCatalog(name="data1", stage="put"),
        DataCatalog(name="data2", stage="get"),
        DataCatalog(name="data3", stage="get"),
    ]
    step_log.data_catalog.extend(catalogs)

    get_catalogs = step_log.get_data_catalogs_by_stage(stage="get")
    assert len(get_catalogs) == 2
    assert all(c.stage == "get" for c in get_catalogs)
    assert {c.name for c in get_catalogs} == {"data2", "data3"}


def test_get_data_catalogs_by_stage_with_branches():
    """Test get_data_catalogs_by_stage with nested branches"""
    step_log = StepLog(name="test_step", internal_name="test_step")

    # Add catalogs to main step
    step_log.data_catalog.extend(
        [
            DataCatalog(name="main_data1", stage="put"),
            DataCatalog(name="main_data2", stage="get"),
        ]
    )

    # Create a branch with its own catalogs
    branch = BranchLog(internal_name="branch1")
    branch_step = StepLog(name="branch_step", internal_name="branch_step")
    branch_step.data_catalog.extend(
        [
            DataCatalog(name="branch_data1", stage="put"),
            DataCatalog(name="branch_data2", stage="put"),
        ]
    )
    branch.steps = {"branch_step": branch_step}

    # Add branch to step
    step_log.branches["branch1"] = branch

    # Test getting all put catalogs
    put_catalogs = step_log.get_data_catalogs_by_stage(stage="put")
    assert len(put_catalogs) == 3
    assert {c.name for c in put_catalogs} == {
        "main_data1",
        "branch_data1",
        "branch_data2",
    }


def test_get_data_catalogs_by_stage_multiple_branches():
    """Test get_data_catalogs_by_stage with multiple branches"""
    step_log = StepLog(name="test_step", internal_name="test_step")

    # Add catalogs to main step
    step_log.data_catalog.extend([DataCatalog(name="main_data", stage="put")])

    # Create multiple branches
    for i in range(2):
        branch = BranchLog(internal_name=f"branch{i}")
        branch_step = StepLog(name=f"branch_step{i}", internal_name=f"branch_step{i}")
        branch_step.data_catalog.extend(
            [
                DataCatalog(name=f"branch{i}_data1", stage="put"),
                DataCatalog(name=f"branch{i}_data2", stage="get"),
            ]
        )
        branch.steps = {f"branch_step{i}": branch_step}
        step_log.branches[f"branch{i}"] = branch

    # Test getting all put catalogs
    put_catalogs = step_log.get_data_catalogs_by_stage(stage="put")
    assert len(put_catalogs) == 3
    assert {c.name for c in put_catalogs} == {
        "main_data",
        "branch0_data1",
        "branch1_data1",
    }

    # Test getting all get catalogs
    get_catalogs = step_log.get_data_catalogs_by_stage(stage="get")
    assert len(get_catalogs) == 2
    assert {c.name for c in get_catalogs} == {"branch0_data2", "branch1_data2"}


def test_data_catalog_init_with_required_fields():
    """Test initializing DataCatalog with only required name field"""
    catalog = DataCatalog(name="test_data")
    assert catalog.name == "test_data"
    assert catalog.data_hash == ""
    assert catalog.catalog_relative_path == ""
    assert catalog.catalog_handler_location == ""
    assert catalog.stage == ""


def test_data_catalog_init_with_all_fields():
    """Test initializing DataCatalog with all fields"""
    catalog = DataCatalog(
        name="test_data",
        data_hash="abc123",
        catalog_relative_path="/path/to/data",
        catalog_handler_location="/root/catalog",
        stage="put",
    )
    assert catalog.name == "test_data"
    assert catalog.data_hash == "abc123"
    assert catalog.catalog_relative_path == "/path/to/data"
    assert catalog.catalog_handler_location == "/root/catalog"
    assert catalog.stage == "put"


def test_data_catalog_hash_based_on_name():
    """Test that hash is based only on name field"""
    catalog1 = DataCatalog(name="test_data", data_hash="abc123", stage="put")
    catalog2 = DataCatalog(name="test_data", data_hash="def456", stage="get")
    # Same name should produce same hash
    assert hash(catalog1) == hash(catalog2)
    assert hash(catalog1) == hash("test_data")


def test_data_catalog_equality_based_on_name():
    """Test equality comparison between DataCatalog objects"""
    catalog1 = DataCatalog(name="test_data", data_hash="abc123", stage="put")
    catalog2 = DataCatalog(name="test_data", data_hash="def456", stage="get")
    # Same name means equal
    assert catalog1 == catalog2


def test_data_catalog_inequality():
    """Test inequality comparison between DataCatalog objects"""
    catalog1 = DataCatalog(name="test_data_1")
    catalog2 = DataCatalog(name="test_data_2")
    assert catalog1 != catalog2


def test_data_catalog_equality_with_non_catalog():
    """Test equality comparison with non-DataCatalog objects"""
    catalog = DataCatalog(name="test_data")
    assert catalog != "test_data"
    assert catalog != {"name": "test_data"}
    assert catalog != 123


def test_data_catalog_allows_extra_fields():
    """Test that extra fields are allowed"""
    catalog = DataCatalog(
        name="test_data", extra_field="extra_value", another_field=123
    )
    assert catalog.extra_field == "extra_value"
    assert catalog.another_field == 123


def test_search_branch_empty_name():
    """Test searching branch with empty name returns the run log itself"""
    run_log = RunLog(run_id="test_run")
    branch, step = run_log.search_branch_by_internal_name("")

    assert branch == run_log
    assert step is None


def test_search_branch_simple_path():
    """Test searching a simple branch path (step.branch)"""
    run_log = RunLog(run_id="test_run")

    # Create a step with a branch
    step = StepLog(name="step1", internal_name="step1")
    branch = BranchLog(internal_name="step1.branch1")
    step.branches["step1.branch1"] = branch
    run_log.steps["step1"] = step

    found_branch, found_step = run_log.search_branch_by_internal_name("step1.branch1")

    assert found_branch == branch
    assert found_step == step


def test_search_branch_nested_path():
    """Test searching a nested branch path (step.branch.step.branch)"""
    run_log = RunLog(run_id="test_run")

    # Create first level
    step1 = StepLog(name="step1", internal_name="step1")
    branch1 = BranchLog(internal_name="step1.branch1")
    step1.branches["step1.branch1"] = branch1
    run_log.steps["step1"] = step1

    # Create second level
    step2 = StepLog(name="step2", internal_name="step1.branch1.step2")
    branch2 = BranchLog(internal_name="step1.branch1.step2.branch2")
    step2.branches["step1.branch1.step2.branch2"] = branch2
    branch1.steps["step1.branch1.step2"] = step2

    found_branch, found_step = run_log.search_branch_by_internal_name(
        "step1.branch1.step2.branch2"
    )

    assert found_branch == branch2
    assert found_step == step2


def test_search_branch_not_found():
    """Test searching for a non-existent branch raises error"""
    run_log = RunLog(run_id="test_run")

    # Create a step without the searched branch
    step = StepLog(name="step1", internal_name="step1")
    run_log.steps["step1"] = step

    with pytest.raises(exceptions.BranchLogNotFoundError) as exc_info:
        run_log.search_branch_by_internal_name("step1.missing_branch")

    assert exc_info.value.run_id == "test_run"
    assert exc_info.value.branch_name == "step1.missing_branch"


def test_search_branch_invalid_path():
    """Test searching with invalid path structure raises error"""
    run_log = RunLog(run_id="test_run")

    # Create a step
    step = StepLog(name="step1", internal_name="step1")
    run_log.steps["step1"] = step

    # Try to search with invalid path (missing step)
    with pytest.raises(exceptions.BranchLogNotFoundError):
        run_log.search_branch_by_internal_name("nonexistent.branch1")


def test_search_branch_multiple_branches():
    """Test searching in a structure with multiple branches"""
    run_log = RunLog(run_id="test_run")

    # Create first branch path
    step1 = StepLog(name="step1", internal_name="step1")
    branch1 = BranchLog(internal_name="step1.branch1")
    step1.branches["step1.branch1"] = branch1
    run_log.steps["step1"] = step1

    # Create parallel branch path
    step2 = StepLog(name="step2", internal_name="step2")
    branch2 = BranchLog(internal_name="step2.branch2")
    step2.branches["step2.branch2"] = branch2
    run_log.steps["step2"] = step2

    # Test finding first branch
    found_branch1, found_step1 = run_log.search_branch_by_internal_name("step1.branch1")
    assert found_branch1 == branch1
    assert found_step1 == step1

    # Test finding second branch
    found_branch2, found_step2 = run_log.search_branch_by_internal_name("step2.branch2")
    assert found_branch2 == branch2
    assert found_step2 == step2


def test_search_step_simple_path():
    """Test searching for a step at the root level"""
    run_log = RunLog(run_id="test_run")

    # Create a simple step
    step = StepLog(name="step1", internal_name="step1")
    run_log.steps["step1"] = step

    found_step, found_branch = run_log.search_step_by_internal_name("step1")
    assert found_step == step
    assert found_branch is None


def test_search_step_nested_path():
    """Test searching for a step inside a branch"""
    run_log = RunLog(run_id="test_run")

    # Create first level
    step1 = StepLog(name="step1", internal_name="step1")
    branch1 = BranchLog(internal_name="step1.branch1")
    step1.branches["step1.branch1"] = branch1
    run_log.steps["step1"] = step1

    # Create nested step
    step2 = StepLog(name="step2", internal_name="step1.branch1.step2")
    branch1.steps["step1.branch1.step2"] = step2

    found_step, found_branch = run_log.search_step_by_internal_name(
        "step1.branch1.step2"
    )
    assert found_step == step2
    assert found_branch == branch1


def test_search_step_deeply_nested():
    """Test searching for a step multiple levels deep"""
    run_log = RunLog(run_id="test_run")

    # Create first level
    step1 = StepLog(name="step1", internal_name="step1")
    branch1 = BranchLog(internal_name="step1.branch1")
    step1.branches["step1.branch1"] = branch1
    run_log.steps["step1"] = step1

    # Create second level
    step2 = StepLog(name="step2", internal_name="step1.branch1.step2")
    branch2 = BranchLog(internal_name="step1.branch1.step2.branch2")
    step2.branches["step1.branch1.step2.branch2"] = branch2
    branch1.steps["step1.branch1.step2"] = step2

    # Create final step
    step3 = StepLog(name="step3", internal_name="step1.branch1.step2.branch2.step3")
    branch2.steps["step1.branch1.step2.branch2.step3"] = step3

    found_step, found_branch = run_log.search_step_by_internal_name(
        "step1.branch1.step2.branch2.step3"
    )
    assert found_step == step3
    assert found_branch == branch2


def test_search_step_not_found():
    """Test searching for a non-existent step"""
    run_log = RunLog(run_id="test_run")

    with pytest.raises(exceptions.StepLogNotFoundError) as exc_info:
        run_log.search_step_by_internal_name("nonexistent.step")

    assert exc_info.value.run_id == "test_run"
    assert exc_info.value.step_name == "nonexistent.step"


def test_search_step_invalid_branch_path():
    """Test searching with invalid branch path"""
    run_log = RunLog(run_id="test_run")

    # Create step but no branch
    step = StepLog(name="step1", internal_name="step1")
    run_log.steps["step1"] = step

    with pytest.raises(exceptions.StepLogNotFoundError) as exc_info:
        run_log.search_step_by_internal_name("step1.nonexistent.step2")

    assert exc_info.value.run_id == "test_run"
    assert exc_info.value.step_name == "step1.nonexistent.step2"


def test_search_step_empty_runlog():
    """Test searching in an empty run log"""
    run_log = RunLog(run_id="test_run")

    with pytest.raises(exceptions.StepLogNotFoundError) as exc_info:
        run_log.search_step_by_internal_name("step1")

    assert exc_info.value.run_id == "test_run"
    assert exc_info.value.step_name == "step1"


def test_add_step_log_basic():
    """Test adding a basic step log to run log"""
    # Create a buffer run log store
    store = BufferRunLogstore(service_name="test_store")
    run_log = store.create_run_log(run_id="test_run")

    # Create a simple step log
    step_log = StepLog(name="step1", internal_name="step1")

    # Add step log
    store.add_step_log(step_log, "test_run")

    # Verify step was added correctly
    assert "step1" in store.run_log.steps
    assert store.run_log.steps["step1"] == step_log


def test_add_step_log_nested():
    """Test adding a step log within a branch"""
    store = BufferRunLogstore(service_name="test_store")
    run_log = store.create_run_log(run_id="test_run")

    # Create parent step and branch
    parent_step = StepLog(name="parent", internal_name="parent")
    parent_branch = BranchLog(internal_name="parent.branch1")
    parent_step.branches["parent.branch1"] = parent_branch
    store.run_log.steps["parent"] = parent_step

    # Create nested step
    nested_step = StepLog(name="nested", internal_name="parent.branch1.nested")

    # Add nested step
    store.add_step_log(nested_step, "test_run")

    # Verify nested step was added correctly
    assert "parent.branch1.nested" in parent_branch.steps
    assert parent_branch.steps["parent.branch1.nested"] == nested_step


def test_add_step_log_run_not_found():
    """Test adding step log to non-existent run"""
    store = BufferRunLogstore(service_name="test_store")
    step_log = StepLog(name="step1", internal_name="step1")

    with pytest.raises(exceptions.RunLogNotFoundError) as exc_info:
        store.add_step_log(step_log, "nonexistent_run")

    assert exc_info.value.run_id == "nonexistent_run"


def test_add_step_log_branch_not_found():
    """Test adding step log to non-existent branch"""
    store = BufferRunLogstore(service_name="test_store")
    run_log = store.create_run_log(run_id="test_run")

    # Create step with non-existent parent branch
    step_log = StepLog(name="step1", internal_name="nonexistent.branch.step1")

    with pytest.raises(exceptions.BranchLogNotFoundError) as exc_info:
        store.add_step_log(step_log, "test_run")

    assert exc_info.value.run_id == "test_run"
    assert exc_info.value.branch_name == "nonexistent.branch"


def test_add_step_log_multiple_levels():
    """Test adding step log in deeply nested structure"""
    store = BufferRunLogstore(service_name="test_store")
    run_log = store.create_run_log(run_id="test_run")

    # Create multi-level structure
    level1_step = StepLog(name="level1", internal_name="level1")
    level1_branch = BranchLog(internal_name="level1.branch1")
    level1_step.branches["level1.branch1"] = level1_branch

    level2_step = StepLog(name="level2", internal_name="level1.branch1.level2")
    level2_branch = BranchLog(internal_name="level1.branch1.level2.branch2")
    level2_step.branches["level1.branch1.level2.branch2"] = level2_branch

    # Add structure to run log
    store.run_log.steps["level1"] = level1_step
    level1_branch.steps["level1.branch1.level2"] = level2_step

    # Create and add deeply nested step
    nested_step = StepLog(
        name="nested", internal_name="level1.branch1.level2.branch2.nested"
    )
    store.add_step_log(nested_step, "test_run")

    # Verify deep nesting
    assert "level1.branch1.level2.branch2.nested" in level2_branch.steps
    assert level2_branch.steps["level1.branch1.level2.branch2.nested"] == nested_step


def test_add_step_log_replace_existing():
    """Test replacing an existing step log"""
    store = BufferRunLogstore(service_name="test_store")
    run_log = store.create_run_log(run_id="test_run")

    # Add initial step
    initial_step = StepLog(name="step1", internal_name="step1", status="CREATED")
    store.add_step_log(initial_step, "test_run")

    # Replace with new step
    replacement_step = StepLog(name="step1", internal_name="step1", status="COMPLETED")
    store.add_step_log(replacement_step, "test_run")

    # Verify replacement
    assert store.run_log.steps["step1"].status == "COMPLETED"


def test_get_branch_log_empty_name():
    """Test getting branch with empty name returns the run log"""
    store = BufferRunLogstore(service_name="test_store")
    run_log = store.create_run_log(run_id="test_run")

    branch = store.get_branch_log("", "test_run")
    assert branch == run_log


def test_get_branch_log_simple_branch():
    """Test getting a simple branch"""
    store = BufferRunLogstore(service_name="test_store")
    run_log = store.create_run_log(run_id="test_run")

    # Create step with branch
    step = StepLog(name="step1", internal_name="step1")
    branch = BranchLog(internal_name="step1.branch1")
    step.branches["step1.branch1"] = branch
    run_log.steps["step1"] = step

    found_branch = store.get_branch_log("step1.branch1", "test_run")
    assert found_branch == branch
    assert found_branch.internal_name == "step1.branch1"


def test_get_branch_log_nested():
    """Test getting a nested branch"""
    store = BufferRunLogstore(service_name="test_store")
    run_log = store.create_run_log(run_id="test_run")

    # Create first level
    step1 = StepLog(name="step1", internal_name="step1")
    branch1 = BranchLog(internal_name="step1.branch1")
    step1.branches["step1.branch1"] = branch1
    run_log.steps["step1"] = step1

    # Create second level
    step2 = StepLog(name="step2", internal_name="step1.branch1.step2")
    branch2 = BranchLog(internal_name="step1.branch1.step2.branch2")
    step2.branches["step1.branch1.step2.branch2"] = branch2
    branch1.steps["step1.branch1.step2"] = step2

    found_branch = store.get_branch_log("step1.branch1.step2.branch2", "test_run")
    assert found_branch == branch2
    assert found_branch.internal_name == "step1.branch1.step2.branch2"


def test_get_branch_log_nonexistent():
    """Test getting a non-existent branch raises error"""
    store = BufferRunLogstore(service_name="test_store")
    store.create_run_log(run_id="test_run")

    with pytest.raises(exceptions.BranchLogNotFoundError) as exc_info:
        store.get_branch_log("nonexistent.branch", "test_run")

    assert exc_info.value.run_id == "test_run"
    assert exc_info.value.branch_name == "nonexistent.branch"


def test_get_branch_log_invalid_run_id():
    """Test getting branch with invalid run ID raises error"""
    store = BufferRunLogstore(service_name="test_store")

    with pytest.raises(exceptions.RunLogNotFoundError) as exc_info:
        store.get_branch_log("branch1", "invalid_run")

    assert exc_info.value.run_id == "invalid_run"


def test_get_branch_log_multiple_branches():
    """Test getting branches from a structure with multiple branches"""
    store = BufferRunLogstore(service_name="test_store")
    run_log = store.create_run_log(run_id="test_run")

    # Create parallel branches
    step1 = StepLog(name="step1", internal_name="step1")
    branch1 = BranchLog(internal_name="step1.branch1")
    step1.branches["step1.branch1"] = branch1
    run_log.steps["step1"] = step1

    step2 = StepLog(name="step2", internal_name="step2")
    branch2 = BranchLog(internal_name="step2.branch2")
    step2.branches["step2.branch2"] = branch2
    run_log.steps["step2"] = step2

    # Test getting both branches
    found_branch1 = store.get_branch_log("step1.branch1", "test_run")
    found_branch2 = store.get_branch_log("step2.branch2", "test_run")

    assert found_branch1 == branch1
    assert found_branch2 == branch2
    assert found_branch1.internal_name == "step1.branch1"
    assert found_branch2.internal_name == "step2.branch2"


def test_add_branch_log_simple():
    """Test adding a simple branch log"""
    store = BufferRunLogstore(service_name="test_store")
    run_log = store.create_run_log(run_id="test_run")

    # Create step that will contain branch
    step = StepLog(name="step1", internal_name="step1")
    run_log.steps["step1"] = step

    # Create and add branch
    branch = BranchLog(internal_name="step1.branch1")
    store.add_branch_log(branch, "test_run")

    # Verify branch was added correctly
    assert "step1.branch1" in step.branches
    assert step.branches["step1.branch1"] == branch


def test_add_branch_log_runlog():
    """Test adding a RunLog as branch (base DAG case)"""
    store = BufferRunLogstore(service_name="test_store")
    run_log = RunLog(run_id="test_run")

    # Add RunLog as branch
    store.add_branch_log(run_log, "test_run")

    # Verify RunLog was stored
    assert store.run_log == run_log


def test_add_branch_log_nested():
    """Test adding a branch in nested structure"""
    store = BufferRunLogstore(service_name="test_store")
    run_log = store.create_run_log(run_id="test_run")

    # Create first level
    step1 = StepLog(name="step1", internal_name="step1")
    branch1 = BranchLog(internal_name="step1.branch1")
    step1.branches["step1.branch1"] = branch1
    run_log.steps["step1"] = step1

    # Create second level step
    step2 = StepLog(name="step2", internal_name="step1.branch1.step2")
    branch1.steps["step1.branch1.step2"] = step2

    # Create and add nested branch
    nested_branch = BranchLog(internal_name="step1.branch1.step2.branch2")
    store.add_branch_log(nested_branch, "test_run")

    # Verify nested branch was added correctly
    assert "step1.branch1.step2.branch2" in step2.branches
    assert step2.branches["step1.branch1.step2.branch2"] == nested_branch


def test_add_branch_log_nonexistent_step():
    """Test adding branch to non-existent step raises error"""
    store = BufferRunLogstore(service_name="test_store")
    store.create_run_log(run_id="test_run")

    # Try to add branch to non-existent step
    branch = BranchLog(internal_name="nonexistent.branch1")

    with pytest.raises(exceptions.StepLogNotFoundError) as exc_info:
        store.add_branch_log(branch, "test_run")

    assert exc_info.value.run_id == "test_run"
    assert exc_info.value.step_name == "nonexistent"


def test_add_branch_log_invalid_run():
    """Test adding branch to non-existent run raises error"""
    store = BufferRunLogstore(service_name="test_store")
    branch = BranchLog(internal_name="step1.branch1")

    with pytest.raises(exceptions.RunLogNotFoundError) as exc_info:
        store.add_branch_log(branch, "nonexistent_run")

    assert exc_info.value.run_id == "nonexistent_run"


def test_add_branch_log_multiple_branches():
    """Test adding multiple branches to same step"""
    store = BufferRunLogstore(service_name="test_store")
    run_log = store.create_run_log(run_id="test_run")

    # Create step that will contain branches
    step = StepLog(name="step1", internal_name="step1")
    run_log.steps["step1"] = step

    # Create and add multiple branches
    branch1 = BranchLog(internal_name="step1.branch1")
    branch2 = BranchLog(internal_name="step1.branch2")

    store.add_branch_log(branch1, "test_run")
    store.add_branch_log(branch2, "test_run")

    # Verify both branches were added correctly
    assert "step1.branch1" in step.branches
    assert "step1.branch2" in step.branches
    assert step.branches["step1.branch1"] == branch1
    assert step.branches["step1.branch2"] == branch2


def test_branch_log_has_parameters() -> None:
    """Test that BranchLog has a parameters field"""
    branch = BranchLog(internal_name="step1.branch1")
    assert hasattr(branch, "parameters")
    assert branch.parameters == {}

    # Test setting parameters
    branch.parameters["test_param"] = JsonParameter(kind="json", value="test_value")
    assert "test_param" in branch.parameters
    assert branch.parameters["test_param"].get_value() == "test_value"


def test_get_parameters_with_internal_branch_name() -> None:
    """Test get_parameters accepts optional internal_branch_name parameter"""
    store = BufferRunLogstore(service_name="test_store")
    run_log = store.create_run_log(run_id="test_run")

    # Create step with branch that has parameters
    step = StepLog(name="step1", internal_name="step1")
    branch = BranchLog(internal_name="step1.branch1")
    branch.parameters["branch_param"] = JsonParameter(kind="json", value="branch_value")
    step.branches["step1.branch1"] = branch
    run_log.steps["step1"] = step

    # Set root-level parameter
    store.set_parameters(run_id="test_run", parameters={
        "root_param": JsonParameter(kind="json", value="root_value")
    })

    # Get root parameters (no internal_branch_name)
    root_params = store.get_parameters(run_id="test_run")
    assert "root_param" in root_params

    # Get branch parameters (with internal_branch_name)
    branch_params = store.get_parameters(run_id="test_run", internal_branch_name="step1.branch1")
    assert "branch_param" in branch_params
    assert branch_params["branch_param"].get_value() == "branch_value"


def test_set_parameters_with_internal_branch_name() -> None:
    """Test set_parameters accepts optional internal_branch_name parameter"""
    store = BufferRunLogstore(service_name="test_store")
    run_log = store.create_run_log(run_id="test_run")

    # Create step with branch
    step = StepLog(name="step1", internal_name="step1")
    branch = BranchLog(internal_name="step1.branch1")
    step.branches["step1.branch1"] = branch
    run_log.steps["step1"] = step

    # Set branch-scoped parameters
    store.set_parameters(
        run_id="test_run",
        parameters={"branch_param": JsonParameter(kind="json", value="branch_value")},
        internal_branch_name="step1.branch1"
    )

    # Verify parameters are on the branch, not root
    assert "branch_param" not in run_log.parameters
    assert "branch_param" in branch.parameters
    assert branch.parameters["branch_param"].get_value() == "branch_value"


def test_create_branch_log_with_parameters() -> None:
    """Test create_branch_log can accept initial parameters"""
    store = BufferRunLogstore(service_name="test_store")

    initial_params = {
        "param1": JsonParameter(kind="json", value="value1"),
        "param2": JsonParameter(kind="json", value="value2"),
    }

    branch = store.create_branch_log(
        internal_branch_name="step1.branch1",
        parameters=initial_params
    )

    assert branch.internal_name == "step1.branch1"
    assert "param1" in branch.parameters
    assert "param2" in branch.parameters
    assert branch.parameters["param1"].get_value() == "value1"
