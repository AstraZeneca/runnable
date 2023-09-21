import pytest
from pydantic import BaseModel, Extra

from magnus import defaults, exceptions
from magnus.extensions.executor import DefaultExecutor
from magnus.extensions import executor
import magnus.extensions.executor as executor


@pytest.fixture(autouse=True)
def instantiable_base_class(monkeypatch, mocker):
    monkeypatch.setattr(DefaultExecutor, "__abstractmethods__", set())
    yield


@pytest.fixture
def mock_run_context(mocker, monkeypatch):
    mock_run_context = mocker.Mock()
    monkeypatch.setattr(executor.context, "run_context", mock_run_context)
    return mock_run_context


def test_get_parameters_gets_parameters_from_parameters_file(mocker, monkeypatch, mock_run_context):
    mock_run_context.parameters_file = "parameters_file"
    mock_load_yaml = mocker.MagicMock(return_value={"executor": "test"})
    monkeypatch.setattr(executor.utils, "load_yaml", mock_load_yaml)

    test_executor = DefaultExecutor()
    assert test_executor._get_parameters() == {"executor": "test"}
    mock_load_yaml.assert_called_once_with("parameters_file")


def test_get_parameters_gets_parameters_from_user_parameters(mocker, monkeypatch, mock_run_context):
    mock_run_context.parameters_file = ""
    monkeypatch.setattr(executor.utils, "get_user_set_parameters", mocker.MagicMock(return_value={"executor": "test"}))

    test_executor = DefaultExecutor()
    assert test_executor._get_parameters() == {"executor": "test"}


def test_get_parameters_user_parameters_overwrites_parameters_from_parameters_file(
    mocker, monkeypatch, mock_run_context
):
    mock_run_context.parameters_file = "parameters_file"

    mock_load_yaml = mocker.MagicMock(return_value={"executor": "this"})
    monkeypatch.setattr(executor.utils, "load_yaml", mock_load_yaml)
    monkeypatch.setattr(executor.utils, "get_user_set_parameters", mocker.MagicMock(return_value={"executor": "that"}))

    test_executor = DefaultExecutor()
    assert test_executor._get_parameters() == {"executor": "that"}


def test_set_up_for_rerun_throws_exception_if_run_log_not_exists(mocker, monkeypatch, mock_run_context):
    mock_run_log_store = mocker.MagicMock()

    mock_run_context.run_log_store = mock_run_log_store
    mock_run_context.original_run_id = "original_run_id"
    mock_run_log_store.get_run_log_by_id = mocker.MagicMock(side_effect=exceptions.RunLogNotFoundError("test"))

    with pytest.raises(Exception, match="Expected a run log with id: original_run_id"):
        DefaultExecutor()._set_up_for_re_run(parameters={})


def test_set_up_for_re_run_syncs_catalog_and_parameters(mocker, monkeypatch, mock_run_context):
    mock_catalog_handler_sync_between_runs = mocker.MagicMock()
    mock_catalog_handler = mocker.MagicMock()
    mock_catalog_handler.sync_between_runs = mock_catalog_handler_sync_between_runs

    mock_run_context.catalog_handler = mock_catalog_handler
    mock_run_context.run_id = "run_id"
    mock_run_context.original_run_id = "original_run_id"

    mock_attempt_run_log = mocker.MagicMock()
    mock_attempt_run_log.parameters = {"ghost": "from past"}

    mock_run_log_store = mocker.MagicMock()
    mock_run_log_store.get_run_log_by_id.return_value = mock_attempt_run_log
    mock_run_context.run_log_store = mock_run_log_store

    parameters = {}
    DefaultExecutor()._set_up_for_re_run(parameters=parameters)

    mock_catalog_handler_sync_between_runs.assert_called_once_with(previous_run_id="original_run_id", run_id="run_id")
    assert parameters == {"ghost": "from past"}


def test_set_up_for_re_run_syncs_catalog_and_updates_parameters(mocker, monkeypatch, mock_run_context):
    mock_catalog_handler_sync_between_runs = mocker.MagicMock()
    mock_catalog_handler = mocker.MagicMock()
    mock_catalog_handler.sync_between_runs = mock_catalog_handler_sync_between_runs

    mock_run_context.catalog_handler = mock_catalog_handler
    mock_run_context.run_id = "run_id"
    mock_run_context.original_run_id = "original_run_id"

    mock_attempt_run_log = mocker.MagicMock()
    mock_attempt_run_log.parameters = {"ghost": "from past"}

    mock_run_log_store = mocker.MagicMock()
    mock_run_log_store.get_run_log_by_id.return_value = mock_attempt_run_log
    mock_run_context.run_log_store = mock_run_log_store

    parameters = {"present": "now"}
    DefaultExecutor()._set_up_for_re_run(parameters=parameters)

    mock_catalog_handler_sync_between_runs.assert_called_once_with(previous_run_id="original_run_id", run_id="run_id")
    assert parameters == {"present": "now", "ghost": "from past"}


def test_set_up_run_log_throws_exception_if_run_log_already_exists(mocker, monkeypatch, mock_run_context):
    mock_run_log_store = mocker.MagicMock()

    mock_run_log_store.get_run_log_by_id = mocker.MagicMock(side_effect=exceptions.RunLogExistsError)

    with pytest.raises(exceptions.RunLogExistsError):
        DefaultExecutor()._set_up_run_log()


def test_set_up_run_log_exists_ok_returns_without_exception(mocker, monkeypatch, mock_run_context):
    DefaultExecutor()._set_up_run_log(exists_ok=True)


def test_set_up_run_log_calls_get_parameters(mocker, monkeypatch, mock_run_context):
    mock_get_parameters = mocker.MagicMock()
    monkeypatch.setattr(DefaultExecutor, "_get_parameters", mock_get_parameters)

    mock_run_context.run_log_store.get_run_log_by_id = mocker.MagicMock(
        side_effect=exceptions.RunLogNotFoundError("test")
    )
    mock_run_context.use_cached = False

    DefaultExecutor()._set_up_run_log()

    assert mock_get_parameters.call_count == 1


def test_set_up_run_log_calls_set_up_for_re_run(mocker, monkeypatch, mock_run_context):
    mock_set_up_for_re_run = mocker.MagicMock()
    monkeypatch.setattr(DefaultExecutor, "_set_up_for_re_run", mock_set_up_for_re_run)

    mock_get_parameters = mocker.MagicMock()
    monkeypatch.setattr(DefaultExecutor, "_get_parameters", mock_get_parameters)

    mock_run_context.run_log_store.get_run_log_by_id = mocker.MagicMock(
        side_effect=exceptions.RunLogNotFoundError("test")
    )

    DefaultExecutor()._set_up_run_log()

    assert mock_set_up_for_re_run.call_count == 1


def test_set_up_run_log_calls_create_run_log(mocker, monkeypatch, mock_run_context):
    mock_get_parameters = mocker.MagicMock()
    monkeypatch.setattr(DefaultExecutor, "_get_parameters", mock_get_parameters)

    mock_run_context.run_log_store.get_run_log_by_id = mocker.MagicMock(
        side_effect=exceptions.RunLogNotFoundError("test")
    )

    mock_create_run_log = mocker.MagicMock()
    mock_run_context.run_log_store.create_run_log = mock_create_run_log

    mock_run_context.run_id = "test"
    mock_run_context.tag = "tag"
    mock_run_context.dag_hash = "dag_hash"
    mock_run_context.use_cached = False
    mock_run_context.original_run_id = "original_run_id"

    DefaultExecutor()._set_up_run_log()

    mock_create_run_log.assert_called_once_with(
        run_id="test",
        tag="tag",
        status=defaults.PROCESSING,
        dag_hash="dag_hash",
        use_cached=False,
        original_run_id="original_run_id",
    )


def test_set_up_run_log_store_calls_set_parameters(mocker, monkeypatch, mock_run_context):
    mock_get_parameters = mocker.MagicMock()
    monkeypatch.setattr(DefaultExecutor, "_get_parameters", mock_get_parameters)

    mock_run_context.run_log_store.get_run_log_by_id = mocker.MagicMock(
        side_effect=exceptions.RunLogNotFoundError("test")
    )

    mock_run_context.use_cached = False
    mock_set_parameters = mocker.MagicMock()
    mock_run_context.run_log_store.set_parameters = mock_set_parameters

    DefaultExecutor()._set_up_run_log()

    assert mock_set_parameters.call_count == 1


def test_set_up_run_log_store_calls_set_run_config(mocker, monkeypatch, mock_run_context):
    mock_get_parameters = mocker.MagicMock()
    monkeypatch.setattr(DefaultExecutor, "_get_parameters", mock_get_parameters)

    mock_run_context.run_log_store.get_run_log_by_id = mocker.MagicMock(
        side_effect=exceptions.RunLogNotFoundError("test")
    )

    mock_run_context.use_cached = False
    mock_set_run_config = mocker.MagicMock()
    mock_run_context.run_log_store.set_parameters = mock_set_run_config

    DefaultExecutor()._set_up_run_log()

    assert mock_set_run_config.call_count == 1


def test_base_executor_prepare_for_graph_execution_calls(mocker, monkeypatch, mock_run_context):
    mock_integration = mocker.MagicMock()
    mock_validate = mocker.MagicMock()
    mock_configure_for_traversal = mocker.MagicMock()

    mock_integration.validate = mock_validate
    mock_integration.configure_for_traversal = mock_configure_for_traversal

    mock_set_up_run_log = mocker.MagicMock()
    monkeypatch.setattr(DefaultExecutor, "_set_up_run_log", mock_set_up_run_log)

    monkeypatch.setattr(executor, "integration", mock_integration)
    monkeypatch.setattr(executor.BaseExecutor, "_set_up_run_log", mocker.MagicMock())

    base_executor = DefaultExecutor()

    base_executor.prepare_for_graph_execution()

    assert mock_configure_for_traversal.call_count == 4
    assert mock_validate.call_count == 4


def test_base_execution_prepare_for_node_calls(mocker, monkeypatch, mock_run_context):
    mock_integration = mocker.MagicMock()
    mock_validate = mocker.MagicMock()
    mock_configure_for_execution = mocker.MagicMock()

    mock_integration.validate = mock_validate
    mock_integration.configure_for_execution = mock_configure_for_execution

    mock_set_up_run_log = mocker.MagicMock()
    monkeypatch.setattr(DefaultExecutor, "_set_up_run_log", mock_set_up_run_log)

    monkeypatch.setattr(executor, "integration", mock_integration)

    base_executor = DefaultExecutor()

    base_executor.prepare_for_node_execution()

    assert mock_configure_for_execution.call_count == 4
    assert mock_validate.call_count == 4


def test_base_executor__sync_catalog_raises_exception_if_stage_not_in_get_or_put(mocker, monkeypatch):
    test_executor = DefaultExecutor()
    with pytest.raises(Exception):
        test_executor._sync_catalog(step_log="test", stage="puts")


def test_sync_catalog_does_nothing_for_terminal_node(mocker, monkeypatch, mock_run_context):
    mock_node = mocker.MagicMock()
    mock_node._get_catalog_settings = mocker.MagicMock(side_effect=exceptions.TerminalNodeError)

    test_executor = DefaultExecutor()
    test_executor._context_node = mock_node

    test_executor._sync_catalog("test", stage="get")


def test_sync_catalog_does_nothing_for_no_catalog_settings(mocker, monkeypatch, mock_run_context):
    mock_node = mocker.MagicMock()
    mock_node._get_catalog_settings = mocker.MagicMock(return_value={})

    test_executor = DefaultExecutor()
    test_executor._context_node = mock_node

    test_executor._sync_catalog("test", stage="get")


def test_sync_catalog_does_nothing_for_catalog_settings_stage_not_in(mocker, monkeypatch, mock_run_context):
    mock_node = mocker.MagicMock()
    mock_node._get_catalog_settings = mocker.MagicMock(return_value={"get": "something"})

    test_executor = DefaultExecutor()
    test_executor._context_node = mock_node

    test_executor._sync_catalog("test", stage="put")


def test_sync_catalog_returns_nothing_if_no_syncing_for_node(mocker, monkeypatch, mock_run_context):
    mock_node = mocker.MagicMock()

    mock_node._get_catalog_settings.return_value = None

    test_executor = DefaultExecutor()
    test_executor._context_node = mock_node

    assert test_executor._sync_catalog("test", stage="get") is None


def test_sync_catalog_returns_empty_list_if_empty_catalog(mocker, monkeypatch, mock_run_context):
    pass


def test_base_executor_add_code_identities_adds_git_identity(mocker, monkeypatch):
    mock_step_log = mocker.MagicMock()

    mock_step_log.code_identities = []

    mock_utils_get_git_code_id = mocker.MagicMock(return_value="code id")
    monkeypatch.setattr(executor.utils, "get_git_code_identity", mock_utils_get_git_code_id)

    base_executor = executor.BaseExecutor()

    base_executor.add_code_identities(node=None, step_log=mock_step_log)

    assert mock_step_log.code_identities == ["code id"]


def test_base_executor_execute_from_graph_executes_node_for_success_or_fail(mocker, monkeypatch):
    mock_node = mocker.MagicMock()
    mock_run_log_store = mocker.MagicMock()
    mock__execute_node = mocker.MagicMock()

    monkeypatch.setattr(executor.BaseExecutor, "add_code_identities", mocker.MagicMock())
    monkeypatch.setattr(executor.BaseExecutor, "_execute_node", mock__execute_node)

    mock_node.node_type = "success"
    base_executor = executor.BaseExecutor()
    base_executor.run_log_store = mock_run_log_store

    base_executor.execute_from_graph(node=mock_node, map_variable=None)

    assert mock__execute_node.call_count == 1

    mock_node.reset_mock()
    mock__execute_node.reset_mock()

    mock_node.node_type = "fail"
    base_executor.execute_from_graph(node=mock_node, map_variable=None)

    assert mock__execute_node.call_count == 1


def test_base_executor_execute_from_graph_makes_step_log_processing(mocker, monkeypatch):
    mock_node = mocker.MagicMock()
    mock_run_log_store = mocker.MagicMock()
    mock__execute_node = mocker.MagicMock()
    mock_step_log = mocker.MagicMock()

    mock_run_log_store.create_step_log.return_value = mock_step_log

    monkeypatch.setattr(executor.BaseExecutor, "add_code_identities", mocker.MagicMock())
    monkeypatch.setattr(executor.BaseExecutor, "_execute_node", mock__execute_node)

    mock_node.node_type = "success"
    base_executor = executor.BaseExecutor()
    base_executor.run_log_store = mock_run_log_store

    base_executor.execute_from_graph(node=mock_node, map_variable=None)

    assert mock_step_log.status == defaults.PROCESSING


def test_base_executor_execute_from_graph_makes_step_log_success_if_previous_run_log_success(mocker, monkeypatch):
    mock_node = mocker.MagicMock()
    mock_run_log_store = mocker.MagicMock()
    mock_step_log = mocker.MagicMock()

    mock_run_log_store.create_step_log.return_value = mock_step_log

    monkeypatch.setattr(executor.BaseExecutor, "add_code_identities", mocker.MagicMock())
    monkeypatch.setattr(executor.BaseExecutor, "_is_eligible_for_rerun", mocker.MagicMock(return_value=False))

    base_executor = executor.BaseExecutor()
    base_executor.run_log_store = mock_run_log_store

    base_executor.execute_from_graph(node=mock_node, map_variable=None)

    assert mock_step_log.status == defaults.SUCCESS


def test_base_executor_execute_from_graph_delegates_to_execute_as_graph_for_composite_nodes(mocker, monkeypatch):
    mock_node = mocker.MagicMock()
    mock_run_log_store = mocker.MagicMock()
    mock_step_log = mocker.MagicMock()
    mock_node_execute_as_graph = mocker.MagicMock()

    mock_run_log_store.create_step_log.return_value = mock_step_log

    mock_node.node_type = "parallel"
    mock_node.execute_as_graph = mock_node_execute_as_graph
    monkeypatch.setattr(executor.BaseExecutor, "add_code_identities", mocker.MagicMock())

    base_executor = executor.BaseExecutor()
    base_executor.run_log_store = mock_run_log_store

    base_executor.execute_from_graph(node=mock_node, map_variable=None)

    assert mock_node_execute_as_graph.call_count == 1
    assert mock_step_log.status == defaults.PROCESSING

    mock_node_execute_as_graph.reset_mock()
    mock_step_log.reset_mock()

    mock_node.node_type = "dag"
    base_executor.execute_from_graph(node=mock_node, map_variable=None)

    assert mock_node_execute_as_graph.call_count == 1
    assert mock_step_log.status == defaults.PROCESSING

    mock_node_execute_as_graph.reset_mock()
    mock_step_log.reset_mock()

    mock_node.node_type = "map"
    base_executor.execute_from_graph(node=mock_node, map_variable=None)

    assert mock_node_execute_as_graph.call_count == 1
    assert mock_step_log.status == defaults.PROCESSING


def test_base_executor_execute_from_graph_triggers_job_for_simple_nodes(mocker, monkeypatch):
    mock_node = mocker.MagicMock()
    mock_run_log_store = mocker.MagicMock()
    mock_step_log = mocker.MagicMock()
    mock_trigger_job = mocker.MagicMock()

    mock_run_log_store.create_step_log.return_value = mock_step_log

    mock_node.node_type = "task"
    mock_node.is_composite = False
    monkeypatch.setattr(executor.BaseExecutor, "add_code_identities", mocker.MagicMock())
    monkeypatch.setattr(executor.BaseExecutor, "trigger_job", mock_trigger_job)

    base_executor = executor.BaseExecutor()
    base_executor.run_log_store = mock_run_log_store

    base_executor.execute_from_graph(node=mock_node, map_variable=None)

    assert mock_trigger_job.call_count == 1
    assert mock_step_log.status == defaults.PROCESSING


def test_base_executor__execute_node_calls_catalog(mocker, monkeypatch):
    mock_node = mocker.MagicMock()
    mock_node._get_max_attempts.return_value = 1

    mock_run_log_store = mocker.MagicMock()
    mock_step_catalog = mocker.MagicMock()
    mock__sync_catalog = mocker.MagicMock()
    mock__sync_catalog.return_value = "data_catalogs_get"

    monkeypatch.setattr(executor, "interaction", mocker.MagicMock())
    monkeypatch.setattr(executor, "utils", mocker.MagicMock())
    monkeypatch.setattr(executor.BaseExecutor, "_sync_catalog", mock__sync_catalog)

    mock_run_log_store.get_step_log.return_value = mock_step_catalog
    base_executor = executor.BaseExecutor()
    base_executor.run_log_store = mock_run_log_store

    base_executor._execute_node(node=mock_node)

    mock__sync_catalog.assert_any_call(mock_node, mock_step_catalog, stage="get")
    mock__sync_catalog.assert_any_call(mock_node, mock_step_catalog, stage="put", synced_catalogs="data_catalogs_get")


def test_base_executor_sets_step_log_to_success_if_node_succeeds(mocker, monkeypatch):
    mock_node = mocker.MagicMock()
    mock_node._get_max_attempts.return_value = 1

    mock_run_log_store = mocker.MagicMock()
    mock_step_catalog = mocker.MagicMock()
    mock__sync_catalog = mocker.MagicMock()
    mock__sync_catalog.return_value = "data_catalogs_get"

    monkeypatch.setattr(executor, "interaction", mocker.MagicMock())
    monkeypatch.setattr(executor, "utils", mocker.MagicMock())
    monkeypatch.setattr(executor.BaseExecutor, "_sync_catalog", mock__sync_catalog)

    mock_run_log_store.get_step_log.return_value = mock_step_catalog
    base_executor = executor.BaseExecutor()
    base_executor.run_log_store = mock_run_log_store

    base_executor._execute_node(node=mock_node)

    assert mock_step_catalog.status == defaults.SUCCESS


def test_base_executor_sets_status_to_fail_if_attempt_log_is_fail(monkeypatch, mocker):
    mock_node = mocker.MagicMock()
    mock_node._get_max_attempts.return_value = 1
    mock_attempt_log = mocker.MagicMock()
    mock_node.execute.return_value = mock_attempt_log
    mock_attempt_log.status = defaults.FAIL

    mock_run_log_store = mocker.MagicMock()
    mock_step_catalog = mocker.MagicMock()
    mock__sync_catalog = mocker.MagicMock()

    monkeypatch.setattr(executor, "interaction", mocker.MagicMock())
    monkeypatch.setattr(executor, "utils", mocker.MagicMock())
    monkeypatch.setattr(executor.BaseExecutor, "_sync_catalog", mock__sync_catalog)

    mock_run_log_store.get_step_log.return_value = mock_step_catalog
    base_executor = executor.BaseExecutor()
    base_executor.run_log_store = mock_run_log_store

    base_executor._execute_node(node=mock_node)

    assert mock_step_catalog.status == defaults.FAIL


def test_base_executor__get_status_and_next_node_name_gets_next_if_success(mocker, monkeypatch):
    mock_node = mocker.MagicMock()
    mock_run_log_store = mocker.MagicMock()
    mock_dag = mocker.MagicMock()
    mock_step_log = mocker.MagicMock()

    mock_run_log_store.get_step_log.return_value = mock_step_log
    mock_step_log.status = defaults.SUCCESS
    mock_node._get_next_node.return_value = "next node"

    base_executor = executor.BaseExecutor()
    base_executor.run_log_store = mock_run_log_store

    status, next_node = base_executor._get_status_and_next_node_name(current_node=mock_node, dag=mock_dag)
    assert status == defaults.SUCCESS
    assert next_node == "next node"


def test_base_executor_get_status_and_next_node_gets_global_failure_node_by_default_if_step_fails(mocker, monkeypatch):
    mock_node = mocker.MagicMock()
    mock_run_log_store = mocker.MagicMock()
    mock_dag = mocker.MagicMock()
    mock_step_log = mocker.MagicMock()
    mock_fail_node = mocker.MagicMock()

    mock_run_log_store.get_step_log.return_value = mock_step_log
    mock_step_log.status = defaults.FAIL
    mock_node._get_on_failure_node.return_value = None
    mock_dag.get_fail_node.return_value = mock_fail_node
    mock_fail_node.name = "global fail node"

    base_executor = executor.BaseExecutor()
    base_executor.run_log_store = mock_run_log_store

    status, next_node = base_executor._get_status_and_next_node_name(current_node=mock_node, dag=mock_dag)
    assert status == defaults.FAIL
    assert next_node == "global fail node"


def test_base_executor_get_status_and_next_node_gets_node_failure_node_if_provided_if_step_fails(mocker, monkeypatch):
    mock_node = mocker.MagicMock()
    mock_run_log_store = mocker.MagicMock()
    mock_dag = mocker.MagicMock()
    mock_step_log = mocker.MagicMock()

    mock_run_log_store.get_step_log.return_value = mock_step_log
    mock_step_log.status = defaults.FAIL
    mock_node._get_on_failure_node.return_value = "node fail node"

    base_executor = executor.BaseExecutor()
    base_executor.run_log_store = mock_run_log_store

    status, next_node = base_executor._get_status_and_next_node_name(current_node=mock_node, dag=mock_dag)
    assert status == defaults.FAIL
    assert next_node == "node fail node"


def test_base_executor__is_eligible_for_rerun_returns_true_if_no_previous_run_log():
    base_executor = executor.BaseExecutor()

    base_executor.previous_run_log = None

    assert base_executor._is_eligible_for_rerun(node=None)


def test_base_executor__is_eligible_for_rerun_returns_true_if_step_log_not_found(mocker, monkeypatch):
    mock_node = mocker.MagicMock()
    mock_previous_run_log = mocker.MagicMock()
    mock_search_step_by_internal_name = mocker.MagicMock(
        side_effect=exceptions.StepLogNotFoundError(run_id="id", name="hi")
    )

    mock_previous_run_log.search_step_by_internal_name = mock_search_step_by_internal_name
    mock_node._get_step_log_name.return_value = "step_log"

    base_executor = executor.BaseExecutor()
    base_executor.previous_run_log = mock_previous_run_log

    assert base_executor._is_eligible_for_rerun(node=mock_node)
    mock_search_step_by_internal_name.assert_called_once_with("step_log")


def test_base_executor__is_eligible_for_rerun_returns_false_if_previous_was_success(mocker, monkeypatch):
    mock_node = mocker.MagicMock()
    mock_step_log = mocker.MagicMock()
    mock_previous_node_log = mocker.MagicMock()
    mock_previous_run_log = mocker.MagicMock()
    mock_run_log_store = mocker.MagicMock()

    mock_search_step_by_internal_name = mocker.MagicMock(return_value=(mock_previous_node_log, None))
    mock_run_log_store.get_step_log.return_value = mock_step_log

    mock_previous_node_log.status = defaults.SUCCESS

    mock_previous_run_log.search_step_by_internal_name = mock_search_step_by_internal_name
    mock_node._get_step_log_name.return_value = "step_log"

    base_executor = executor.BaseExecutor()
    base_executor.previous_run_log = mock_previous_run_log
    base_executor.run_log_store = mock_run_log_store

    assert base_executor._is_eligible_for_rerun(node=mock_node) is False
    assert mock_step_log.status == defaults.SUCCESS


def test_base_executor__is_eligible_for_rerun_returns_true_if_previous_was_not_success(mocker, monkeypatch):
    mock_node = mocker.MagicMock()
    mock_step_log = mocker.MagicMock()
    mock_previous_node_log = mocker.MagicMock()
    mock_previous_run_log = mocker.MagicMock()
    mock_run_log_store = mocker.MagicMock()

    mock_search_step_by_internal_name = mocker.MagicMock(return_value=(mock_previous_node_log, None))
    mock_run_log_store.get_step_log.return_value = mock_step_log

    mock_previous_node_log.status = defaults.FAIL

    mock_previous_run_log.search_step_by_internal_name = mock_search_step_by_internal_name
    mock_node._get_step_log_name.return_value = "step_log"

    base_executor = executor.BaseExecutor()
    base_executor.previous_run_log = mock_previous_run_log
    base_executor.run_log_store = mock_run_log_store

    assert base_executor._is_eligible_for_rerun(node=mock_node)
    assert base_executor.previous_run_log is None


def test_base_executor_execute_graph_breaks_if_node_status_is_triggered(mocker, monkeypatch):
    mock_dag = mocker.MagicMock()
    mock_execute_from_graph = mocker.MagicMock()
    mock__get_status_and_next_node_name = mocker.MagicMock()
    mock_run_log_store = mocker.MagicMock()

    mock__get_status_and_next_node_name.return_value = defaults.TRIGGERED, None

    monkeypatch.setattr(executor.BaseExecutor, "execute_from_graph", mock_execute_from_graph)
    monkeypatch.setattr(executor.BaseExecutor, "_get_status_and_next_node_name", mock__get_status_and_next_node_name)
    monkeypatch.setattr(executor, "json", mocker.MagicMock())
    base_executor = executor.BaseExecutor()
    base_executor.run_log_store = mock_run_log_store

    base_executor.execute_graph(dag=mock_dag)

    assert mock_execute_from_graph.call_count == 1


def test_base_executor_execute_graph_breaks_if_node_status_is_terminal(mocker, monkeypatch):
    mock_dag = mocker.MagicMock()
    mock_execute_from_graph = mocker.MagicMock()
    mock__get_status_and_next_node_name = mocker.MagicMock()
    mock_run_log_store = mocker.MagicMock()
    mock_node = mocker.MagicMock()

    mock_dag.get_node_by_name.return_value = mock_node
    mock_node.node_type = "success"

    mock__get_status_and_next_node_name.return_value = defaults.SUCCESS, None

    monkeypatch.setattr(executor.BaseExecutor, "execute_from_graph", mock_execute_from_graph)
    monkeypatch.setattr(executor.BaseExecutor, "_get_status_and_next_node_name", mock__get_status_and_next_node_name)
    monkeypatch.setattr(executor, "json", mocker.MagicMock())
    base_executor = executor.BaseExecutor()
    base_executor.run_log_store = mock_run_log_store

    base_executor.execute_graph(dag=mock_dag)

    assert mock_execute_from_graph.call_count == 1


def test_base_executor__resolve_node_config_gives_global_config_if_node_does_not_override(mocker, monkeypatch):
    mock_node = mocker.MagicMock()
    mock_node._get_executor_config.return_value = {}

    base_executor = executor.BaseExecutor()

    assert base_executor._resolve_executor_config(mock_node) == {"a": 1}


def test_base_executor__resolve_node_config_updates_global_config_if_node_overrides(mocker, monkeypatch):
    mock_node = mocker.MagicMock()
    mock_node._get_executor_config.return_value = {"a": 2}

    class MockConfig(BaseModel, extra=Extra.allow):
        placeholders: dict = {}
        a: int = 1

    base_executor = executor.BaseExecutor()
    base_executor.config = MockConfig()

    assert base_executor._resolve_executor_config(mock_node) == {"a": 2}


def test_base_executor__resolve_node_config_updates_global_config_if_node_adds(mocker, monkeypatch):
    mock_node = mocker.MagicMock()
    mock_node._get_executor_config.return_value = {"b": 2}

    class MockConfig(BaseModel, extra=Extra.allow):
        placeholders: dict = {}
        a: int = 1

    monkeypatch.setattr(executor.BaseExecutor, "Config", MockConfig)

    base_executor = executor.BaseExecutor()
    base_executor.config = MockConfig()

    assert base_executor._resolve_executor_config(mock_node) == {"a": 1, "b": 2}


def test_base_executor__resolve_node_config_updates_global_config_from_placeholders(mocker, monkeypatch):
    mock_node = mocker.MagicMock()
    mock_node._get_executor_config.return_value = {"b": 2, "replace": None}

    config = {"a": 1, "placeholders": {"replace": {"c": 3}}}

    class MockConfig(BaseModel, extra=Extra.allow):
        placeholders: dict = {"replace": {"c": 3}}
        a: int = 1

    base_executor = executor.BaseExecutor()
    base_executor.config = MockConfig()

    assert base_executor._resolve_executor_config(mock_node) == {"a": 1, "c": 3, "b": 2}


def test_base_executor_resolve_node_supresess_global_config_from_placeholders_if_its_not_mapping(mocker, monkeypatch):
    mock_node = mocker.MagicMock()
    mock_node._get_executor_config.return_value = {"b": 2, "replace": None}

    config = {"a": 1, "placeholders": {"replace": [1, 2, 3]}}

    class MockConfig(BaseModel, extra=Extra.allow):
        placeholders: dict = {"replace": [1, 2, 3]}
        a: int = 1

    base_executor = executor.BaseExecutor()
    base_executor.config = MockConfig()

    assert base_executor._resolve_executor_config(mock_node) == {"a": 1, "b": 2}


def test_base_executor_execute_graph_raises_exception_if_loop(mocker, monkeypatch):
    mock_dag = mocker.MagicMock()
    mock_execute_from_graph = mocker.MagicMock()
    mock__get_status_and_next_node_name = mocker.MagicMock()
    mock_run_log_store = mocker.MagicMock()
    mock_node = mocker.MagicMock()

    mock_dag.get_node_by_name.return_value = mock_node

    mock__get_status_and_next_node_name.return_value = defaults.SUCCESS, None

    monkeypatch.setattr(executor.BaseExecutor, "execute_from_graph", mock_execute_from_graph)
    monkeypatch.setattr(executor.BaseExecutor, "_get_status_and_next_node_name", mock__get_status_and_next_node_name)
    monkeypatch.setattr(executor, "json", mocker.MagicMock())
    base_executor = executor.BaseExecutor()
    base_executor.run_log_store = mock_run_log_store
    with pytest.raises(Exception):
        base_executor.execute_graph(dag=mock_dag)


def test_local_executor__is_parallel_execution_sends_defaults_if_not_config():
    local_executor = executor.LocalExecutor(config=None)

    assert defaults.ENABLE_PARALLEL == local_executor._is_parallel_execution()


def test_local_executor__is_parallel_execution_sends_from_config_if_present():
    config = {"enable_parallel": "true"}

    local_executor = executor.LocalExecutor(config=config)

    assert local_executor._is_parallel_execution()


def test_local_executor_trigger_job_calls(mocker, monkeypatch):
    mock_node = mocker.MagicMock()
    mock_prepare_for_node_execution = mocker.MagicMock()
    mock__execute_node = mocker.MagicMock()

    monkeypatch.setattr(executor.LocalExecutor, "prepare_for_node_execution", mock_prepare_for_node_execution)
    monkeypatch.setattr(executor.LocalExecutor, "_execute_node", mock__execute_node)

    local_executor = executor.LocalExecutor(config=None)

    local_executor.trigger_job(mock_node)
    assert mock_prepare_for_node_execution.call_count == 1
    assert mock__execute_node.call_count == 1


def test_local_container_executor__is_parallel_execution_sends_defaults_if_not_config():
    local_container_executor = executor.LocalContainerExecutor(config={"docker_image": "test"})

    assert defaults.ENABLE_PARALLEL == local_container_executor._is_parallel_execution()


def test_local_container_executor__is_parallel_execution_sends_from_config_if_present():
    local_container_executor = executor.LocalContainerExecutor(config={"enable_parallel": True, "docker_image": "test"})

    assert local_container_executor._is_parallel_execution()


def test_local_container_executor_docker_image_is_retrieved_from_config():
    config = {"enable_parallel": "true", "docker_image": "docker"}

    local_container_executor = executor.LocalContainerExecutor(config=config)

    assert local_container_executor.docker_image == "docker"


def test_local_container_executor_add_code_ids_uses_global_docker_image(mocker, monkeypatch):
    mock_super_add_code_ids = mocker.MagicMock()
    monkeypatch.setattr(executor.BaseExecutor, "add_code_identities", mock_super_add_code_ids)

    mock_node = mocker.MagicMock()
    mock_node._get_mode_config.return_value = {}

    mock_get_local_docker_image_id = mocker.MagicMock()
    monkeypatch.setattr(executor.utils, "get_local_docker_image_id", mock_get_local_docker_image_id)

    mock_run_log_store = mocker.MagicMock()
    mock_step_log = mocker.MagicMock()
    local_container_executor = executor.LocalContainerExecutor(config={"docker_image": "global"})
    local_container_executor.run_log_store = mock_run_log_store

    local_container_executor.add_code_identities(node=mock_node, step_log=mock_step_log)

    mock_get_local_docker_image_id.assert_called_once_with("global")


def test_local_container_executor_add_code_ids_uses_local_docker_image_if_provided(mocker, monkeypatch):
    mock_super_add_code_ids = mocker.MagicMock()
    monkeypatch.setattr(executor.BaseExecutor, "add_code_identities", mock_super_add_code_ids)

    mock_node = mocker.MagicMock()
    mock_node._get_executor_config.return_value = {"docker_image": "local"}

    mock_get_local_docker_image_id = mocker.MagicMock()
    monkeypatch.setattr(executor.utils, "get_local_docker_image_id", mock_get_local_docker_image_id)

    mock_run_log_store = mocker.MagicMock()
    mock_step_log = mocker.MagicMock()
    local_container_executor = executor.LocalContainerExecutor(config={"docker_image": "global"})
    local_container_executor.run_log_store = mock_run_log_store

    local_container_executor.add_code_identities(node=mock_node, step_log=mock_step_log)

    mock_get_local_docker_image_id.assert_called_once_with("local")


def test_local_container_executor_calls_spin_container_during_trigger_job(mocker, monkeypatch):
    mock_spin_container = mocker.MagicMock()
    mock_step_log = mocker.MagicMock()
    mock_run_log_store = mocker.MagicMock()
    mock_node = mocker.MagicMock()

    mock_run_log_store.get_step_log.return_value = mock_step_log
    mock_step_log.status = defaults.SUCCESS

    monkeypatch.setattr(executor.LocalContainerExecutor, "_spin_container", mock_spin_container)

    local_container_executor = executor.LocalContainerExecutor(config={"docker_image": "test"})
    local_container_executor.run_log_store = mock_run_log_store

    local_container_executor.trigger_job(node=mock_node)

    assert mock_spin_container.call_count == 1


def test_local_container_executor_marks_step_fail_if_status_is_not_success(mocker, monkeypatch):
    mock_spin_container = mocker.MagicMock()
    mock_step_log = mocker.MagicMock()
    mock_run_log_store = mocker.MagicMock()
    mock_node = mocker.MagicMock()

    mock_run_log_store.get_step_log.return_value = mock_step_log
    mock_step_log.status = defaults.PROCESSING

    monkeypatch.setattr(executor.LocalContainerExecutor, "_spin_container", mock_spin_container)

    local_container_executor = executor.LocalContainerExecutor(config={"docker_image": "test"})
    local_container_executor.run_log_store = mock_run_log_store

    local_container_executor.trigger_job(node=mock_node)

    assert mock_step_log.status == defaults.FAIL
