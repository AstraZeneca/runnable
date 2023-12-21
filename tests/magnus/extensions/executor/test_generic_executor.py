import pytest
import logging

from magnus import defaults, exceptions
from magnus.extensions.executor import GenericExecutor
from magnus.extensions import executor
import magnus.extensions.executor as executor


@pytest.fixture(autouse=True)
def instantiable_base_class(monkeypatch, mocker):
    monkeypatch.setattr(GenericExecutor, "__abstractmethods__", set())
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

    test_executor = GenericExecutor()
    assert test_executor._get_parameters() == {"executor": "test"}
    mock_load_yaml.assert_called_once_with("parameters_file")


def test_get_parameters_gets_parameters_from_user_parameters(mocker, monkeypatch, mock_run_context):
    mock_run_context.parameters_file = ""
    monkeypatch.setattr(
        executor.parameters, "get_user_set_parameters", mocker.MagicMock(return_value={"executor": "test"})
    )

    test_executor = GenericExecutor()
    assert test_executor._get_parameters() == {"executor": "test"}


def test_get_parameters_user_parameters_overwrites_parameters_from_parameters_file(
    mocker, monkeypatch, mock_run_context
):
    mock_run_context.parameters_file = "parameters_file"

    mock_load_yaml = mocker.MagicMock(return_value={"executor": "this"})
    monkeypatch.setattr(executor.utils, "load_yaml", mock_load_yaml)
    monkeypatch.setattr(
        executor.parameters, "get_user_set_parameters", mocker.MagicMock(return_value={"executor": "that"})
    )

    test_executor = GenericExecutor()
    assert test_executor._get_parameters() == {"executor": "that"}


def test_set_up_for_rerun_throws_exception_if_run_log_not_exists(mocker, monkeypatch, mock_run_context):
    mock_run_log_store = mocker.MagicMock()

    mock_run_context.run_log_store = mock_run_log_store
    mock_run_context.original_run_id = "original_run_id"
    mock_run_log_store.get_run_log_by_id = mocker.MagicMock(side_effect=exceptions.RunLogNotFoundError("test"))

    with pytest.raises(Exception, match="Expected a run log with id: original_run_id"):
        GenericExecutor()._set_up_for_re_run(parameters={})


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
    GenericExecutor()._set_up_for_re_run(parameters=parameters)

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
    GenericExecutor()._set_up_for_re_run(parameters=parameters)

    mock_catalog_handler_sync_between_runs.assert_called_once_with(previous_run_id="original_run_id", run_id="run_id")
    assert parameters == {"present": "now", "ghost": "from past"}


def test_set_up_run_log_throws_exception_if_run_log_already_exists(mocker, monkeypatch, mock_run_context):
    mock_run_log_store = mocker.MagicMock()

    mock_run_log_store.get_run_log_by_id = mocker.MagicMock(side_effect=exceptions.RunLogExistsError)

    with pytest.raises(exceptions.RunLogExistsError):
        GenericExecutor()._set_up_run_log()


def test_set_up_run_log_exists_ok_returns_without_exception(mocker, monkeypatch, mock_run_context):
    GenericExecutor()._set_up_run_log(exists_ok=True)


def test_set_up_run_log_calls_get_parameters(mocker, monkeypatch, mock_run_context):
    mock_get_parameters = mocker.MagicMock()
    monkeypatch.setattr(GenericExecutor, "_get_parameters", mock_get_parameters)

    mock_run_context.run_log_store.get_run_log_by_id = mocker.MagicMock(
        side_effect=exceptions.RunLogNotFoundError("test")
    )
    mock_run_context.use_cached = False

    GenericExecutor()._set_up_run_log()

    assert mock_get_parameters.call_count == 1


def test_set_up_run_log_calls_set_up_for_re_run(mocker, monkeypatch, mock_run_context):
    mock_set_up_for_re_run = mocker.MagicMock()
    monkeypatch.setattr(GenericExecutor, "_set_up_for_re_run", mock_set_up_for_re_run)

    mock_get_parameters = mocker.MagicMock()
    monkeypatch.setattr(GenericExecutor, "_get_parameters", mock_get_parameters)

    mock_run_context.run_log_store.get_run_log_by_id = mocker.MagicMock(
        side_effect=exceptions.RunLogNotFoundError("test")
    )

    GenericExecutor()._set_up_run_log()

    assert mock_set_up_for_re_run.call_count == 1


def test_set_up_run_log_calls_create_run_log(mocker, monkeypatch, mock_run_context):
    mock_get_parameters = mocker.MagicMock()
    monkeypatch.setattr(GenericExecutor, "_get_parameters", mock_get_parameters)

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

    GenericExecutor()._set_up_run_log()

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
    monkeypatch.setattr(GenericExecutor, "_get_parameters", mock_get_parameters)

    mock_run_context.run_log_store.get_run_log_by_id = mocker.MagicMock(
        side_effect=exceptions.RunLogNotFoundError("test")
    )

    mock_run_context.use_cached = False
    mock_set_parameters = mocker.MagicMock()
    mock_run_context.run_log_store.set_parameters = mock_set_parameters

    GenericExecutor()._set_up_run_log()

    assert mock_set_parameters.call_count == 1


def test_set_up_run_log_store_calls_set_run_config(mocker, monkeypatch, mock_run_context):
    mock_get_parameters = mocker.MagicMock()
    monkeypatch.setattr(GenericExecutor, "_get_parameters", mock_get_parameters)

    mock_run_context.run_log_store.get_run_log_by_id = mocker.MagicMock(
        side_effect=exceptions.RunLogNotFoundError("test")
    )

    mock_run_context.use_cached = False
    mock_set_run_config = mocker.MagicMock()
    mock_run_context.run_log_store.set_parameters = mock_set_run_config

    GenericExecutor()._set_up_run_log()

    assert mock_set_run_config.call_count == 1


def test_base_executor_prepare_for_graph_execution_calls(mocker, monkeypatch, mock_run_context):
    mock_integration = mocker.MagicMock()
    mock_validate = mocker.MagicMock()
    mock_configure_for_traversal = mocker.MagicMock()

    mock_integration.validate = mock_validate
    mock_integration.configure_for_traversal = mock_configure_for_traversal

    mock_set_up_run_log = mocker.MagicMock()
    monkeypatch.setattr(GenericExecutor, "_set_up_run_log", mock_set_up_run_log)

    monkeypatch.setattr(executor, "integration", mock_integration)
    monkeypatch.setattr(executor.BaseExecutor, "_set_up_run_log", mocker.MagicMock())

    base_executor = GenericExecutor()

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
    monkeypatch.setattr(GenericExecutor, "_set_up_run_log", mock_set_up_run_log)

    monkeypatch.setattr(executor, "integration", mock_integration)

    base_executor = GenericExecutor()

    base_executor.prepare_for_node_execution()

    assert mock_configure_for_execution.call_count == 4
    assert mock_validate.call_count == 4


def test_base_executor__sync_catalog_raises_exception_if_stage_not_in_get_or_put(mocker, monkeypatch):
    test_executor = GenericExecutor()
    with pytest.raises(Exception):
        test_executor._sync_catalog(step_log="test", stage="puts")


def test_sync_catalog_does_nothing_for_terminal_node(mocker, monkeypatch, mock_run_context):
    mock_node = mocker.MagicMock()
    mock_node._get_catalog_settings = mocker.MagicMock(side_effect=exceptions.TerminalNodeError)

    test_executor = GenericExecutor()
    test_executor._context_node = mock_node

    test_executor._sync_catalog("test", stage="get")


def test_sync_catalog_does_nothing_for_no_catalog_settings(mocker, monkeypatch, mock_run_context):
    mock_node = mocker.MagicMock()
    mock_node._get_catalog_settings = mocker.MagicMock(return_value={})

    test_executor = GenericExecutor()
    test_executor._context_node = mock_node

    test_executor._sync_catalog("test", stage="get")


def test_sync_catalog_does_nothing_for_catalog_settings_stage_not_in(mocker, monkeypatch, mock_run_context):
    mock_node = mocker.MagicMock()
    mock_node._get_catalog_settings = mocker.MagicMock(return_value={"get": "something"})

    test_executor = GenericExecutor()
    test_executor._context_node = mock_node

    test_executor._sync_catalog("test", stage="put")


def test_sync_catalog_returns_nothing_if_no_syncing_for_node(mocker, monkeypatch, mock_run_context):
    mock_node = mocker.MagicMock()

    mock_node._get_catalog_settings.return_value = None

    test_executor = GenericExecutor()
    test_executor._context_node = mock_node

    assert test_executor._sync_catalog("test", stage="get") is None


def test_sync_catalog_returns_empty_list_if_asked_nothing_in_stage(mocker, monkeypatch, mock_run_context):
    mock_node = mocker.MagicMock()
    mock_node._get_catalog_settings.return_value = {"get": [], "put": []}

    mock_get_effective_compute_folder = mocker.MagicMock(return_value="compute_folder")
    monkeypatch.setattr(GenericExecutor, "get_effective_compute_data_folder", mock_get_effective_compute_folder)

    test_executor = GenericExecutor()
    test_executor._context_node = mock_node

    assert test_executor._sync_catalog("test", stage="get") == []
    assert test_executor._sync_catalog("test", stage="put") == []


def test_sync_catalog_calls_get_from_catalog_handler(mocker, monkeypatch, mock_run_context):
    mock_node = mocker.MagicMock()
    mock_node._get_catalog_settings.return_value = {"get": ["me"], "put": []}
    mock_step_log = mocker.MagicMock()

    mock_get_effective_compute_folder = mocker.MagicMock(return_value="compute_folder")
    monkeypatch.setattr(GenericExecutor, "get_effective_compute_data_folder", mock_get_effective_compute_folder)

    mock_catalog_handler_get = mocker.MagicMock()
    mock_catalog_handler_get.return_value = ["data_catalog"]
    mock_run_context.catalog_handler.get = mock_catalog_handler_get
    mock_run_context.run_id = "run_id"

    test_executor = GenericExecutor()
    test_executor._context_node = mock_node

    data_catalogs = test_executor._sync_catalog(step_log=mock_step_log, stage="get")

    assert data_catalogs == ["data_catalog"]
    mock_catalog_handler_get.assert_called_once_with(name="me", run_id="run_id", compute_data_folder="compute_folder")


def test_sync_catalog_calls_get_from_catalog_handler_as_per_input(mocker, monkeypatch, mock_run_context):
    mock_node = mocker.MagicMock()
    mock_node._get_catalog_settings.return_value = {"get": ["me", "you"], "put": []}
    mock_step_log = mocker.MagicMock()

    mock_get_effective_compute_folder = mocker.MagicMock(return_value="compute_folder")
    monkeypatch.setattr(GenericExecutor, "get_effective_compute_data_folder", mock_get_effective_compute_folder)

    mock_catalog_handler_get = mocker.MagicMock()
    mock_catalog_handler_get.return_value = ["data_catalog"]
    mock_run_context.catalog_handler.get = mock_catalog_handler_get
    mock_run_context.run_id = "run_id"

    test_executor = GenericExecutor()
    test_executor._context_node = mock_node

    data_catalogs = test_executor._sync_catalog(step_log=mock_step_log, stage="get")

    assert data_catalogs == ["data_catalog", "data_catalog"]
    assert mock_catalog_handler_get.call_count == 2


def test_sync_catalog_calls_put_from_catalog_handler(mocker, monkeypatch, mock_run_context):
    mock_node = mocker.MagicMock()
    mock_node._get_catalog_settings.return_value = {"get": [], "put": ["me"]}
    mock_step_log = mocker.MagicMock()

    mock_get_effective_compute_folder = mocker.MagicMock(return_value="compute_folder")
    monkeypatch.setattr(GenericExecutor, "get_effective_compute_data_folder", mock_get_effective_compute_folder)

    mock_catalog_handler_put = mocker.MagicMock()
    mock_catalog_handler_put.return_value = ["data_catalog"]
    mock_run_context.catalog_handler.put = mock_catalog_handler_put
    mock_run_context.run_id = "run_id"

    test_executor = GenericExecutor()
    test_executor._context_node = mock_node

    data_catalogs = test_executor._sync_catalog(step_log=mock_step_log, stage="put")

    assert data_catalogs == ["data_catalog"]
    mock_catalog_handler_put.assert_called_once_with(
        name="me", run_id="run_id", compute_data_folder="compute_folder", synced_catalogs=None
    )


def test_sync_catalog_calls_put_from_catalog_handler_as_per_input(mocker, monkeypatch, mock_run_context):
    mock_node = mocker.MagicMock()
    mock_node._get_catalog_settings.return_value = {"get": [], "put": ["me", "you"]}
    mock_step_log = mocker.MagicMock()

    mock_get_effective_compute_folder = mocker.MagicMock(return_value="compute_folder")
    monkeypatch.setattr(GenericExecutor, "get_effective_compute_data_folder", mock_get_effective_compute_folder)

    mock_catalog_handler_put = mocker.MagicMock()
    mock_catalog_handler_put.return_value = ["data_catalog"]
    mock_run_context.catalog_handler.put = mock_catalog_handler_put
    mock_run_context.run_id = "run_id"

    test_executor = GenericExecutor()
    test_executor._context_node = mock_node

    data_catalogs = test_executor._sync_catalog(step_log=mock_step_log, stage="put")

    assert data_catalogs == ["data_catalog", "data_catalog"]
    assert mock_catalog_handler_put.call_count == 2


def test_sync_catalog_calls_put_sends_synced_catalogs_to_catalog_handler(mocker, monkeypatch, mock_run_context):
    mock_node = mocker.MagicMock()
    mock_node._get_catalog_settings.return_value = {"get": [], "put": ["me"]}
    mock_step_log = mocker.MagicMock()

    mock_get_effective_compute_folder = mocker.MagicMock(return_value="compute_folder")
    monkeypatch.setattr(GenericExecutor, "get_effective_compute_data_folder", mock_get_effective_compute_folder)

    mock_catalog_handler_put = mocker.MagicMock()
    mock_catalog_handler_put.return_value = ["data_catalog"]
    mock_run_context.catalog_handler.put = mock_catalog_handler_put
    mock_run_context.run_id = "run_id"

    test_executor = GenericExecutor()
    test_executor._context_node = mock_node

    data_catalogs = test_executor._sync_catalog(step_log=mock_step_log, stage="put", synced_catalogs="in_sync")

    assert data_catalogs == ["data_catalog"]
    mock_catalog_handler_put.assert_called_once_with(
        name="me", run_id="run_id", compute_data_folder="compute_folder", synced_catalogs="in_sync"
    )


def test_get_effective_compute_data_folder_returns_default(mocker, mock_run_context):
    mock_run_context.catalog_handler.compute_data_folder = "default"

    mock_node = mocker.MagicMock()
    mock_node._get_catalog_settings.return_value = {}

    test_executor = GenericExecutor()
    test_executor._context_node = mock_node

    assert test_executor.get_effective_compute_data_folder() == "default"


def test_get_effective_compute_data_folder_returns_from_node_settings(mocker, mock_run_context):
    mock_run_context.catalog_handler.compute_data_folder = "default"

    mock_node = mocker.MagicMock()
    mock_node._get_catalog_settings.return_value = {"compute_data_folder": "not_default"}

    test_executor = GenericExecutor()
    test_executor._context_node = mock_node

    assert test_executor.get_effective_compute_data_folder() == "not_default"


def test_step_attempt_returns_one_by_default():
    test_executor = GenericExecutor()

    assert test_executor.step_attempt_number == 1


def test_step_attempt_returns_from_env(monkeypatch):
    test_executor = GenericExecutor()

    monkeypatch.setenv("MAGNUS_STEP_ATTEMPT", "2")

    assert test_executor.step_attempt_number == 2


def test_base_executor__is_step_eligible_for_rerun_returns_true_if_not_use_cached(mock_run_context):
    test_executor = GenericExecutor()

    mock_run_context.use_cached = False

    assert test_executor._is_step_eligible_for_rerun(node=None)


def test_base_executor__is_step_eligible_for_rerun_returns_true_if_step_log_not_found(mocker, mock_run_context):
    mock_run_context.use_cached = True

    mock_node = mocker.MagicMock()
    mock_node._get_step_log_name.return_value = "IdontExist"

    mock_run_context.run_log_store.get_step_log.side_effect = exceptions.StepLogNotFoundError(
        run_id="test", name="test"
    )

    test_executor = GenericExecutor()

    assert test_executor._is_step_eligible_for_rerun(node=mock_node)


def test_base_executor__is_step_eligible_for_rerun_returns_true_if_step_failed(mocker, mock_run_context):
    mock_run_context.use_cached = True

    mock_node = mocker.MagicMock()
    mock_node._get_step_log_name.return_value = "IExist"

    mock_run_context.run_log_store.get_step_log.return_value.status = defaults.FAIL

    test_executor = GenericExecutor()

    assert test_executor._is_step_eligible_for_rerun(node=mock_node) is True


def test_base_executor__is_step_eligible_for_rerun_returns_false_if_step_succeeded(mocker, mock_run_context):
    mock_run_context.use_cached = True

    mock_node = mocker.MagicMock()
    mock_node._get_step_log_name.return_value = "IExist"

    mock_run_context.run_log_store.get_step_log.return_value.status = defaults.SUCCESS

    test_executor = GenericExecutor()

    assert test_executor._is_step_eligible_for_rerun(node=mock_node) is False


def test_base_executor_resolve_executor_config_gives_global_config_if_node_does_not_override(mocker, mock_run_context):
    mock_node = mocker.MagicMock()
    mock_node._get_executor_config.return_value = {}

    test_executor = GenericExecutor()

    assert test_executor._resolve_executor_config(mock_node) == {**test_executor.model_dump()}


def test_base_executor__resolve_node_config_updates_global_config_if_node_overrides(mocker, monkeypatch):
    mock_node = mocker.MagicMock()
    mock_node._get_executor_config.return_value = {"enable_parallel": True}

    test_executor = GenericExecutor()

    assert test_executor._resolve_executor_config(mock_node)["enable_parallel"] is True


def test_resolve_node_config_updates_config_with_nested_config(mocker):
    mock_node = mocker.MagicMock()
    mock_node._get_executor_config.return_value = {"first": {"second": {"third": {"a": 1}}}}

    test_executor = GenericExecutor()

    assert test_executor._resolve_executor_config(mock_node)["first"] == {"second": {"third": {"a": 1}}}


def test_base_executor__resolve_node_config_updates_global_config_if_node_adds(mocker, monkeypatch):
    mock_node = mocker.MagicMock()
    mock_node._get_executor_config.return_value = {"b": 2}

    test_executor = GenericExecutor()
    assert test_executor._resolve_executor_config(mock_node) == {**test_executor.model_dump(), **{"b": 2}}


def test_base_executor_resolve_node_supresess_global_config_from_placeholders_if_its_not_mapping(mocker, monkeypatch):
    mock_node = mocker.MagicMock()
    mock_node._get_executor_config.return_value = {"b": 2, "replace": None}

    test_executor = executor.GenericExecutor(placeholders={"replace": {"a": 1}})

    assert test_executor._resolve_executor_config(mock_node) == {**test_executor.model_dump(), **{"b": 2, "a": 1}}


def test_get_status_and_next_node_name_returns_empty_for_terminal_node(mocker, monkeypatch, mock_run_context):
    mock_node = mocker.MagicMock()
    mock_node._get_next_node = mocker.MagicMock(side_effect=exceptions.TerminalNodeError)

    mock_step_log = mocker.MagicMock()
    mock_step_log.status = defaults.SUCCESS
    mock_run_context.run_log_store.get_step_log.return_value = mock_step_log

    test_executor = GenericExecutor()

    assert test_executor._get_status_and_next_node_name(mock_node, "dag") == (defaults.SUCCESS, "")


def test_get_status_and_next_node_name_returns_next_node_if_success(mocker, monkeypatch, mock_run_context):
    mock_node = mocker.MagicMock()
    mock_node._get_next_node.return_value = "next_node"

    mock_step_log = mocker.MagicMock()
    mock_step_log.status = defaults.SUCCESS
    mock_run_context.run_log_store.get_step_log.return_value = mock_step_log

    test_executor = GenericExecutor()

    assert test_executor._get_status_and_next_node_name(mock_node, "dag") == (defaults.SUCCESS, "next_node")


def test_get_status_and_next_node_name_returns_terminal_node_in_case_of_failure(mocker, monkeypatch, mock_run_context):
    mock_node = mocker.MagicMock()
    mock_node._get_next_node.return_value = "next_node"
    mock_node._get_on_failure_node.return_value = ""

    mock_run_context.run_log_store.get_step_log.return_value.status = defaults.FAIL

    mock_dag = mocker.MagicMock()
    mock_dag.get_fail_node.return_value.name = "fail_node"

    test_executor = GenericExecutor()

    assert test_executor._get_status_and_next_node_name(mock_node, mock_dag) == (defaults.FAIL, "fail_node")


def test_get_status_and_next_node_name_returns_on_failure_node_if_failed(mocker, monkeypatch, mock_run_context):
    mock_node = mocker.MagicMock()
    mock_node._get_next_node.return_value = "next_node"
    mock_node._get_on_failure_node.return_value = "me_please"

    mock_run_context.run_log_store.get_step_log.return_value.status = defaults.FAIL

    mock_dag = mocker.MagicMock()
    mock_dag.get_fail_node.return_value.name = "fail_node"

    test_executor = GenericExecutor()

    assert test_executor._get_status_and_next_node_name(mock_node, mock_dag) == (defaults.FAIL, "me_please")


def test_execute_node_calls_store_parameter_with_update_false(mocker, monkeypatch, mock_run_context):
    mock_parameters = mocker.MagicMock()
    monkeypatch.setattr(executor, "parameters", mock_parameters)

    mock_run_context.run_log_store.get_parameters.return_value = {"a": 1}

    test_executor = GenericExecutor()
    test_executor._sync_catalog = mocker.MagicMock()

    mock_node = mocker.MagicMock()
    test_executor._execute_node(mock_node)

    args, kwargs = mock_parameters.set_user_defined_params_as_environment_variables.call_args
    assert args[0] == {"a": 1}


def test_execute_node_raises_exception_if_node_execute_raises_one(mocker, monkeypatch, mock_run_context, caplog):
    mock_run_context.run_log_store.get_parameters.return_value = {"a": 1}
    test_executor = GenericExecutor()
    test_executor._sync_catalog = mocker.MagicMock()

    mock_node = mocker.MagicMock()
    mock_node.execute.side_effect = Exception()

    with caplog.at_level(logging.ERROR, logger="magnus") and pytest.raises(Exception):
        test_executor._execute_node(mock_node)

    assert "This is clearly magnus fault, " in caplog.text


def test_execute_node_sets_step_log_status_to_fail_if_node_fails(mocker, monkeypatch, mock_run_context):
    mock_step_log = mocker.MagicMock()
    mock_run_context.run_log_store.get_step_log.return_value = mock_step_log
    mock_run_context.run_log_store.get_parameters.return_value = {"a": 1}

    mock_attempt_log = mocker.MagicMock()
    mock_attempt_log.status = defaults.FAIL

    mock_node = mocker.MagicMock()
    mock_node.execute.return_value = mock_attempt_log

    test_executor = GenericExecutor()
    test_executor._sync_catalog = mocker.MagicMock()

    test_executor._execute_node(mock_node)

    assert mock_step_log.status == defaults.FAIL


def test_execute_node_sets_step_log_status_to_success_if_node_succeeds(mocker, monkeypatch, mock_run_context):
    mock_step_log = mocker.MagicMock()
    mock_run_context.run_log_store.get_step_log.return_value = mock_step_log
    mock_run_context.run_log_store.get_parameters.return_value = {"a": 1}

    mock_node = mocker.MagicMock()
    mock_node.execute.return_value.status = defaults.SUCCESS

    test_executor = GenericExecutor()
    test_executor._sync_catalog = mocker.MagicMock()

    test_executor._execute_node(mock_node)

    assert mock_step_log.status == defaults.SUCCESS


def test_execute_node_step_log_gets_tracked_data(mocker, monkeypatch, mock_run_context):
    mock_run_context.run_log_store.get_parameters.return_value = {"a": 1}

    mock_step_log = mocker.MagicMock()
    mock_run_context.run_log_store.get_step_log.return_value = mock_step_log

    mock_utils = mocker.MagicMock()
    mock_utils.get_tracked_data.return_value = {"a": 2}
    monkeypatch.setattr(executor, "utils", mock_utils)

    mock_node = mocker.MagicMock()
    mock_node.execute.return_value.status = defaults.SUCCESS

    test_executor = GenericExecutor()
    test_executor._sync_catalog = mocker.MagicMock()

    test_executor._execute_node(mock_node)
    assert mock_step_log.user_defined_metrics == {"a": 2}


def test_send_return_code_raises_exception_if_pipeline_execution_failed(mocker, mock_run_context):
    mock_run_context.run_log_store.get_run_log_by_id.return_value.status = defaults.FAIL

    test_executor = GenericExecutor()

    with pytest.raises(exceptions.ExecutionFailedError):
        test_executor.send_return_code()


def test_send_return_code_does_not_raise_exception_if_pipeline_execution_succeeded(mocker, mock_run_context):
    mock_run_context.run_log_store.get_run_log_by_id.return_value.status = defaults.SUCCESS

    test_executor = GenericExecutor()
    test_executor.send_return_code()
