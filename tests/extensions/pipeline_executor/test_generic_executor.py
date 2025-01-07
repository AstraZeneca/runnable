import pytest

from runnable import defaults, exceptions


@pytest.fixture(autouse=True)
def instantiable_base_class(monkeypatch):
    monkeypatch.setattr(GenericExecutor, "__abstractmethods__", set())
    yield


@pytest.fixture
def mock_base_node(monkeypatch, mocker):
    from runnable.nodes import BaseNode

    monkeypatch.setattr(BaseNode, "__abstractmethods__", set())

    class MockNode(BaseNode):
        node_type: str = "test"
        name: str = "name"
        internal_name: str = "internal_name"

    yield MockNode


@pytest.fixture
def mock_run_context(mocker, monkeypatch):
    mock_run_context = mocker.Mock()
    monkeypatch.setattr(module.context, "run_context", mock_run_context)
    return mock_run_context


def test_get_parameters_gets_parameters_from_user_parameters(
    mocker, monkeypatch, mock_run_context
):
    mock_run_context.parameters_file = ""
    monkeypatch.setattr(
        module.parameters,
        "get_user_set_parameters",
        mocker.MagicMock(return_value={"executor": "test"}),
    )

    test_executor = GenericExecutor()
    assert test_executor._get_parameters() == {"executor": "test"}


def test_get_parameters_user_parameters_overwrites_parameters_from_parameters_file(
    mocker, monkeypatch, mock_run_context
):
    mock_run_context.parameters_file = "parameters_file"

    mock_load_yaml = mocker.MagicMock(return_value={"executor": "this"})
    monkeypatch.setattr(module.utils, "load_yaml", mock_load_yaml)
    monkeypatch.setattr(
        module.parameters,
        "get_user_set_parameters",
        mocker.MagicMock(return_value={"executor": "that"}),
    )

    test_executor = GenericExecutor()
    assert test_executor._get_parameters() == {"executor": "that"}


def test_set_up_run_log_throws_exception_if_run_log_already_exists(
    mocker, monkeypatch, mock_run_context
):
    mock_run_log_store = mocker.MagicMock()

    mock_run_log_store.get_run_log_by_id = mocker.MagicMock(
        side_effect=exceptions.RunLogExistsError
    )

    with pytest.raises(exceptions.RunLogExistsError):
        GenericExecutor()._set_up_run_log()


def test_set_up_run_log_exists_ok_returns_without_exception(
    mocker, monkeypatch, mock_run_context
):
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

    GenericExecutor()._set_up_run_log()

    mock_create_run_log.assert_called_once_with(
        run_id="test",
        tag="tag",
        status=defaults.PROCESSING,
        dag_hash="dag_hash",
    )


def test_set_up_run_log_store_calls_set_parameters(
    mocker, monkeypatch, mock_run_context
):
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


def test_set_up_run_log_store_calls_set_run_config(
    mocker, monkeypatch, mock_run_context
):
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


def test_base_executor__sync_catalog_raises_exception_if_stage_not_in_get_or_put(
    mocker, monkeypatch
):
    test_executor = GenericExecutor()
    with pytest.raises(Exception):
        test_executor._sync_catalog(step_log="test", stage="puts")


def test_sync_catalog_does_nothing_for_terminal_node(
    mocker, monkeypatch, mock_run_context, mock_base_node
):
    mock_node = mock_base_node()
    mock_node._get_catalog_settings = mocker.MagicMock(
        side_effect=exceptions.TerminalNodeError
    )

    test_executor = GenericExecutor()
    test_executor._context_node = mock_node

    test_executor._sync_catalog(stage="get")


def test_sync_catalog_does_nothing_for_no_catalog_settings(
    mocker, monkeypatch, mock_run_context, mock_base_node
):
    mock_node = mock_base_node()
    mock_node._get_catalog_settings = mocker.MagicMock(return_value={})

    test_executor = GenericExecutor()
    test_executor._context_node = mock_node

    test_executor._sync_catalog(stage="get")


def test_sync_catalog_does_nothing_for_catalog_settings_stage_not_in(
    mocker, monkeypatch, mock_run_context, mock_base_node
):
    mock_node = mock_base_node()
    mock_node._get_catalog_settings = mocker.MagicMock(
        return_value={"get": "something"}
    )

    test_executor = GenericExecutor()
    test_executor._context_node = mock_node

    test_executor._sync_catalog(stage="put")


def test_sync_catalog_returns_nothing_if_no_syncing_for_node(
    mocker, monkeypatch, mock_run_context, mock_base_node
):
    mock_node = mock_base_node()
    setattr(mock_node, "_get_catalog_settings", lambda: None)

    test_executor = GenericExecutor()
    test_executor._context_node = mock_node

    assert test_executor._sync_catalog(stage="get") is None


def test_sync_catalog_returns_empty_list_if_asked_nothing_in_stage(
    mocker, monkeypatch, mock_run_context, mock_base_node
):
    mock_node = mock_base_node()
    setattr(mock_node, "_get_catalog_settings", lambda: {"get": [], "put": []})

    mock_get_effective_compute_folder = mocker.MagicMock(return_value="compute_folder")
    monkeypatch.setattr(
        GenericExecutor,
        "get_effective_compute_data_folder",
        mock_get_effective_compute_folder,
    )

    test_executor = GenericExecutor()
    test_executor._context_node = mock_node

    assert test_executor._sync_catalog(stage="get") == []
    assert test_executor._sync_catalog(stage="put") == []


def test_sync_catalog_calls_get_from_catalog_handler(
    mocker, monkeypatch, mock_run_context, mock_base_node
):
    mock_node = mock_base_node()
    setattr(mock_node, "_get_catalog_settings", lambda: {"get": ["me"], "put": []})

    mock_get_effective_compute_folder = mocker.MagicMock(return_value="compute_folder")
    monkeypatch.setattr(
        GenericExecutor,
        "get_effective_compute_data_folder",
        mock_get_effective_compute_folder,
    )

    mock_catalog_handler_get = mocker.MagicMock()
    mock_catalog_handler_get.return_value = ["data_catalog"]
    mock_run_context.catalog_handler.get = mock_catalog_handler_get
    mock_run_context.run_id = "run_id"

    test_executor = GenericExecutor()
    test_executor._context_node = mock_node

    data_catalogs = test_executor._sync_catalog(stage="get")

    assert data_catalogs == ["data_catalog"]
    mock_catalog_handler_get.assert_called_once_with(
        name="me", run_id="run_id", compute_data_folder="compute_folder"
    )


def test_sync_catalog_calls_get_from_catalog_handler_as_per_input(
    mocker, monkeypatch, mock_run_context, mock_base_node
):
    mock_node = mock_base_node()
    setattr(
        mock_node, "_get_catalog_settings", lambda: {"get": ["me", "you"], "put": []}
    )

    mock_get_effective_compute_folder = mocker.MagicMock(return_value="compute_folder")
    monkeypatch.setattr(
        GenericExecutor,
        "get_effective_compute_data_folder",
        mock_get_effective_compute_folder,
    )

    mock_catalog_handler_get = mocker.MagicMock()
    mock_catalog_handler_get.return_value = ["data_catalog"]
    mock_run_context.catalog_handler.get = mock_catalog_handler_get
    mock_run_context.run_id = "run_id"

    test_executor = GenericExecutor()
    test_executor._context_node = mock_node

    data_catalogs = test_executor._sync_catalog(stage="get")

    assert data_catalogs == ["data_catalog", "data_catalog"]
    assert mock_catalog_handler_get.call_count == 2


def test_sync_catalog_calls_put_from_catalog_handler(
    mocker, monkeypatch, mock_run_context, mock_base_node
):
    mock_node = mock_base_node()
    setattr(mock_node, "_get_catalog_settings", lambda: {"get": [], "put": ["me"]})

    mock_get_effective_compute_folder = mocker.MagicMock(return_value="compute_folder")
    monkeypatch.setattr(
        GenericExecutor,
        "get_effective_compute_data_folder",
        mock_get_effective_compute_folder,
    )

    mock_catalog_handler_put = mocker.MagicMock()
    mock_catalog_handler_put.return_value = ["data_catalog"]
    mock_run_context.catalog_handler.put = mock_catalog_handler_put
    mock_run_context.run_id = "run_id"

    test_executor = GenericExecutor()
    test_executor._context_node = mock_node

    data_catalogs = test_executor._sync_catalog(stage="put")

    assert data_catalogs == ["data_catalog"]
    mock_catalog_handler_put.assert_called_once_with(
        name="me",
        run_id="run_id",
        compute_data_folder="compute_folder",
        synced_catalogs=None,
    )


def test_sync_catalog_calls_put_from_catalog_handler_as_per_input(
    mocker, monkeypatch, mock_run_context, mock_base_node
):
    mock_node = mock_base_node()
    setattr(
        mock_node, "_get_catalog_settings", lambda: {"get": [], "put": ["me", "you"]}
    )

    mock_get_effective_compute_folder = mocker.MagicMock(return_value="compute_folder")
    monkeypatch.setattr(
        GenericExecutor,
        "get_effective_compute_data_folder",
        mock_get_effective_compute_folder,
    )

    mock_catalog_handler_put = mocker.MagicMock()
    mock_catalog_handler_put.return_value = ["data_catalog"]
    mock_run_context.catalog_handler.put = mock_catalog_handler_put
    mock_run_context.run_id = "run_id"

    test_executor = GenericExecutor()
    test_executor._context_node = mock_node

    data_catalogs = test_executor._sync_catalog(stage="put")

    assert data_catalogs == ["data_catalog", "data_catalog"]
    assert mock_catalog_handler_put.call_count == 2


def test_sync_catalog_calls_put_sends_synced_catalogs_to_catalog_handler(
    mocker, monkeypatch, mock_run_context, mock_base_node
):
    mock_node = mock_base_node()
    setattr(mock_node, "_get_catalog_settings", lambda: {"get": [], "put": ["me"]})

    mock_get_effective_compute_folder = mocker.MagicMock(return_value="compute_folder")
    monkeypatch.setattr(
        GenericExecutor,
        "get_effective_compute_data_folder",
        mock_get_effective_compute_folder,
    )

    mock_catalog_handler_put = mocker.MagicMock()
    mock_catalog_handler_put.return_value = ["data_catalog"]
    mock_run_context.catalog_handler.put = mock_catalog_handler_put
    mock_run_context.run_id = "run_id"

    test_executor = GenericExecutor()
    test_executor._context_node = mock_node

    data_catalogs = test_executor._sync_catalog(stage="put", synced_catalogs="in_sync")

    assert data_catalogs == ["data_catalog"]
    mock_catalog_handler_put.assert_called_once_with(
        name="me",
        run_id="run_id",
        compute_data_folder="compute_folder",
        synced_catalogs="in_sync",
    )


def test_get_effective_compute_data_folder_returns_default(
    mocker, mock_run_context, mock_base_node
):
    mock_run_context.catalog_handler.compute_data_folder = "default"

    mock_node = mock_base_node()
    setattr(mock_node, "_get_catalog_settings", lambda: {})

    test_executor = GenericExecutor()
    test_executor._context_node = mock_node

    assert test_executor.get_effective_compute_data_folder() == "default"


def test_get_effective_compute_data_folder_returns_from_node_settings(
    mocker, mock_run_context, mock_base_node
):
    mock_run_context.catalog_handler.compute_data_folder = "default"

    mock_node = mock_base_node()
    setattr(
        mock_node,
        "_get_catalog_settings",
        lambda: {"compute_data_folder": "not_default"},
    )

    test_executor = GenericExecutor()
    test_executor._context_node = mock_node

    assert test_executor.get_effective_compute_data_folder() == "not_default"


def test_step_attempt_returns_one_by_default():
    test_executor = GenericExecutor()

    assert test_executor.step_attempt_number == 1


def test_step_attempt_returns_from_env(monkeypatch):
    test_executor = GenericExecutor()

    monkeypatch.setenv("RUNNABLE_STEP_ATTEMPT", "2")

    assert test_executor.step_attempt_number == 2


def test_base_executor_resolve_executor_config_gives_global_config_if_node_does_not_override(
    mocker, mock_run_context, mock_base_node
):
    mock_node = mock_base_node()
    setattr(mock_node, "_get_catalog_settings", lambda: {})

    mock_run_context.variables = {}

    test_executor = GenericExecutor()

    assert test_executor._resolve_executor_config(mock_node) == {
        **test_executor.model_dump()
    }


def test_get_status_and_next_node_name_returns_empty_for_terminal_node(
    mocker, monkeypatch, mock_run_context, mock_base_node
):
    mock_node = mock_base_node()

    def f():
        raise exceptions.TerminalNodeError()

    setattr(mock_node, "_get_catalog_settings", f)
    mock_node._get_next_node = mocker.MagicMock(
        side_effect=exceptions.TerminalNodeError
    )

    mock_step_log = mocker.MagicMock()
    mock_step_log.status = defaults.SUCCESS
    mock_run_context.run_log_store.get_step_log.return_value = mock_step_log

    test_executor = GenericExecutor()

    assert test_executor._get_status_and_next_node_name(mock_node, "dag") == (
        defaults.SUCCESS,
        "",
    )


def test_get_status_and_next_node_name_returns_next_node_if_success(
    mocker, monkeypatch, mock_run_context, mock_base_node
):
    mock_node = mock_base_node()
    setattr(mock_node, "_get_next_node", lambda: "next_node")

    mock_step_log = mocker.MagicMock()
    mock_step_log.status = defaults.SUCCESS
    mock_run_context.run_log_store.get_step_log.return_value = mock_step_log

    test_executor = GenericExecutor()

    assert test_executor._get_status_and_next_node_name(mock_node, "dag") == (
        defaults.SUCCESS,
        "next_node",
    )


def test_get_status_and_next_node_name_returns_terminal_node_in_case_of_failure(
    mocker, monkeypatch, mock_run_context, mock_base_node
):
    mock_node = mock_base_node()
    setattr(mock_node, "_get_next_node", lambda: "next node")
    setattr(mock_node, "_get_on_failure_node", lambda: "")

    mock_run_context.run_log_store.get_step_log.return_value.status = defaults.FAIL

    mock_dag = mocker.MagicMock()
    mock_dag.get_fail_node.return_value.name = "fail_node"

    test_executor = GenericExecutor()

    assert test_executor._get_status_and_next_node_name(mock_node, mock_dag) == (
        defaults.FAIL,
        "fail_node",
    )


def test_get_status_and_next_node_name_returns_on_failure_node_if_failed(
    mocker, monkeypatch, mock_run_context, mock_base_node
):
    mock_node = mock_base_node()
    setattr(mock_node, "_get_next_node", lambda: "next_node")
    setattr(mock_node, "_get_on_failure_node", lambda: "me_please")

    mock_run_context.run_log_store.get_step_log.return_value.status = defaults.FAIL

    mock_dag = mocker.MagicMock()
    mock_dag.get_fail_node.return_value.name = "fail_node"

    test_executor = GenericExecutor()

    assert test_executor._get_status_and_next_node_name(mock_node, mock_dag) == (
        defaults.FAIL,
        "me_please",
    )


def test_send_return_code_raises_exception_if_pipeline_execution_failed(
    mocker, mock_run_context
):
    mock_run_context.run_log_store.get_run_log_by_id.return_value.status = defaults.FAIL

    test_executor = GenericExecutor()

    with pytest.raises(exceptions.ExecutionFailedError):
        test_executor.send_return_code()


def test_send_return_code_does_not_raise_exception_if_pipeline_execution_succeeded(
    mocker, mock_run_context
):
    mock_run_context.run_log_store.get_run_log_by_id.return_value.status = (
        defaults.SUCCESS
    )

    test_executor = GenericExecutor()
    test_executor.send_return_code()


from extensions import pipeline_executor as module
from extensions.pipeline_executor import GenericPipelineExecutor as GenericExecutor
