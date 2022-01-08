import pytest

from magnus import executor
from magnus import defaults


def test_base_executor_is_parallel_execution_uses_default():
    base_executor = executor.BaseExecutor(config=None)

    assert base_executor.is_parallel_execution() == defaults.ENABLE_PARALLEL


def test_base_executor_set_up_run_log_with_no_previous_run_log(mocker, monkeypatch):
    base_executor = executor.BaseExecutor(config=None)

    mock_run_log_store = mocker.MagicMock()
    mock_run_log = mocker.MagicMock()
    mock_run_log_store.create_run_log.return_value = mock_run_log

    base_executor.run_log_store = mock_run_log_store
    base_executor.run_id = 'run_id'
    base_executor.cmd_line_arguments = {'a': 1}

    monkeypatch.setattr(executor.utils, 'get_run_config', mocker.MagicMock(return_value={'executor': 'test'}))

    base_executor.set_up_run_log()

    assert mock_run_log.status == defaults.PROCESSING
    assert mock_run_log.use_cached == False
    assert mock_run_log.parameters == {'a': 1}
    assert mock_run_log.run_config == {'executor': 'test'}


def test_base_executor_set_up_run_log_with_previous_run_log(mocker, monkeypatch):
    base_executor = executor.BaseExecutor(config=None)

    mock_run_log_store = mocker.MagicMock()
    mock_run_log = mocker.MagicMock()
    mock_run_log_store.create_run_log.return_value = mock_run_log

    mock_previous_run_log = mocker.MagicMock()
    mock_previous_run_log.run_id = 'old run id'
    mock_previous_run_log.parameters = {'b': 1}

    base_executor.run_log_store = mock_run_log_store
    base_executor.run_id = 'run_id'
    base_executor.cmd_line_arguments = {'a': 1}
    base_executor.previous_run_log = mock_previous_run_log
    base_executor.catalog_handler = mocker.MagicMock()

    monkeypatch.setattr(executor.utils, 'get_run_config', mocker.MagicMock(return_value={'executor': 'test'}))

    base_executor.set_up_run_log()

    assert mock_run_log.status == defaults.PROCESSING
    assert mock_run_log.use_cached == True
    assert mock_run_log.parameters == {'a': 1, 'b': 1}
    assert mock_run_log.run_config == {'executor': 'test'}


def test_base_executor_prepare_for_graph_execution_calls(mocker, monkeypatch):
    mock_integration = mocker.MagicMock()
    mock_validate = mocker.MagicMock()
    mock_configure_for_traversal = mocker.MagicMock()

    mock_integration.validate = mock_validate
    mock_integration.configure_for_traversal = mock_configure_for_traversal

    monkeypatch.setattr(executor, 'integration', mock_integration)
    monkeypatch.setattr(executor.BaseExecutor, 'set_up_run_log', mocker.MagicMock())

    base_executor = executor.BaseExecutor(config=None)

    base_executor.prepare_for_graph_execution()

    assert mock_configure_for_traversal.call_count == 3
    assert mock_validate.call_count == 3


def test_base_execution_prepare_for_node_calls(mocker, monkeypatch):
    mock_integration = mocker.MagicMock()
    mock_validate = mocker.MagicMock()
    mock_configure_for_execution = mocker.MagicMock()

    mock_integration.validate = mock_validate
    mock_integration.configure_for_execution = mock_configure_for_execution

    monkeypatch.setattr(executor, 'integration', mock_integration)

    base_executor = executor.BaseExecutor(config=None)

    base_executor.prepare_for_node_execution('test')

    assert mock_configure_for_execution.call_count == 3
    assert mock_validate.call_count == 3


def test_base_executor_sync_catalog_returns_nothing_if_no_syncing_for_node(mocker, monkeypatch):
    mock_node = mocker.MagicMock()

    mock_node.get_catalog_settings.return_value = None

    base_executor = executor.BaseExecutor(config=None)

    assert base_executor.sync_catalog(mock_node, None, stage='get') is None


def test_base_executor_sync_catalog_raises_exception_if_stage_not_in_get_or_put(mocker, monkeypatch):
    base_executor = executor.BaseExecutor(config=None)

    with pytest.raises(Exception):
        base_executor.sync_catalog(node=None, step_log=None, stage='puts')
