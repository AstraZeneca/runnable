import pytest

from magnus import defaults, exceptions, executor


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

    monkeypatch.setattr(executor.utils, 'get_run_config', mocker.MagicMock(return_value={'executor': 'test'}))

    base_executor.set_up_run_log()

    assert mock_run_log.status == defaults.PROCESSING
    assert mock_run_log.use_cached == False
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

    base_executor.previous_run_log = mock_previous_run_log
    base_executor.catalog_handler = mocker.MagicMock()

    monkeypatch.setattr(executor.utils, 'get_run_config', mocker.MagicMock(return_value={'executor': 'test'}))

    base_executor.set_up_run_log()

    assert mock_run_log.status == defaults.PROCESSING
    assert mock_run_log.use_cached == True
    assert mock_run_log.parameters == {'b': 1}
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


def test_base_executor_sync_catalog_uses_catalog_handler_compute_folder_by_default(mocker, monkeypatch):
    mock_node = mocker.MagicMock()
    mock_node.get_catalog_settings.return_value = {'get': ['all']}

    mock_catalog = mocker.MagicMock()
    mock_catalog.compute_data_folder = 'data/'

    mock_catalog_get = mocker.MagicMock()
    mock_catalog.get = mock_catalog_get

    mock_step_log = mocker.MagicMock()

    base_executor = executor.BaseExecutor(config=None)
    base_executor.run_id = 'run_id'
    base_executor.catalog_handler = mock_catalog

    base_executor.sync_catalog(mock_node, mock_step_log, stage='get')

    mock_catalog_get.assert_called_once_with(
        name='all', run_id='run_id', compute_data_folder='data/', synced_catalogs=None)


def test_base_executor_sync_catalog_uses_compute_folder_if_provided_by_node(mocker, monkeypatch):
    mock_node = mocker.MagicMock()
    mock_node.get_catalog_settings.return_value = {'get': ['all'], 'compute_data_folder': 'data_from_node'}

    mock_catalog = mocker.MagicMock()
    mock_catalog.compute_data_folder = 'data/'

    mock_catalog_get = mocker.MagicMock()
    mock_catalog.get = mock_catalog_get

    mock_step_log = mocker.MagicMock()

    base_executor = executor.BaseExecutor(config=None)
    base_executor.run_id = 'run_id'
    base_executor.catalog_handler = mock_catalog

    base_executor.sync_catalog(mock_node, mock_step_log, stage='get')

    mock_catalog_get.assert_called_once_with(
        name='all', run_id='run_id', compute_data_folder='data_from_node', synced_catalogs=None)


def test_base_executor_add_code_identities_adds_git_identity(mocker, monkeypatch):
    mock_step_log = mocker.MagicMock()

    mock_step_log.code_identities = []

    mock_utils_get_git_code_id = mocker.MagicMock(return_value='code id')
    monkeypatch.setattr(executor.utils, 'get_git_code_identity', mock_utils_get_git_code_id)

    base_executor = executor.BaseExecutor(config=None)

    base_executor.add_code_identities(node=None, step_log=mock_step_log)

    assert mock_step_log.code_identities == ['code id']


def test_base_executor_trigger_job_raises_exception():
    base_executor = executor.BaseExecutor(config=None)

    with pytest.raises(NotImplementedError):
        base_executor.trigger_job(node=None)


def test_base_executor_execute_from_graph_executes_node_for_success_or_fail(mocker, monkeypatch):
    mock_node = mocker.MagicMock()
    mock_run_log_store = mocker.MagicMock()
    mock_execute_node = mocker.MagicMock()

    monkeypatch.setattr(executor.BaseExecutor, 'add_code_identities', mocker.MagicMock())
    monkeypatch.setattr(executor.BaseExecutor, 'execute_node', mock_execute_node)

    mock_node.node_type = 'success'
    base_executor = executor.BaseExecutor(config=None)
    base_executor.run_log_store = mock_run_log_store

    base_executor.execute_from_graph(node=mock_node, map_variable=None)

    assert mock_execute_node.call_count == 1

    mock_node.reset_mock()
    mock_execute_node.reset_mock()

    mock_node.node_type = 'fail'
    base_executor.execute_from_graph(node=mock_node, map_variable=None)

    assert mock_execute_node.call_count == 1


def test_base_executor_execute_from_graph_makes_step_log_processing(mocker, monkeypatch):
    mock_node = mocker.MagicMock()
    mock_run_log_store = mocker.MagicMock()
    mock_execute_node = mocker.MagicMock()
    mock_step_log = mocker.MagicMock()

    mock_run_log_store.create_step_log.return_value = mock_step_log

    monkeypatch.setattr(executor.BaseExecutor, 'add_code_identities', mocker.MagicMock())
    monkeypatch.setattr(executor.BaseExecutor, 'execute_node', mock_execute_node)

    mock_node.node_type = 'success'
    base_executor = executor.BaseExecutor(config=None)
    base_executor.run_log_store = mock_run_log_store

    base_executor.execute_from_graph(node=mock_node, map_variable=None)

    assert mock_step_log.status == defaults.PROCESSING


def test_base_executor_execute_from_graph_makes_step_log_success_if_previous_run_log_success(mocker, monkeypatch):
    mock_node = mocker.MagicMock()
    mock_run_log_store = mocker.MagicMock()
    mock_step_log = mocker.MagicMock()

    mock_run_log_store.create_step_log.return_value = mock_step_log

    monkeypatch.setattr(executor.BaseExecutor, 'add_code_identities', mocker.MagicMock())
    monkeypatch.setattr(executor.BaseExecutor, 'is_eligible_for_rerun', mocker.MagicMock(return_value=False))

    base_executor = executor.BaseExecutor(config=None)
    base_executor.run_log_store = mock_run_log_store

    base_executor.execute_from_graph(node=mock_node, map_variable=None)

    assert mock_step_log.status == defaults.SUCCESS


def test_base_executor_execute_from_graph_delegates_to_execute_as_graph_for_composite_nodes(mocker, monkeypatch):
    mock_node = mocker.MagicMock()
    mock_run_log_store = mocker.MagicMock()
    mock_step_log = mocker.MagicMock()
    mock_node_execute_as_graph = mocker.MagicMock()

    mock_run_log_store.create_step_log.return_value = mock_step_log

    mock_node.node_type = 'parallel'
    mock_node.execute_as_graph = mock_node_execute_as_graph
    monkeypatch.setattr(executor.BaseExecutor, 'add_code_identities', mocker.MagicMock())

    base_executor = executor.BaseExecutor(config=None)
    base_executor.run_log_store = mock_run_log_store

    base_executor.execute_from_graph(node=mock_node, map_variable=None)

    assert mock_node_execute_as_graph.call_count == 1
    assert mock_step_log.status == defaults.PROCESSING

    mock_node_execute_as_graph.reset_mock()
    mock_step_log.reset_mock()

    mock_node.node_type = 'dag'
    base_executor.execute_from_graph(node=mock_node, map_variable=None)

    assert mock_node_execute_as_graph.call_count == 1
    assert mock_step_log.status == defaults.PROCESSING

    mock_node_execute_as_graph.reset_mock()
    mock_step_log.reset_mock()

    mock_node.node_type = 'map'
    base_executor.execute_from_graph(node=mock_node, map_variable=None)

    assert mock_node_execute_as_graph.call_count == 1
    assert mock_step_log.status == defaults.PROCESSING


def test_base_executor_execute_from_graph_triggers_job_for_simple_nodes(mocker, monkeypatch):
    mock_node = mocker.MagicMock()
    mock_run_log_store = mocker.MagicMock()
    mock_step_log = mocker.MagicMock()
    mock_trigger_job = mocker.MagicMock()

    mock_run_log_store.create_step_log.return_value = mock_step_log

    mock_node.node_type = 'task'
    mock_node.is_composite = False
    monkeypatch.setattr(executor.BaseExecutor, 'add_code_identities', mocker.MagicMock())
    monkeypatch.setattr(executor.BaseExecutor, 'trigger_job', mock_trigger_job)

    base_executor = executor.BaseExecutor(config=None)
    base_executor.run_log_store = mock_run_log_store

    base_executor.execute_from_graph(node=mock_node, map_variable=None)

    assert mock_trigger_job.call_count == 1
    assert mock_step_log.status == defaults.PROCESSING


def test_base_executor_execute_node_calls_catalog(mocker, monkeypatch):
    mock_node = mocker.MagicMock()
    mock_node.get_max_attempts.return_value = 1

    mock_run_log_store = mocker.MagicMock()
    mock_step_catalog = mocker.MagicMock()
    mock_sync_catalog = mocker.MagicMock()
    mock_sync_catalog.return_value = 'data_catalogs_get'

    monkeypatch.setattr(executor, 'interaction', mocker.MagicMock())
    monkeypatch.setattr(executor, 'utils', mocker.MagicMock())
    monkeypatch.setattr(executor.BaseExecutor, 'sync_catalog', mock_sync_catalog)

    mock_run_log_store.get_step_log.return_value = mock_step_catalog
    base_executor = executor.BaseExecutor(config=None)
    base_executor.run_log_store = mock_run_log_store

    base_executor.execute_node(node=mock_node)

    mock_sync_catalog.assert_any_call(mock_node, mock_step_catalog, stage='get')
    mock_sync_catalog.assert_any_call(mock_node, mock_step_catalog, stage='put', synced_catalogs='data_catalogs_get')


def test_base_executor_sets_step_log_to_success_if_node_succeeds(mocker, monkeypatch):
    mock_node = mocker.MagicMock()
    mock_node.get_max_attempts.return_value = 1

    mock_run_log_store = mocker.MagicMock()
    mock_step_catalog = mocker.MagicMock()
    mock_sync_catalog = mocker.MagicMock()
    mock_sync_catalog.return_value = 'data_catalogs_get'

    monkeypatch.setattr(executor, 'interaction', mocker.MagicMock())
    monkeypatch.setattr(executor, 'utils', mocker.MagicMock())
    monkeypatch.setattr(executor.BaseExecutor, 'sync_catalog', mock_sync_catalog)

    mock_run_log_store.get_step_log.return_value = mock_step_catalog
    base_executor = executor.BaseExecutor(config=None)
    base_executor.run_log_store = mock_run_log_store

    base_executor.execute_node(node=mock_node)

    assert mock_step_catalog.status == defaults.SUCCESS


def test_base_executor_sets_status_to_fail_if_attempt_log_is_fail(monkeypatch, mocker):
    mock_node = mocker.MagicMock()
    mock_node.get_max_attempts.return_value = 1
    mock_attempt_log = mocker.MagicMock()
    mock_node.execute.return_value = mock_attempt_log
    mock_attempt_log.status = defaults.FAIL

    mock_run_log_store = mocker.MagicMock()
    mock_step_catalog = mocker.MagicMock()
    mock_sync_catalog = mocker.MagicMock()

    monkeypatch.setattr(executor, 'interaction', mocker.MagicMock())
    monkeypatch.setattr(executor, 'utils', mocker.MagicMock())
    monkeypatch.setattr(executor.BaseExecutor, 'sync_catalog', mock_sync_catalog)

    mock_run_log_store.get_step_log.return_value = mock_step_catalog
    base_executor = executor.BaseExecutor(config=None)
    base_executor.run_log_store = mock_run_log_store

    base_executor.execute_node(node=mock_node)

    assert mock_step_catalog.status == defaults.FAIL


def test_base_executor_sets_status_to_fail_if_node_raises_exception(monkeypatch, mocker):
    mock_node = mocker.MagicMock()
    mock_node.get_max_attempts.return_value = 1
    mock_attempt_log = mocker.MagicMock()
    mock_node.execute.side_effect = Exception()

    mock_run_log_store = mocker.MagicMock()
    mock_step_catalog = mocker.MagicMock()
    mock_sync_catalog = mocker.MagicMock()

    monkeypatch.setattr(executor, 'interaction', mocker.MagicMock())
    monkeypatch.setattr(executor, 'utils', mocker.MagicMock())
    monkeypatch.setattr(executor.BaseExecutor, 'sync_catalog', mock_sync_catalog)

    mock_run_log_store.get_step_log.return_value = mock_step_catalog
    base_executor = executor.BaseExecutor(config=None)
    base_executor.run_log_store = mock_run_log_store

    base_executor.execute_node(node=mock_node)

    assert mock_step_catalog.status == defaults.FAIL


def test_base_executor_get_status_and_next_node_name_gets_next_if_success(mocker, monkeypatch):
    mock_node = mocker.MagicMock()
    mock_run_log_store = mocker.MagicMock()
    mock_dag = mocker.MagicMock()
    mock_step_log = mocker.MagicMock()

    mock_run_log_store.get_step_log.return_value = mock_step_log
    mock_step_log.status = defaults.SUCCESS
    mock_node.get_next_node.return_value = 'next node'

    base_executor = executor.BaseExecutor(config=None)
    base_executor.run_log_store = mock_run_log_store

    status, next_node = base_executor.get_status_and_next_node_name(current_node=mock_node, dag=mock_dag)
    assert status == defaults.SUCCESS
    assert next_node == 'next node'


def test_base_executor_get_status_and_next_node_gets_global_failure_node_by_default_if_step_fails(mocker, monkeypatch):
    mock_node = mocker.MagicMock()
    mock_run_log_store = mocker.MagicMock()
    mock_dag = mocker.MagicMock()
    mock_step_log = mocker.MagicMock()
    mock_fail_node = mocker.MagicMock()

    mock_run_log_store.get_step_log.return_value = mock_step_log
    mock_step_log.status = defaults.FAIL
    mock_node.get_on_failure_node.return_value = None
    mock_dag.get_fail_node.return_value = mock_fail_node
    mock_fail_node.name = 'global fail node'

    base_executor = executor.BaseExecutor(config=None)
    base_executor.run_log_store = mock_run_log_store

    status, next_node = base_executor.get_status_and_next_node_name(current_node=mock_node, dag=mock_dag)
    assert status == defaults.FAIL
    assert next_node == 'global fail node'


def test_base_executor_get_status_and_next_node_gets_node_failure_node_if_provided_if_step_fails(mocker, monkeypatch):
    mock_node = mocker.MagicMock()
    mock_run_log_store = mocker.MagicMock()
    mock_dag = mocker.MagicMock()
    mock_step_log = mocker.MagicMock()

    mock_run_log_store.get_step_log.return_value = mock_step_log
    mock_step_log.status = defaults.FAIL
    mock_node.get_on_failure_node.return_value = 'node fail node'

    base_executor = executor.BaseExecutor(config=None)
    base_executor.run_log_store = mock_run_log_store

    status, next_node = base_executor.get_status_and_next_node_name(current_node=mock_node, dag=mock_dag)
    assert status == defaults.FAIL
    assert next_node == 'node fail node'


def test_base_executor_is_eligible_for_rerun_returns_true_if_no_previous_run_log():
    base_executor = executor.BaseExecutor(config=None)

    base_executor.previous_run_log = None

    assert base_executor.is_eligible_for_rerun(node=None)


def test_base_executor_is_eligible_for_rerun_returns_true_if_step_log_not_found(mocker, monkeypatch):
    mock_node = mocker.MagicMock()
    mock_previous_run_log = mocker.MagicMock()
    mock_search_step_by_internal_name = mocker.MagicMock(side_effect=exceptions.StepLogNotFoundError(
        run_id='id', name='hi'))

    mock_previous_run_log.search_step_by_internal_name = mock_search_step_by_internal_name
    mock_node.get_step_log_name.return_value = 'step_log'

    base_executor = executor.BaseExecutor(config=None)
    base_executor.previous_run_log = mock_previous_run_log

    assert base_executor.is_eligible_for_rerun(node=mock_node)
    mock_search_step_by_internal_name.assert_called_once_with('step_log')


def test_base_executor_is_eligible_for_rerun_returns_false_if_previous_was_success(mocker, monkeypatch):
    mock_node = mocker.MagicMock()
    mock_step_log = mocker.MagicMock()
    mock_previous_node_log = mocker.MagicMock()
    mock_previous_run_log = mocker.MagicMock()
    mock_run_log_store = mocker.MagicMock()

    mock_search_step_by_internal_name = mocker.MagicMock(return_value=(mock_previous_node_log, None))
    mock_run_log_store.get_step_log.return_value = mock_step_log

    mock_previous_node_log.status = defaults.SUCCESS

    mock_previous_run_log.search_step_by_internal_name = mock_search_step_by_internal_name
    mock_node.get_step_log_name.return_value = 'step_log'

    base_executor = executor.BaseExecutor(config=None)
    base_executor.previous_run_log = mock_previous_run_log
    base_executor.run_log_store = mock_run_log_store

    assert base_executor.is_eligible_for_rerun(node=mock_node) is False
    assert mock_step_log.status == defaults.SUCCESS


def test_base_executor_is_eligible_for_rerun_returns_true_if_previous_was_not_success(mocker, monkeypatch):
    mock_node = mocker.MagicMock()
    mock_step_log = mocker.MagicMock()
    mock_previous_node_log = mocker.MagicMock()
    mock_previous_run_log = mocker.MagicMock()
    mock_run_log_store = mocker.MagicMock()

    mock_search_step_by_internal_name = mocker.MagicMock(return_value=(mock_previous_node_log, None))
    mock_run_log_store.get_step_log.return_value = mock_step_log

    mock_previous_node_log.status = defaults.FAIL

    mock_previous_run_log.search_step_by_internal_name = mock_search_step_by_internal_name
    mock_node.get_step_log_name.return_value = 'step_log'

    base_executor = executor.BaseExecutor(config=None)
    base_executor.previous_run_log = mock_previous_run_log
    base_executor.run_log_store = mock_run_log_store

    assert base_executor.is_eligible_for_rerun(node=mock_node)
    assert base_executor.previous_run_log is None


def test_base_executor_execute_graph_breaks_if_node_status_is_triggered(mocker, monkeypatch):
    mock_dag = mocker.MagicMock()
    mock_execute_from_graph = mocker.MagicMock()
    mock_get_status_and_next_node_name = mocker.MagicMock()
    mock_run_log_store = mocker.MagicMock()

    mock_get_status_and_next_node_name.return_value = defaults.TRIGGERED, None

    monkeypatch.setattr(executor.BaseExecutor, 'execute_from_graph', mock_execute_from_graph)
    monkeypatch.setattr(executor.BaseExecutor, 'get_status_and_next_node_name', mock_get_status_and_next_node_name)
    monkeypatch.setattr(executor, 'json', mocker.MagicMock())
    base_executor = executor.BaseExecutor(config=None)
    base_executor.run_log_store = mock_run_log_store

    base_executor.execute_graph(dag=mock_dag)

    assert mock_execute_from_graph.call_count == 1


def test_base_executor_execute_graph_breaks_if_node_status_is_terminal(mocker, monkeypatch):
    mock_dag = mocker.MagicMock()
    mock_execute_from_graph = mocker.MagicMock()
    mock_get_status_and_next_node_name = mocker.MagicMock()
    mock_run_log_store = mocker.MagicMock()
    mock_node = mocker.MagicMock()

    mock_dag.get_node_by_name.return_value = mock_node
    mock_node.node_type = 'success'

    mock_get_status_and_next_node_name.return_value = defaults.SUCCESS, None

    monkeypatch.setattr(executor.BaseExecutor, 'execute_from_graph', mock_execute_from_graph)
    monkeypatch.setattr(executor.BaseExecutor, 'get_status_and_next_node_name', mock_get_status_and_next_node_name)
    monkeypatch.setattr(executor, 'json', mocker.MagicMock())
    base_executor = executor.BaseExecutor(config=None)
    base_executor.run_log_store = mock_run_log_store

    base_executor.execute_graph(dag=mock_dag)

    assert mock_execute_from_graph.call_count == 1


def test_base_executor_resolve_node_config_gives_global_config_if_node_does_not_override(mocker, monkeypatch):
    mock_node = mocker.MagicMock()
    mock_node.get_mode_config.return_value = {}

    base_executor = executor.BaseExecutor(config={'a': 1})

    assert base_executor.resolve_node_config(mock_node) == {'a': 1}


def test_base_executor_resolve_node_config_updates_global_config_if_node_overrides(mocker, monkeypatch):
    mock_node = mocker.MagicMock()
    mock_node.get_mode_config.return_value = {'a': 2}

    base_executor = executor.BaseExecutor(config={'a': 1})

    assert base_executor.resolve_node_config(mock_node) == {'a': 2}


def test_base_executor_resolve_node_config_updates_global_config_if_node_adds(mocker, monkeypatch):
    mock_node = mocker.MagicMock()
    mock_node.get_mode_config.return_value = {'b': 2}

    base_executor = executor.BaseExecutor(config={'a': 1})

    assert base_executor.resolve_node_config(mock_node) == {'a': 1, 'b': 2}


def test_base_executor_resolve_node_config_updates_global_config_from_placeholders(mocker, monkeypatch):
    mock_node = mocker.MagicMock()
    mock_node.get_mode_config.return_value = {'b': 2, 'replace': None}

    config = {
        'a': 1,
        'placeholders': {
            'replace': {'c': 3}
        }

    }
    base_executor = executor.BaseExecutor(config=config)

    assert base_executor.resolve_node_config(mock_node) == {'a': 1, 'c': 3, 'b': 2}


def test_base_executor_resolve_node_supresess_global_config_from_placeholders_if_its_not_mapping(mocker, monkeypatch):
    mock_node = mocker.MagicMock()
    mock_node.get_mode_config.return_value = {'b': 2, 'replace': None}

    config = {
        'a': 1,
        'placeholders': {
            'replace': [1, 2, 3]
        }

    }
    base_executor = executor.BaseExecutor(config=config)

    assert base_executor.resolve_node_config(mock_node) == {'a': 1, 'b': 2}


def test_base_executor_execute_graph_raises_exception_if_loop(mocker, monkeypatch):
    mock_dag = mocker.MagicMock()
    mock_execute_from_graph = mocker.MagicMock()
    mock_get_status_and_next_node_name = mocker.MagicMock()
    mock_run_log_store = mocker.MagicMock()
    mock_node = mocker.MagicMock()

    mock_dag.get_node_by_name.return_value = mock_node

    mock_get_status_and_next_node_name.return_value = defaults.SUCCESS, None

    monkeypatch.setattr(executor.BaseExecutor, 'execute_from_graph', mock_execute_from_graph)
    monkeypatch.setattr(executor.BaseExecutor, 'get_status_and_next_node_name', mock_get_status_and_next_node_name)
    monkeypatch.setattr(executor, 'json', mocker.MagicMock())
    base_executor = executor.BaseExecutor(config=None)
    base_executor.run_log_store = mock_run_log_store
    with pytest.raises(Exception):
        base_executor.execute_graph(dag=mock_dag)


def test_local_executor_is_parallel_execution_sends_defaults_if_not_config():
    local_executor = executor.LocalExecutor(config=None)

    assert defaults.ENABLE_PARALLEL == local_executor.is_parallel_execution()


def test_local_executor_is_parallel_execution_sends_from_config_if_present():
    config = {'enable_parallel': 'true'}

    local_executor = executor.LocalExecutor(config=config)

    assert local_executor.is_parallel_execution()


def test_local_executor_trigger_job_calls(mocker, monkeypatch):
    mock_node = mocker.MagicMock()
    mock_prepare_for_node_execution = mocker.MagicMock()
    mock_execute_node = mocker.MagicMock()

    monkeypatch.setattr(executor.LocalExecutor, 'prepare_for_node_execution', mock_prepare_for_node_execution)
    monkeypatch.setattr(executor.LocalExecutor, 'execute_node', mock_execute_node)

    local_executor = executor.LocalExecutor(config=None)

    local_executor.trigger_job(mock_node)
    assert mock_prepare_for_node_execution.call_count == 1
    assert mock_execute_node.call_count == 1


def test_local_container_executor_is_parallel_execution_sends_defaults_if_not_config():
    local_container_executor = executor.LocalContainerExecutor(config=None)

    assert defaults.ENABLE_PARALLEL == local_container_executor.is_parallel_execution()


def test_local_container_executor_is_parallel_execution_sends_from_config_if_present():
    config = {'enable_parallel': 'true'}

    local_container_executor = executor.LocalContainerExecutor(config=config)

    assert local_container_executor.is_parallel_execution()


def test_local_container_executor_docker_image_is_none_if_not_present_in_config():
    config = {'enable_parallel': 'true'}

    local_container_executor = executor.LocalContainerExecutor(config=config)

    assert local_container_executor.docker_image is None


def test_local_container_executor_docker_image_is_none_if_no_config():
    config = None

    local_container_executor = executor.LocalContainerExecutor(config=config)

    assert local_container_executor.docker_image is None


def test_local_container_executor_docker_image_is_retrieved_from_config():
    config = {'enable_parallel': 'true', 'docker_image': 'docker'}

    local_container_executor = executor.LocalContainerExecutor(config=config)

    assert local_container_executor.docker_image is 'docker'


def test_local_container_executor_add_code_ids_calls_super(mocker, monkeypatch):
    mock_node = mocker.MagicMock()
    mock_node.get_mode_config.return_value = {}

    mock_super_add_code_ids = mocker.MagicMock()
    monkeypatch.setattr(executor.BaseExecutor, 'add_code_identities', mock_super_add_code_ids)

    local_container_executor = executor.LocalContainerExecutor(config={})
    local_container_executor.add_code_identities(node=mock_node, step_log={})

    mock_super_add_code_ids.assert_called_once_with(mock_node, {})


def test_local_container_executor_add_code_ids_uses_global_docker_image(mocker, monkeypatch):
    mock_super_add_code_ids = mocker.MagicMock()
    monkeypatch.setattr(executor.BaseExecutor, 'add_code_identities', mock_super_add_code_ids)

    mock_node = mocker.MagicMock()
    mock_node.get_mode_config.return_value = {}

    mock_get_local_docker_image_id = mocker.MagicMock()
    monkeypatch.setattr(executor.utils, 'get_local_docker_image_id', mock_get_local_docker_image_id)

    mock_run_log_store = mocker.MagicMock()
    mock_step_log = mocker.MagicMock()
    local_container_executor = executor.LocalContainerExecutor(config={'docker_image': 'global'})
    local_container_executor.run_log_store = mock_run_log_store

    local_container_executor.add_code_identities(node=mock_node, step_log=mock_step_log)

    mock_get_local_docker_image_id.assert_called_once_with('global')


def test_local_container_executor_add_code_ids_uses_local_docker_image_if_provided(mocker, monkeypatch):
    mock_super_add_code_ids = mocker.MagicMock()
    monkeypatch.setattr(executor.BaseExecutor, 'add_code_identities', mock_super_add_code_ids)

    mock_node = mocker.MagicMock()
    mock_node.get_mode_config.return_value = {'docker_image': 'local'}

    mock_get_local_docker_image_id = mocker.MagicMock()
    monkeypatch.setattr(executor.utils, 'get_local_docker_image_id', mock_get_local_docker_image_id)

    mock_run_log_store = mocker.MagicMock()
    mock_step_log = mocker.MagicMock()
    local_container_executor = executor.LocalContainerExecutor(config={'docker_image': 'global'})
    local_container_executor.run_log_store = mock_run_log_store

    local_container_executor.add_code_identities(node=mock_node, step_log=mock_step_log)

    mock_get_local_docker_image_id.assert_called_once_with('local')


def test_local_container_executor_calls_spin_container_during_trigger_job(mocker, monkeypatch):
    mock_spin_container = mocker.MagicMock()
    mock_step_log = mocker.MagicMock()
    mock_run_log_store = mocker.MagicMock()
    mock_node = mocker.MagicMock()

    mock_run_log_store.get_step_log.return_value = mock_step_log
    mock_step_log.status = defaults.SUCCESS

    monkeypatch.setattr(executor.LocalContainerExecutor, '_spin_container', mock_spin_container)

    local_container_executor = executor.LocalContainerExecutor(config=None)
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

    monkeypatch.setattr(executor.LocalContainerExecutor, '_spin_container', mock_spin_container)

    local_container_executor = executor.LocalContainerExecutor(config=None)
    local_container_executor.run_log_store = mock_run_log_store

    local_container_executor.trigger_job(node=mock_node)

    assert mock_step_log.status == defaults.FAIL
