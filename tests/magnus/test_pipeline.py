import pytest

from magnus import pipeline
from magnus import defaults


def test_load_user_extensions_does_nothing_if_no_extensions_provided(monkeypatch, mocker):
    mock_utils = mocker.MagicMock()
    mock_utils.does_file_exist = mocker.MagicMock(return_value=False)
    monkeypatch.setattr(pipeline, 'utils', mock_utils)

    pipeline.load_user_extensions()


def test_load_user_extensions_reads_extension_file(monkeypatch, mocker):
    mock_utils = mocker.MagicMock()
    mock_load_yaml = mocker.MagicMock()

    mock_utils.does_file_exist = mocker.MagicMock(return_value=True)
    mock_utils.load_yaml = mock_load_yaml

    monkeypatch.setattr(pipeline, 'utils', mock_utils)

    pipeline.load_user_extensions()
    mock_load_yaml.assert_called_once_with(defaults.USER_CONFIG_FILE)


def test_send_return_code_does_nothing_if_success(mocker, monkeypatch):
    mock_mode_executor = mocker.MagicMock()
    mock_run_log = mocker.MagicMock()

    mock_mode_executor.run_log_store.get_run_log_by_id.return_value = mock_run_log
    mock_run_log.status = defaults.SUCCESS

    pipeline.send_return_code(mock_mode_executor)


def test_send_return_code_raises_exception_if_failure(mocker, monkeypatch):
    mock_mode_executor = mocker.MagicMock()
    mock_run_log = mocker.MagicMock()

    mock_mode_executor.run_log_store.get_run_log_by_id.return_value = mock_run_log
    mock_run_log.status = defaults.FAIL

    with pytest.raises(Exception):
        pipeline.send_return_code(mock_mode_executor)


def test_prepare_configurations_does_not_apply_variables_if_none_sent(mocker, monkeypatch):
    mock_utils = mocker.MagicMock()
    mock_load_yaml = mocker.MagicMock(return_value={'dag': 1})

    mock_utils.load_yaml = mock_load_yaml

    monkeypatch.setattr(pipeline, 'utils', mock_utils)
    monkeypatch.setattr(pipeline, 'json', mocker.MagicMock())
    monkeypatch.setattr(pipeline, 'graph', mocker.MagicMock())

    pipeline.prepare_configurations(variables_file='', pipeline_file='', run_id=1, tag='tag', use_cached=False)
    assert mock_load_yaml.call_count == 1


def test_prepare_configurations_apply_variables_if_sent(mocker, monkeypatch):
    mock_utils = mocker.MagicMock()
    mock_load_yaml = mocker.MagicMock(return_value={'dag': 1})

    mock_utils.load_yaml = mock_load_yaml

    monkeypatch.setattr(pipeline, 'utils', mock_utils)
    monkeypatch.setattr(pipeline, 'json', mocker.MagicMock())
    monkeypatch.setattr(pipeline, 'graph', mocker.MagicMock())

    pipeline.prepare_configurations(variables_file='var', pipeline_file='', run_id=1, tag='tag', use_cached=False)
    assert mock_load_yaml.call_count == 2


def test_prepare_configurations_stores_dag_hash_and_graph_to_executor(mocker, monkeypatch):
    mock_utils = mocker.MagicMock()
    mock_utils.get_dag_hash.return_value = 'hash'

    mock_executor_obj = mocker.MagicMock()

    def mock_get_provider_by_name_and_type(service_type, *args, **kwargs):
        if service_type == 'executor':
            return mock_executor_obj
        return mocker.MagicMock()

    mock_utils.get_provider_by_name_and_type = mock_get_provider_by_name_and_type

    mock_graph = mocker.MagicMock()
    mock_graph.create_graph.return_value = 1

    monkeypatch.setattr(pipeline, 'utils', mock_utils)
    monkeypatch.setattr(pipeline, 'json', mocker.MagicMock())
    monkeypatch.setattr(pipeline, 'graph', mock_graph)

    pipeline.prepare_configurations(variables_file='var', pipeline_file='', run_id=1, tag='tag', use_cached=False)

    assert mock_executor_obj.dag_hash == 'hash'
    assert mock_executor_obj.dag == 1


# def test_prepare_configurations_uses_empty_dict_for_run_log_config_by_default(mocker, monkeypatch):
#     mock_utils = mocker.MagicMock()
#     #mock_utils.apply_variables.return_value = {'dag': 'iamadag'}

#     mock_service = mocker.MagicMock()

#     class MockProvider:
#         sent_in_config = None

#         @classmethod
#         def mock_get_provider_by_name_and_type(cls, service_type, service_config):
#             if service_type == 'run_log_store':
#                 cls.sent_in_config = service_config
#                 return mock_service
#             return mocker.MagicMock()

#     mock_utils.get_provider_by_name_and_type = MockProvider.mock_get_provider_by_name_and_type

#     monkeypatch.setattr(pipeline, 'utils', mock_utils)
#     monkeypatch.setattr(pipeline, 'json', mocker.MagicMock())
#     monkeypatch.setattr(pipeline, 'graph', mocker.MagicMock())

#     pipeline.prepare_configurations(variables_file='', pipeline_file='', run_id=1, tag='tag', use_cached=False)

#     assert MockProvider.sent_in_config == defaults.DEFAULT_RUN_LOG_STORE


# def test_prepare_configurations_uses_config_if_passed(mocker, monkeypatch):
#     mock_utils = mocker.MagicMock()
#     mock_utils.apply_variables.return_value = {'dag': 'iamadag', 'run_log': 'run_log_config'}

#     mock_data_store = mocker.MagicMock()
#     mock_data_store_get = mocker.MagicMock()
#     mock_data_store.get_run_log_store = mock_data_store_get

#     mock_executor = mocker.MagicMock()

#     monkeypatch.setattr(pipeline, 'utils', mock_utils)
#     monkeypatch.setattr(pipeline, 'json', mocker.MagicMock())
#     monkeypatch.setattr(pipeline, 'graph', mocker.MagicMock())
#     monkeypatch.setattr(pipeline, 'datastore', mock_data_store)
#     monkeypatch.setattr(pipeline, 'catalog', mocker.MagicMock())
#     monkeypatch.setattr(pipeline, 'secrets', mocker.MagicMock())
#     monkeypatch.setattr(pipeline, 'executor', mock_executor)

#     pipeline.prepare_configurations(variables_file='', pipeline_file='', run_id=1, tag='tag', use_cached=False)

#     mock_data_store_get.assert_called_once_with(config='run_log_config')


# def test_prepare_configurations_uses_empty_dict_for_catalog_by_default(mocker, monkeypatch):
#     mock_utils = mocker.MagicMock()
#     mock_utils.apply_variables.return_value = {'dag': 'iamadag'}

#     mock_catalog = mocker.MagicMock()
#     mock_catalog_get = mocker.MagicMock()
#     mock_catalog.get_catalog_handler = mock_catalog_get

#     mock_executor = mocker.MagicMock()

#     monkeypatch.setattr(pipeline, 'utils', mock_utils)
#     monkeypatch.setattr(pipeline, 'json', mocker.MagicMock())
#     monkeypatch.setattr(pipeline, 'graph', mocker.MagicMock())
#     monkeypatch.setattr(pipeline, 'datastore', mocker.MagicMock())
#     monkeypatch.setattr(pipeline, 'catalog', mock_catalog)
#     monkeypatch.setattr(pipeline, 'secrets', mocker.MagicMock())
#     monkeypatch.setattr(pipeline, 'executor', mock_executor)

#     pipeline.prepare_configurations(variables_file='', pipeline_file='', run_id=1, tag='tag', use_cached=False)

#     mock_catalog_get.assert_called_once_with({})


# def test_prepare_configurations_uses_config_for_catalog_if_provided(mocker, monkeypatch):
#     mock_utils = mocker.MagicMock()
#     mock_utils.apply_variables.return_value = {'dag': 'iamadag', 'catalog': 'catalog_config'}

#     mock_catalog = mocker.MagicMock()
#     mock_catalog_get = mocker.MagicMock()
#     mock_catalog.get_catalog_handler = mock_catalog_get

#     mock_executor = mocker.MagicMock()

#     monkeypatch.setattr(pipeline, 'utils', mock_utils)
#     monkeypatch.setattr(pipeline, 'json', mocker.MagicMock())
#     monkeypatch.setattr(pipeline, 'graph', mocker.MagicMock())
#     monkeypatch.setattr(pipeline, 'datastore', mocker.MagicMock())
#     monkeypatch.setattr(pipeline, 'catalog', mock_catalog)
#     monkeypatch.setattr(pipeline, 'secrets', mocker.MagicMock())
#     monkeypatch.setattr(pipeline, 'executor', mock_executor)

#     pipeline.prepare_configurations(variables_file='', pipeline_file='', run_id=1, tag='tag', use_cached=False)

#     mock_catalog_get.assert_called_once_with('catalog_config')


# def test_prepare_configurations_uses_empty_dict_for_secrets_by_default(mocker, monkeypatch):
#     mock_utils = mocker.MagicMock()
#     mock_utils.apply_variables.return_value = {'dag': 'iamadag'}

#     mock_secrets = mocker.MagicMock()
#     mock_secrets_get = mocker.MagicMock()
#     mock_secrets.get_secrets_handler = mock_secrets_get

#     mock_executor = mocker.MagicMock()

#     monkeypatch.setattr(pipeline, 'utils', mock_utils)
#     monkeypatch.setattr(pipeline, 'json', mocker.MagicMock())
#     monkeypatch.setattr(pipeline, 'graph', mocker.MagicMock())
#     monkeypatch.setattr(pipeline, 'datastore', mocker.MagicMock())
#     monkeypatch.setattr(pipeline, 'catalog', mocker.MagicMock())
#     monkeypatch.setattr(pipeline, 'secrets', mock_secrets)
#     monkeypatch.setattr(pipeline, 'executor', mock_executor)

#     pipeline.prepare_configurations(variables_file='', pipeline_file='', run_id=1, tag='tag', use_cached=False)

#     mock_secrets_get.assert_called_once_with({})


# def test_prepare_configurations_uses_config_for_secrets_if_provided(mocker, monkeypatch):
#     mock_utils = mocker.MagicMock()
#     mock_utils.apply_variables.return_value = {'dag': 'iamadag', 'secrets': 'secrets_config'}

#     mock_secrets = mocker.MagicMock()
#     mock_secrets_get = mocker.MagicMock()
#     mock_secrets.get_secrets_handler = mock_secrets_get

#     mock_executor = mocker.MagicMock()

#     monkeypatch.setattr(pipeline, 'utils', mock_utils)
#     monkeypatch.setattr(pipeline, 'json', mocker.MagicMock())
#     monkeypatch.setattr(pipeline, 'graph', mocker.MagicMock())
#     monkeypatch.setattr(pipeline, 'datastore', mocker.MagicMock())
#     monkeypatch.setattr(pipeline, 'catalog', mocker.MagicMock())
#     monkeypatch.setattr(pipeline, 'secrets', mock_secrets)
#     monkeypatch.setattr(pipeline, 'executor', mock_executor)

#     pipeline.prepare_configurations(variables_file='', pipeline_file='', run_id=1, tag='tag', use_cached=False)

#     mock_secrets_get.assert_called_once_with('secrets_config')


# def test_prepare_configurations_uses_empty_dict_for_executor_by_default(mocker, monkeypatch):
#     mock_utils = mocker.MagicMock()
#     mock_utils.apply_variables.return_value = {'dag': 'iamadag'}

#     mock_executor = mocker.MagicMock()
#     mock_executor_get = mocker.MagicMock()
#     mock_executor.get_executor_class = mock_executor_get

#     monkeypatch.setattr(pipeline, 'utils', mock_utils)
#     monkeypatch.setattr(pipeline, 'json', mocker.MagicMock())
#     monkeypatch.setattr(pipeline, 'graph', mocker.MagicMock())
#     monkeypatch.setattr(pipeline, 'datastore', mocker.MagicMock())
#     monkeypatch.setattr(pipeline, 'catalog', mocker.MagicMock())
#     monkeypatch.setattr(pipeline, 'secrets', mocker.MagicMock())
#     monkeypatch.setattr(pipeline, 'executor', mock_executor)

#     pipeline.prepare_configurations(variables_file='', pipeline_file='', run_id=1, tag='tag', use_cached=False)

#     mock_executor_get.assert_called_once_with({})


# def test_prepare_configurations_uses_config_for_executor_if_provided(mocker, monkeypatch):
#     mock_utils = mocker.MagicMock()
#     mock_utils.apply_variables.return_value = {'dag': 'iamadag', 'mode': 'mode_config'}

#     mock_executor = mocker.MagicMock()
#     mock_executor_get = mocker.MagicMock()
#     mock_executor.get_executor_class = mock_executor_get

#     monkeypatch.setattr(pipeline, 'utils', mock_utils)
#     monkeypatch.setattr(pipeline, 'json', mocker.MagicMock())
#     monkeypatch.setattr(pipeline, 'graph', mocker.MagicMock())
#     monkeypatch.setattr(pipeline, 'datastore', mocker.MagicMock())
#     monkeypatch.setattr(pipeline, 'catalog', mocker.MagicMock())
#     monkeypatch.setattr(pipeline, 'secrets', mocker.MagicMock())
#     monkeypatch.setattr(pipeline, 'executor', mock_executor)

#     pipeline.prepare_configurations(variables_file='', pipeline_file='', run_id=1, tag='tag', use_cached=False)

#     mock_executor_get.assert_called_once_with('mode_config')


# def test_execute_prepares_a_new_run_id(monkeypatch, mocker):
#     mock_utils = mocker.MagicMock()
#     monkeypatch.setattr(pipeline, 'utils', mock_utils)

#     mock_utils.generate_run_id.return_value = 'new_run_id'

#     mock_prepare_configs = mocker.MagicMock()
#     monkeypatch.setattr(pipeline, 'prepare_configurations', mock_prepare_configs)
#     monkeypatch.setattr(pipeline, 'send_return_code', mocker.MagicMock())

#     pipeline.execute(variables_file='var', pipeline_file='pipe', run_id=1)
#     _, kwargs = mock_prepare_configs.call_args

#     assert kwargs['run_id'] == 'new_run_id'


# def test_execute_calls_graph_prepare_and_execute(mocker, monkeypatch):
#     mock_utils = mocker.MagicMock()
#     monkeypatch.setattr(pipeline, 'utils', mock_utils)

#     mock_executor = mocker.MagicMock()
#     mock_prepare_graph_execution = mocker.MagicMock()
#     mock_execute_graph = mocker.MagicMock()

#     mock_executor.prepare_for_graph_execution = mock_prepare_graph_execution
#     mock_executor.execute_graph = mock_execute_graph

#     mock_prepare_configs = mocker.MagicMock(return_value=mock_executor)
#     monkeypatch.setattr(pipeline, 'prepare_configurations', mock_prepare_configs)
#     monkeypatch.setattr(pipeline, 'send_return_code', mocker.MagicMock())

#     pipeline.execute(variables_file='var', pipeline_file='pipe', run_id=1)

#     mock_prepare_graph_execution.assert_called_once_with()
#     assert mock_execute_graph.call_count == 1
