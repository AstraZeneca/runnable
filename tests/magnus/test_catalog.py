import pytest

from magnus import catalog  # pylint: disable=import-error
from magnus import pipeline  # pylint: disable=import-error
from magnus import defaults  # pylint: disable=import-error


def test_get_run_log_store_returns_global_executor_run_log_store(mocker, monkeypatch):
    mock_global_executor = mocker.MagicMock()
    mock_global_executor.run_log_store = 'RunLogStore'

    monkeypatch.setattr(pipeline, 'global_executor', mock_global_executor)
    run_log_store = catalog.get_run_log_store()

    assert run_log_store == 'RunLogStore'


def test_is_catalog_out_of_sync_returns_true_for_empty_synced_catalogs():
    assert catalog.is_catalog_out_of_sync(1, []) is True


def test_is_catalog_out_of_sync_returns_false_for_same_objects():
    class MockCatalog:
        catalog_relative_path = None
        data_hash = None

    catalog_item = MockCatalog()
    catalog_item.catalog_relative_path = 'path'
    catalog_item.data_hash = 'hash'

    synced_catalog = [catalog_item]
    assert catalog.is_catalog_out_of_sync(catalog_item, synced_catalog) is False


def test_is_catalog_out_of_sync_returns_true_for_different_hash():
    class MockCatalog:
        catalog_relative_path = None
        data_hash = None

    catalog_item1 = MockCatalog()
    catalog_item1.catalog_relative_path = 'path'
    catalog_item1.data_hash = 'hash'

    catalog_item2 = MockCatalog()
    catalog_item2.catalog_relative_path = 'path'
    catalog_item2.data_hash = 'not-hash'

    synced_catalog = [catalog_item1]
    assert catalog.is_catalog_out_of_sync(catalog_item2, synced_catalog) is True


def test_is_catalog_out_of_sync_returns_true_for_different_paths():
    class MockCatalog:
        catalog_relative_path = None
        data_hash = None

    catalog_item1 = MockCatalog()
    catalog_item1.catalog_relative_path = 'path'
    catalog_item1.data_hash = 'hash'

    catalog_item2 = MockCatalog()
    catalog_item2.catalog_relative_path = 'path1'
    catalog_item2.data_hash = 'hash'

    synced_catalog = [catalog_item1]
    assert catalog.is_catalog_out_of_sync(catalog_item2, synced_catalog) is True


def test_base_catalog_inits_empty_config_if_none_config():
    base_catalog = catalog.BaseCatalog(config=None)
    assert base_catalog.config == {}


def test_base_catalog_get_raises_exception():
    base_catalog = catalog.BaseCatalog(config=None)
    with pytest.raises(NotImplementedError):
        base_catalog.get(name='test', run_id='test')


def test_base_catalog_put_raises_exception():
    base_catalog = catalog.BaseCatalog(config=None)
    with pytest.raises(NotImplementedError):
        base_catalog.put(name='test', run_id='test')


def test_base_catalog_sync_between_runs_raises_exception():
    base_catalog = catalog.BaseCatalog(config=None)
    with pytest.raises(NotImplementedError):
        base_catalog.sync_between_runs(previous_run_id=1, run_id=2)


def test_base_catalog_inits_default_compute_folder_if_none_config():
    base_catalog = catalog.BaseCatalog(config=None)
    assert base_catalog.compute_data_folder == defaults.COMPUTE_DATA_FOLDER


def test_do_nothing_catalog_get_returns_empty_list(monkeypatch, mocker):
    mock_base_catalog = mocker.MagicMock()

    monkeypatch.setattr(catalog, 'BaseCatalog', mock_base_catalog)

    catalog_handler = catalog.DoNothingCatalog(config=None)
    assert catalog_handler.get(name='does not matter', run_id='none') == []


def test_do_nothing_catalog_put_returns_empty_list(monkeypatch, mocker):
    mock_base_catalog = mocker.MagicMock()

    monkeypatch.setattr(catalog, 'BaseCatalog', mock_base_catalog)

    catalog_handler = catalog.DoNothingCatalog(config=None)
    assert catalog_handler.put(name='does not matter', run_id='none') == []


def test_file_system_catalog_get_catalog_location_defaults_if_location_not_provided(monkeypatch, mocker):
    mock_base_catalog = mocker.MagicMock()

    monkeypatch.setattr(catalog, 'BaseCatalog', mock_base_catalog)

    catalog_handler = catalog.FileSystemCatalog(config=None)

    assert catalog_handler.get_catalog_location() == defaults.CATALOG_LOCATION_FOLDER


def test_file_system_catalog_get_catalog_location_returns_config_catalog_location_if_provided(monkeypatch, mocker):
    mock_base_catalog = mocker.MagicMock()

    monkeypatch.setattr(catalog, 'BaseCatalog', mock_base_catalog)

    catalog_handler = catalog.FileSystemCatalog(config={'type': 'file-system',
                                                        'catalog_location': 'this'})

    assert catalog_handler.get_catalog_location() == 'this'


def test_file_system_catalog_get_creates_catalog_location_using_run_id(monkeypatch, mocker):
    mock_does_dir_exist = mocker.MagicMock(side_effect=[True, Exception()])
    monkeypatch.setattr(catalog.utils, 'does_dir_exist', mock_does_dir_exist)
    monkeypatch.setattr(catalog.FileSystemCatalog, 'get_catalog_location',
                        mocker.MagicMock(return_value='this_location'))

    monkeypatch.setattr(catalog, 'BaseCatalog', mocker.MagicMock())
    catalog_handler = catalog.FileSystemCatalog(config={'type': 'file-system'})
    with pytest.raises(Exception):
        catalog_handler.get('testing', run_id='dummy_run_id')

    mock_does_dir_exist.assert_any_call(catalog.Path('this_location') / 'dummy_run_id')


def test_file_system_catalog_get_uses_compute_data_folder_provided(monkeypatch, mocker):
    mock_does_dir_exist = mocker.MagicMock(side_effect=Exception())
    monkeypatch.setattr(catalog.utils, 'does_dir_exist', mock_does_dir_exist)
    monkeypatch.setattr(catalog.FileSystemCatalog, 'get_catalog_location',
                        mocker.MagicMock(return_value='this_location'))
    monkeypatch.setattr(catalog, 'BaseCatalog', mocker.MagicMock())

    catalog_handler = catalog.FileSystemCatalog(config={'type': 'file-system'})
    with pytest.raises(Exception):
        catalog_handler.get('testing', run_id='dummy_run_id', compute_data_folder='this_compute_folder')

    mock_does_dir_exist.assert_called_once_with('this_compute_folder')


def test_file_system_catalog_get_raises_exception_if_compute_data_folder_does_not_exist(monkeypatch, mocker):
    mock_does_dir_exist = mocker.MagicMock(return_value=False)
    monkeypatch.setattr(catalog.utils, 'does_dir_exist', mock_does_dir_exist)

    monkeypatch.setattr(catalog, 'BaseCatalog', mocker.MagicMock())

    catalog_handler = catalog.FileSystemCatalog(config={'type': 'file-system'})
    with pytest.raises(Exception):
        catalog_handler.get('testing', run_id='dummy_run_id', compute_data_folder='this_compute_folder')


def test_file_system_catalog_get_raises_exception_if_catalog_does_not_exist(monkeypatch, mocker):
    # mock_does_dir_exist = mocker.MagicMock(return_value=[True, False]) # Should be better than this
    def mock_does_dir_exist(dir_name):
        if dir_name == 'this_compute_folder':
            return True
        return False
    monkeypatch.setattr(catalog.utils, 'does_dir_exist', mock_does_dir_exist)
    monkeypatch.setattr(catalog.FileSystemCatalog, 'get_catalog_location',
                        mocker.MagicMock(return_value='this_location'))

    monkeypatch.setattr(catalog, 'BaseCatalog', mocker.MagicMock())

    catalog_handler = catalog.FileSystemCatalog(config={'type': 'file-system'})
    with pytest.raises(Exception):
        catalog_handler.get('testing', run_id='dummy_run_id', compute_data_folder='this_compute_folder')


def test_file_system_catalog_get_copies_files_from_catalog_to_compute_folder(mocker, monkeypatch):
    monkeypatch.setattr(catalog.utils, 'safe_make_dir', mocker.MagicMock())
    monkeypatch.setattr(catalog.utils, 'does_dir_exist', mocker.MagicMock(return_value=True))
    monkeypatch.setattr(catalog.utils, 'remove_prefix', mocker.MagicMock())
    monkeypatch.setattr(catalog.utils, 'get_data_hash', mocker.MagicMock())

    monkeypatch.setattr(catalog.FileSystemCatalog, 'get_catalog_location',
                        mocker.MagicMock(return_value='this_location'))

    class MockFile:
        def __init__(self, name):
            self.name = name

    file1 = MockFile('file1')
    file2 = MockFile('file2')
    mock_glob = mocker.MagicMock(return_value=[file1, file2])
    monkeypatch.setattr(catalog.Path, 'glob', mock_glob)

    monkeypatch.setattr(catalog, 'get_run_log_store', mocker.MagicMock())

    mock_copy = mocker.MagicMock()
    monkeypatch.setattr(catalog.shutil, 'copy', mock_copy)
    monkeypatch.setattr(catalog, 'BaseCatalog', mocker.MagicMock())

    catalog_handler = catalog.FileSystemCatalog(config={'type': 'file-system'})
    catalog_handler.get('testing', run_id='dummy_run_id')

    assert mock_copy.call_count == 2
    mock_copy.assert_any_call(file1, 'data')
    mock_copy.assert_any_call(file2, 'data')


def test_file_system_catalog_put_copies_files_from_compute_folder_to_catalog_if_synced_changed(mocker, monkeypatch):
    monkeypatch.setattr(catalog.utils, 'safe_make_dir', mocker.MagicMock())
    monkeypatch.setattr(catalog.utils, 'does_dir_exist', mocker.MagicMock(return_value=True))
    monkeypatch.setattr(catalog.utils, 'remove_prefix', mocker.MagicMock())
    monkeypatch.setattr(catalog.utils, 'get_data_hash', mocker.MagicMock())

    monkeypatch.setattr(catalog.FileSystemCatalog, 'get_catalog_location',
                        mocker.MagicMock(return_value='this_location'))

    class MockFile:
        def __init__(self, name):
            self.name = name

    file1 = MockFile('file1')
    file2 = MockFile('file2')
    mock_glob = mocker.MagicMock(return_value=[file1, file2])
    monkeypatch.setattr(catalog.Path, 'glob', mock_glob)

    monkeypatch.setattr(catalog, 'get_run_log_store', mocker.MagicMock())
    monkeypatch.setattr(catalog, 'is_catalog_out_of_sync', mocker.MagicMock(return_value=True))

    mock_copy = mocker.MagicMock()
    monkeypatch.setattr(catalog.shutil, 'copy', mock_copy)
    monkeypatch.setattr(catalog, 'BaseCatalog', mocker.MagicMock())

    catalog_handler = catalog.FileSystemCatalog(config={'type': 'file-system'})
    catalog_handler.put('testing', run_id='dummy_run_id')

    assert mock_copy.call_count == 2
    mock_copy.assert_any_call(file1, catalog.Path('this_location') / 'dummy_run_id')
    mock_copy.assert_any_call(file2, catalog.Path('this_location') / 'dummy_run_id')


def test_file_system_catalog_put_does_not_copy_files_from_compute_folder_to_catalog_if_synced_same(mocker, monkeypatch):
    monkeypatch.setattr(catalog.utils, 'safe_make_dir', mocker.MagicMock())
    monkeypatch.setattr(catalog.utils, 'does_dir_exist', mocker.MagicMock(return_value=True))
    monkeypatch.setattr(catalog.utils, 'remove_prefix', mocker.MagicMock())
    monkeypatch.setattr(catalog.utils, 'get_data_hash', mocker.MagicMock())

    monkeypatch.setattr(catalog.FileSystemCatalog, 'get_catalog_location',
                        mocker.MagicMock(return_value='this_location'))

    class MockFile:
        def __init__(self, name):
            self.name = name

    file1 = MockFile('file1')
    file2 = MockFile('file2')
    mock_glob = mocker.MagicMock(return_value=[file1, file2])
    monkeypatch.setattr(catalog.Path, 'glob', mock_glob)

    monkeypatch.setattr(catalog, 'get_run_log_store', mocker.MagicMock())
    monkeypatch.setattr(catalog, 'is_catalog_out_of_sync', mocker.MagicMock(return_value=False))

    mock_copy = mocker.MagicMock()
    monkeypatch.setattr(catalog.shutil, 'copy', mock_copy)
    monkeypatch.setattr(catalog, 'BaseCatalog', mocker.MagicMock())

    catalog_handler = catalog.FileSystemCatalog(config={'type': 'file-system'})
    catalog_handler.put('testing', run_id='dummy_run_id')

    assert mock_copy.call_count == 0


def test_file_system_catalog_put_uses_compute_folder_by_default(monkeypatch, mocker):
    mock_safe_make_dir = mocker.MagicMock()
    monkeypatch.setattr(catalog.utils, 'safe_make_dir', mock_safe_make_dir)

    mock_does_dir_exist = mocker.MagicMock(side_effect=Exception())
    monkeypatch.setattr(catalog.utils, 'does_dir_exist', mock_does_dir_exist)
    monkeypatch.setattr(catalog.FileSystemCatalog, 'get_catalog_location',
                        mocker.MagicMock(return_value='this_location'))
    monkeypatch.setattr(catalog, 'BaseCatalog', mocker.MagicMock())

    catalog_handler = catalog.FileSystemCatalog(config={'type': 'file-system'})
    with pytest.raises(Exception):
        catalog_handler.put('testing', run_id='dummy_run_id')

    mock_does_dir_exist.assert_called_once_with('data')


def test_file_system_catalog_put_uses_compute_folder_provided(monkeypatch, mocker):
    mock_safe_make_dir = mocker.MagicMock()
    monkeypatch.setattr(catalog.utils, 'safe_make_dir', mock_safe_make_dir)

    mock_does_dir_exist = mocker.MagicMock(side_effect=Exception())
    monkeypatch.setattr(catalog.utils, 'does_dir_exist', mock_does_dir_exist)
    monkeypatch.setattr(catalog.FileSystemCatalog, 'get_catalog_location',
                        mocker.MagicMock(return_value='this_location'))
    monkeypatch.setattr(catalog, 'BaseCatalog', mocker.MagicMock())

    catalog_handler = catalog.FileSystemCatalog(config={'type': 'file-system'})
    with pytest.raises(Exception):
        catalog_handler.put('testing', run_id='dummy_run_id', compute_data_folder='not_data')

    mock_does_dir_exist.assert_called_once_with('not_data')


def test_file_system_catalog_put_raises_exception_if_compute_data_folder_does_not_exist(monkeypatch, mocker):
    mock_safe_make_dir = mocker.MagicMock()
    monkeypatch.setattr(catalog.utils, 'safe_make_dir', mock_safe_make_dir)

    mock_does_dir_exist = mocker.MagicMock(return_value=False)
    monkeypatch.setattr(catalog.utils, 'does_dir_exist', mock_does_dir_exist)
    monkeypatch.setattr(catalog.FileSystemCatalog, 'get_catalog_location',
                        mocker.MagicMock(return_value='this_location'))
    monkeypatch.setattr(catalog, 'BaseCatalog', mocker.MagicMock())

    catalog_handler = catalog.FileSystemCatalog(config={'type': 'file-system'})
    with pytest.raises(Exception):
        catalog_handler.put('testing', run_id='dummy_run_id', compute_data_folder='this_compute_folder')


def test_file_system_catalog_put_creates_catalog_location_using_run_id(monkeypatch, mocker):
    mock_safe_make_dir = mocker.MagicMock()
    monkeypatch.setattr(catalog.utils, 'safe_make_dir', mock_safe_make_dir)

    mock_does_dir_exist = mocker.MagicMock(side_effect=Exception())
    monkeypatch.setattr(catalog.utils, 'does_dir_exist', mock_does_dir_exist)
    monkeypatch.setattr(catalog.FileSystemCatalog, 'get_catalog_location',
                        mocker.MagicMock(return_value='this_location'))
    monkeypatch.setattr(catalog, 'BaseCatalog', mocker.MagicMock())

    catalog_handler = catalog.FileSystemCatalog(config={'type': 'file-system'})
    with pytest.raises(Exception):
        catalog_handler.put('testing', run_id='dummy_run_id')

    mock_safe_make_dir.assert_called_once_with(catalog.Path('this_location') / 'dummy_run_id')


def test_file_system_sync_between_runs_raises_exception_if_previous_catalog_does_not_exist(monkeypatch, mocker):
    mock_safe_make_dir = mocker.MagicMock()
    monkeypatch.setattr(catalog.utils, 'safe_make_dir', mock_safe_make_dir)

    mock_does_dir_exist = mocker.MagicMock(return_value=False)
    monkeypatch.setattr(catalog.utils, 'does_dir_exist', mock_does_dir_exist)
    monkeypatch.setattr(catalog.FileSystemCatalog, 'get_catalog_location',
                        mocker.MagicMock(return_value='this_location'))
    monkeypatch.setattr(catalog, 'BaseCatalog', mocker.MagicMock())

    catalog_handler = catalog.FileSystemCatalog(config={'type': 'file-system'})
    with pytest.raises(Exception):
        catalog_handler.sync_between_runs('previous', 'current')


# def test_file_system_sync_between_runs_copies_all_files_from_old_to_new(monkeypatch, mocker):
#     monkeypatch.setattr(catalog.utils, 'safe_make_dir', mocker.MagicMock())

#     mock_does_dir_exist = mocker.MagicMock(return_value=True)
#     monkeypatch.setattr(catalog.utils, 'does_dir_exist', mock_does_dir_exist)

#     monkeypatch.setattr(catalog.FileSystemCatalog, 'get_catalog_location',
#                         mocker.MagicMock(return_value='this_location'))
#     monkeypatch.setattr(catalog, 'BaseCatalog', mocker.MagicMock())

#     mock_path = mocker.MagicMock()
#     monkeypatch.setattr(catalog, 'Path', mock_path)
#     mock_path.glob.return_value = ['file1', 'file2']

#     mock_copy = mocker.MagicMock()
#     monkeypatch.setattr(catalog.shutil, 'copy', mock_copy)

#     catalog_handler = catalog.FileSystemCatalog(config={'type': 'file-system'})

#     print(mock_path.__dict__)
#     assert mock_copy.call_count == 2
