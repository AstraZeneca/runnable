import pytest

from magnus import datastore


def test_data_catalog_eq_is_equal_if_name_is_same():
    this = datastore.DataCatalog(name='test')
    that = datastore.DataCatalog(name='test')

    assert this == that
    assert len(set([this, that])) == 1


def test_data_catalog_eq_is_not_equal_if_name_is_not_same():
    this = datastore.DataCatalog(name='test')
    that = datastore.DataCatalog(name='test1')

    assert this != that
    assert len(set([this, that])) == 2


def test_data_catalog_eq_is_not_equal_if_objs_are_not_same(mocker):
    this = datastore.DataCatalog(name='test')
    that = mocker.MagicMock()

    assert this != that
    assert len(set([this, that])) == 2


def test_branchlog_get_data_catalogs_by_state_raises_exception_for_incorrect_stage():
    branch_log = datastore.BranchLog(internal_name='test')
    with pytest.raises(Exception):
        branch_log.get_data_catalogs_by_stage(stage='notright')


def test_branchlog_calls_step_logs_catalogs(mocker):
    mock_step = mocker.MagicMock()
    mock_get_data_catalogs_by_stage = mocker.MagicMock()

    mock_step.get_data_catalogs_by_stage = mock_get_data_catalogs_by_stage

    branch_log = datastore.BranchLog(internal_name='test')
    branch_log.steps['test_step'] = mock_step

    branch_log.get_data_catalogs_by_stage(stage='put')

    mock_get_data_catalogs_by_stage.assert_called_once_with(stage='put')


def test_steplog_get_data_catalogs_by_state_raises_exception_for_incorrect_stage():
    step_log = datastore.StepLog(name='test', internal_name='test')
    with pytest.raises(Exception):
        step_log.get_data_catalogs_by_stage(stage='notright')


def test_steplog_get_data_catalogs_filters_by_stage(mocker):
    step_log = datastore.StepLog(name='test', internal_name='test')

    mock_data_catalog_put = mocker.MagicMock()
    mock_data_catalog_put.stage = 'put'
    mock_data_catalog_get = mocker.MagicMock()
    mock_data_catalog_get.stage = 'get'

    step_log.data_catalog = [mock_data_catalog_get, mock_data_catalog_put]

    assert step_log.get_data_catalogs_by_stage() == [mock_data_catalog_put]
    assert step_log.get_data_catalogs_by_stage(stage='get') == [mock_data_catalog_get]


def test_steplog_add_data_catalogs_inits_a_new_data_catalog():
    step_log = datastore.StepLog(name='test', internal_name='test')

    step_log.add_data_catalogs(data_catalogs=[])

    assert step_log.data_catalog == []


def test_steplog_add_data_catalogs_appends_to_existing_catalog():
    step_log = datastore.StepLog(name='test', internal_name='test')
    step_log.data_catalog = ['a']

    step_log.add_data_catalogs(data_catalogs=['b'])

    assert step_log.data_catalog == ['a', 'b']


def test_runlog_get_data_catalogs_by_state_raises_exception_for_incorrect_stage():
    run_log = datastore.RunLog(run_id='run_id')
    with pytest.raises(Exception):
        run_log.get_data_catalogs_by_stage(stage='notright')


def test_runlog_search_branch_by_internal_name_returns_run_log_if_internal_name_is_none():
    run_log = datastore.RunLog(run_id='run_id')

    branch, step = run_log.search_branch_by_internal_name(i_name=None)

    assert run_log == branch
    assert step is None


def test_step_log_get_data_catalogs_by_stage_gets_catalogs_from_branches(mocker, monkeypatch):
    mock_branch_log = mocker.MagicMock()
    mock_branch_log.get_data_catalogs_by_stage.return_value = ['from_branch']

    step_log = datastore.StepLog(name='test', internal_name='for_testing')

    step_log.branches['single_branch'] = mock_branch_log

    assert ['from_branch'] == step_log.get_data_catalogs_by_stage()


def test_step_log_get_data_catalogs_by_stage_adds_catalogs_from_branches_to_current_ones(mocker, monkeypatch):
    mock_branch_log = mocker.MagicMock()
    mock_branch_log.get_data_catalogs_by_stage.return_value = ['from_branch']

    mock_data_catalog = mocker.MagicMock()
    mock_data_catalog.stage = 'put'

    step_log = datastore.StepLog(name='test', internal_name='for_testing')

    step_log.branches['single_branch'] = mock_branch_log
    step_log.data_catalog = [mock_data_catalog]

    assert [mock_data_catalog, 'from_branch'] == step_log.get_data_catalogs_by_stage()
