import pytest

from runnable import (
    catalog,  # pylint: disable=import-error
    defaults,  # pylint: disable=import-error
)


@pytest.fixture
def instantiable_base_class(monkeypatch):
    monkeypatch.setattr(catalog.BaseCatalog, "__abstractmethods__", set())
    yield


def test_base_run_log_store_context_property(
    mocker, monkeypatch, instantiable_base_class
):
    mock_run_context = mocker.Mock()

    monkeypatch.setattr(catalog.context, "run_context", mock_run_context)

    assert catalog.BaseCatalog()._context == mock_run_context


def test_base_catalog_get_raises_exception(instantiable_base_class):
    base_catalog = catalog.BaseCatalog()
    with pytest.raises(NotImplementedError):
        base_catalog.get(name="test", run_id="test")


def test_base_catalog_put_raises_exception(instantiable_base_class):
    base_catalog = catalog.BaseCatalog()
    with pytest.raises(NotImplementedError):
        base_catalog.put(name="test", run_id="test")


def test_base_catalog_sync_between_runs_raises_exception(instantiable_base_class):
    base_catalog = catalog.BaseCatalog()
    with pytest.raises(NotImplementedError):
        base_catalog.sync_between_runs(previous_run_id=1, run_id=2)


def test_base_catalog_config_default_compute_folder_if_none_config(
    instantiable_base_class,
):
    assert catalog.BaseCatalog().compute_data_folder == defaults.COMPUTE_DATA_FOLDER


def test_do_nothing_catalog_get_returns_empty_list(monkeypatch, mocker):
    catalog_handler = catalog.DoNothingCatalog()
    assert catalog_handler.get(name="does not matter", run_id="none") == []


def test_do_nothing_catalog_put_returns_empty_list(monkeypatch, mocker):
    catalog_handler = catalog.DoNothingCatalog()
    assert catalog_handler.put(name="does not matter", run_id="none") == []


def test_do_nothing_catalog_sync_between_runs_does_nothing(monkeypatch, mocker):
    catalog_handler = catalog.DoNothingCatalog()
    catalog_handler.sync_between_runs(previous_run_id="1", run_id="2")
