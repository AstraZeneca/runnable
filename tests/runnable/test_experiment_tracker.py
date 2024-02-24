import pytest
import contextlib

from runnable import experiment_tracker


@pytest.fixture(autouse=True)
def instantiable_base_class(monkeypatch):
    monkeypatch.setattr(experiment_tracker.BaseExperimentTracker, "__abstractmethods__", set())
    yield


def test_base_run_log_store_context_property(mocker, monkeypatch, instantiable_base_class):
    mock_run_context = mocker.Mock()

    monkeypatch.setattr(experiment_tracker.context, "run_context", mock_run_context)

    assert experiment_tracker.BaseExperimentTracker()._context == mock_run_context


def test_client_connection_is_null_context():
    ep = experiment_tracker.BaseExperimentTracker()

    assert isinstance(ep.client_context, contextlib.nullcontext)


def test_do_nothing_experiment_tracker_log_metric_does_nothing():
    ep = experiment_tracker.DoNothingTracker()

    ep.log_metric(key="foo", value=3.41)


def test_do_nothing_experiment_tracker_log_parmeter_does_nothing():
    ep = experiment_tracker.DoNothingTracker()

    ep.log_parameter(key="foo", value="bar")
