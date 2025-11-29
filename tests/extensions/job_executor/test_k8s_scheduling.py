import pytest
from extensions.job_executor.k8s import GenericK8sJobExecutor, Spec
from pydantic import ValidationError


def test_schedule_field_accepts_valid_cron_expression():
    """Test that schedule field accepts valid cron expressions"""
    config = {
        "job_spec": {"template": {"spec": {"container": {"image": "test"}}}},
        "schedule": "0 2 * * *"
    }
    executor = GenericK8sJobExecutor(**config)
    assert executor.schedule == "0 2 * * *"


def test_schedule_field_defaults_to_none():
    """Test that schedule field defaults to None for backward compatibility"""
    config = {
        "job_spec": {"template": {"spec": {"container": {"image": "test"}}}}
    }
    executor = GenericK8sJobExecutor(**config)
    assert executor.schedule is None


def test_schedule_field_validates_cron_format():
    """Test that invalid cron expressions are rejected"""
    config = {
        "job_spec": {"template": {"spec": {"container": {"image": "test"}}}},
        "schedule": "invalid cron"
    }
    with pytest.raises(ValidationError, match="valid cron expression"):
        GenericK8sJobExecutor(**config)
