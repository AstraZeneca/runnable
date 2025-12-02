from unittest.mock import Mock, patch

import pytest
from pydantic import ValidationError

from extensions.job_executor.k8s import GenericK8sJobExecutor, Spec
from runnable.tasks import BaseTaskType


# Mock Kubernetes imports at the module level for all tests
@pytest.fixture(autouse=True)
def mock_kubernetes():
    """Auto-use fixture to mock Kubernetes imports for all tests in this module."""
    with (
        patch("extensions.job_executor.k8s.client") as mock_client,
        patch("extensions.job_executor.k8s.k8s_config") as mock_k8s_config,
    ):
        # Set up mock client with commonly used attributes
        mock_batch_v1_api = Mock()
        mock_client.BatchV1Api.return_value = mock_batch_v1_api
        mock_client.BatchV1beta1Api.return_value = Mock()
        mock_client.CoreV1Api.return_value = Mock()

        # Mock all the V1 Kubernetes objects that tests use
        mock_client.V1VolumeMount = Mock(side_effect=lambda **kwargs: Mock(**kwargs))
        mock_client.V1EnvVar = Mock(side_effect=lambda **kwargs: Mock(**kwargs))
        mock_client.V1Container = Mock(side_effect=lambda **kwargs: Mock(**kwargs))
        mock_client.V1Volume = Mock(side_effect=lambda **kwargs: Mock(**kwargs))
        mock_client.V1Toleration = Mock(side_effect=lambda **kwargs: Mock(**kwargs))
        mock_client.V1PodSpec = Mock(side_effect=lambda **kwargs: Mock(**kwargs))
        mock_client.V1ObjectMeta = Mock(
            side_effect=lambda **kwargs: type("obj", (), kwargs)
        )
        mock_client.V1PodTemplateSpec = Mock(
            side_effect=lambda **kwargs: Mock(**kwargs)
        )
        mock_client.V1JobSpec = Mock(side_effect=lambda **kwargs: Mock(**kwargs))
        mock_client.V1Job = Mock(side_effect=lambda **kwargs: Mock(**kwargs))

        # CronJob-specific mocks
        def create_cronjob_spec(**kwargs):
            spec = Mock()
            spec.schedule = kwargs.get("schedule")
            spec.job_template = kwargs.get("job_template")
            return spec

        def create_job_template_spec(**kwargs):
            template = Mock()
            template.spec = kwargs.get("spec")
            return template

        def create_cronjob(**kwargs):
            cronjob_mock = Mock()
            cronjob_mock.metadata = Mock()
            metadata = kwargs.get("metadata")
            cronjob_mock.metadata.name = (
                metadata.name if metadata and hasattr(metadata, "name") else None
            )
            cronjob_mock.spec = Mock()
            if "spec" in kwargs:
                cronjob_mock.spec.schedule = (
                    kwargs["spec"].schedule
                    if hasattr(kwargs["spec"], "schedule")
                    else None
                )
                cronjob_mock.spec.job_template = (
                    kwargs["spec"].job_template
                    if hasattr(kwargs["spec"], "job_template")
                    else None
                )
            # Set common CronJob attributes that tests expect
            cronjob_mock.kind = "CronJob"
            cronjob_mock.api_version = "batch/v1"
            return cronjob_mock

        def create_job(**kwargs):
            job_mock = Mock()
            job_mock.metadata = Mock()
            metadata = kwargs.get("metadata")
            job_mock.metadata.name = (
                metadata.name if metadata and hasattr(metadata, "name") else None
            )
            job_mock.spec = Mock()
            if "spec" in kwargs:
                job_mock.spec.active_deadline_seconds = (
                    kwargs["spec"].active_deadline_seconds
                    if hasattr(kwargs["spec"], "active_deadline_seconds")
                    else None
                )
            # Set common Job attributes that tests expect
            job_mock.kind = "Job"
            job_mock.api_version = "batch/v1"
            return job_mock

        mock_client.V1CronJobSpec = Mock(side_effect=create_cronjob_spec)
        mock_client.V1JobTemplateSpec = Mock(side_effect=create_job_template_spec)
        mock_client.V1CronJob = Mock(side_effect=create_cronjob)
        mock_client.V1Job = Mock(side_effect=create_job)

        # Mock k8s_config methods
        mock_k8s_config.load_incluster_config = Mock()
        mock_k8s_config.load_kube_config = Mock()

        yield {
            "client": mock_client,
            "k8s_config": mock_k8s_config,
            "batch_api": mock_batch_v1_api,
        }


def test_schedule_field_accepts_valid_cron_expression():
    """Test that schedule field accepts valid cron expressions"""
    config = {
        "job_spec": {"template": {"spec": {"container": {"image": "test"}}}},
        "schedule": "0 2 * * *",
    }
    executor = GenericK8sJobExecutor(**config)
    assert executor.schedule == "0 2 * * *"


def test_schedule_field_defaults_to_none():
    """Test that schedule field defaults to None for backward compatibility"""
    config = {"job_spec": {"template": {"spec": {"container": {"image": "test"}}}}}
    executor = GenericK8sJobExecutor(**config)
    assert executor.schedule is None


def test_schedule_field_validates_cron_format():
    """Test that invalid cron expressions are rejected"""
    config = {
        "job_spec": {"template": {"spec": {"container": {"image": "test"}}}},
        "schedule": "invalid cron",
    }
    with pytest.raises(ValidationError, match="valid cron expression"):
        GenericK8sJobExecutor(**config)


def test_submit_k8s_job_creates_cronjob_when_schedule_present():
    """Test that submit_k8s_job creates a CronJob when schedule is configured"""
    from runnable import context as runnable_context

    config = {
        "job_spec": {"template": {"spec": {"container": {"image": "test"}}}},
        "schedule": "0 2 * * *",
        "mock": True,  # Use mock mode to avoid actual K8s calls
    }
    executor = GenericK8sJobExecutor(**config)

    # Mock the context and task
    mock_task = Mock(spec=BaseTaskType)
    mock_context = Mock(spec=runnable_context.JobContext)
    mock_context.run_id = "test-run-123"
    mock_context.get_job_callable_command.return_value = "python test.py"

    # This should not raise an exception and should handle CronJob creation
    with patch("runnable.context.run_context", mock_context):
        executor.submit_k8s_job(mock_task)


def test_cronjob_has_correct_schedule_and_job_template(mock_kubernetes):
    """Test that the CronJob contains the correct schedule and wraps the job spec"""
    from runnable import context as runnable_context

    config = {
        "job_spec": {"template": {"spec": {"container": {"image": "test-image"}}}},
        "schedule": "0 2 * * *",
        "mock": False,  # Set to False so we can test the actual K8s API call
    }
    executor = GenericK8sJobExecutor(**config)

    # Mock the context
    mock_context = Mock(spec=runnable_context.JobContext)
    mock_context.run_id = "test-run-123"
    mock_context.get_job_callable_command.return_value = "python test.py"

    # Use the global mocks from the fixture
    mock_batch_api = mock_kubernetes["batch_api"]

    with patch("runnable.context.run_context", mock_context):
        mock_task = Mock(spec=BaseTaskType)
        executor.submit_k8s_job(mock_task)

        # Verify CronJob creation was called
        mock_batch_api.create_namespaced_cron_job.assert_called_once()

        # Get the CronJob that was created
        call_args = mock_batch_api.create_namespaced_cron_job.call_args
        cronjob = call_args[1]["body"]  # body parameter

        assert cronjob.spec.schedule == "0 2 * * *"
        assert cronjob.spec.job_template is not None


def test_submit_job_creates_cronjob_when_schedule_present():
    """Test that submit_job calls submit_k8s_job which creates a CronJob when schedule is configured"""
    from runnable import context as runnable_context

    config = {
        "job_spec": {"template": {"spec": {"container": {"image": "test"}}}},
        "schedule": "0 2 * * *",
        "mock": True,
    }
    executor = GenericK8sJobExecutor(**config)

    # Mock the context
    mock_context = Mock(spec=runnable_context.JobContext)
    mock_context.run_id = "test-run-123"
    mock_context.run_log_store = Mock()
    mock_context.run_log_store.create_job_log.return_value = Mock()
    mock_context.get_job_callable_command.return_value = "python test.py"

    # Mock the methods we'll call using patch
    with patch("runnable.context.run_context", mock_context):
        with patch.object(executor, "_set_up_run_log"):
            with patch.object(executor, "_create_volumes"):
                # Use patch to spy on submit_k8s_job method
                with patch.object(GenericK8sJobExecutor, "submit_k8s_job") as mock_job:
                    mock_task = Mock(spec=BaseTaskType)
                    executor.submit_job(mock_task, catalog_settings=[])

                    # Should call submit_k8s_job method (which internally handles CronJob)
                    mock_job.assert_called_once_with(mock_task)


def test_submit_job_creates_regular_job_when_no_schedule():
    """Test that submit_job calls submit_k8s_job which creates a regular Job when schedule is not configured"""
    from runnable import context as runnable_context

    config = {
        "job_spec": {"template": {"spec": {"container": {"image": "test"}}}},
        "mock": True,
    }
    executor = GenericK8sJobExecutor(**config)

    # Mock the context
    mock_context = Mock(spec=runnable_context.JobContext)
    mock_context.run_id = "test-run-123"
    mock_context.run_log_store = Mock()
    mock_context.run_log_store.create_job_log.return_value = Mock()
    mock_context.get_job_callable_command.return_value = "python test.py"

    # Mock the methods we'll call using patch
    with patch("runnable.context.run_context", mock_context):
        with patch.object(executor, "_set_up_run_log"):
            with patch.object(executor, "_create_volumes"):
                # Use patch to spy on submit_k8s_job method
                with patch.object(GenericK8sJobExecutor, "submit_k8s_job") as mock_job:
                    mock_task = Mock(spec=BaseTaskType)
                    executor.submit_job(mock_task, catalog_settings=[])

                    # Should call submit_k8s_job method (which internally handles regular Job)
                    mock_job.assert_called_once_with(mock_task)


# Task 4: Tests for Derived Classes (MiniK8sJobExecutor and K8sJobExecutor)


def test_mini_k8s_job_executor_inherits_schedule_support():
    """Test that MiniK8sJobExecutor supports scheduling"""
    from extensions.job_executor.k8s import MiniK8sJobExecutor

    config = {
        "job_spec": {"template": {"spec": {"container": {"image": "test"}}}},
        "schedule": "0 2 * * *",
    }
    executor = MiniK8sJobExecutor(**config)
    assert executor.schedule == "0 2 * * *"


def test_k8s_job_executor_inherits_schedule_support():
    """Test that K8sJobExecutor supports scheduling"""
    from extensions.job_executor.k8s import K8sJobExecutor

    config = {
        "job_spec": {"template": {"spec": {"container": {"image": "test"}}}},
        "pvc_claim_name": "test-pvc",
        "schedule": "0 2 * * *",
    }
    executor = K8sJobExecutor(**config)
    assert executor.schedule == "0 2 * * *"


def test_mini_k8s_job_executor_validates_schedule():
    """Test that MiniK8sJobExecutor validates cron expressions"""
    from extensions.job_executor.k8s import MiniK8sJobExecutor

    config = {
        "job_spec": {"template": {"spec": {"container": {"image": "test"}}}},
        "schedule": "invalid cron",
    }
    with pytest.raises(ValidationError, match="valid cron expression"):
        MiniK8sJobExecutor(**config)


def test_k8s_job_executor_validates_schedule():
    """Test that K8sJobExecutor validates cron expressions"""
    from extensions.job_executor.k8s import K8sJobExecutor

    config = {
        "job_spec": {"template": {"spec": {"container": {"image": "test"}}}},
        "pvc_claim_name": "test-pvc",
        "schedule": "invalid cron",
    }
    with pytest.raises(ValidationError, match="valid cron expression"):
        K8sJobExecutor(**config)


def test_mini_k8s_executor_submit_job_calls_parent_cronjob_implementation():
    """Test that MiniK8sJobExecutor uses parent's submit_job logic for scheduling"""
    from extensions.job_executor.k8s import GenericK8sJobExecutor, MiniK8sJobExecutor
    from runnable import context as runnable_context

    config = {
        "job_spec": {"template": {"spec": {"container": {"image": "test"}}}},
        "schedule": "0 2 * * *",
        "mock": True,
    }
    executor = MiniK8sJobExecutor(**config)

    # Mock the context
    mock_context = Mock(spec=runnable_context.JobContext)
    mock_context.run_id = "test-run-123"
    mock_context.run_log_store = Mock()
    mock_context.run_log_store.create_job_log.return_value = Mock()
    mock_context.get_job_callable_command.return_value = "python test.py"

    # Mock the methods at the parent class level
    with patch("runnable.context.run_context", mock_context):
        with patch.object(GenericK8sJobExecutor, "_set_up_run_log"):
            with patch.object(MiniK8sJobExecutor, "_create_volumes"):
                with patch.object(GenericK8sJobExecutor, "submit_k8s_job") as mock_job:
                    mock_task = Mock(spec=BaseTaskType)
                    executor.submit_job(mock_task, catalog_settings=[])

                    # Should call submit_k8s_job method since schedule is set
                    mock_job.assert_called_once_with(mock_task)


def test_k8s_executor_submit_job_calls_parent_cronjob_implementation():
    """Test that K8sJobExecutor uses parent's submit_job logic for scheduling"""
    from extensions.job_executor.k8s import GenericK8sJobExecutor, K8sJobExecutor
    from runnable import context as runnable_context

    config = {
        "job_spec": {"template": {"spec": {"container": {"image": "test"}}}},
        "pvc_claim_name": "test-pvc",
        "schedule": "0 2 * * *",
        "mock": True,
    }
    executor = K8sJobExecutor(**config)

    # Mock the context
    mock_context = Mock(spec=runnable_context.JobContext)
    mock_context.run_id = "test-run-123"
    mock_context.run_log_store = Mock()
    mock_context.run_log_store.create_job_log.return_value = Mock()
    mock_context.get_job_callable_command.return_value = "python test.py"

    # Mock the methods at the parent class level
    with patch("runnable.context.run_context", mock_context):
        with patch.object(GenericK8sJobExecutor, "_set_up_run_log"):
            with patch.object(K8sJobExecutor, "_create_volumes"):
                with patch.object(GenericK8sJobExecutor, "submit_k8s_job") as mock_job:
                    mock_task = Mock(spec=BaseTaskType)
                    executor.submit_job(mock_task, catalog_settings=[])

                    # Should call submit_k8s_job method since schedule is set
                    mock_job.assert_called_once_with(mock_task)


def test_mini_k8s_executor_submit_job_calls_parent_regular_job_implementation():
    """Test that MiniK8sJobExecutor uses parent's submit_job logic for regular jobs"""
    from extensions.job_executor.k8s import GenericK8sJobExecutor, MiniK8sJobExecutor
    from runnable import context as runnable_context

    config = {
        "job_spec": {"template": {"spec": {"container": {"image": "test"}}}},
        "mock": True,
        # No schedule - should create regular job
    }
    executor = MiniK8sJobExecutor(**config)

    # Mock the context
    mock_context = Mock(spec=runnable_context.JobContext)
    mock_context.run_id = "test-run-123"
    mock_context.run_log_store = Mock()
    mock_context.run_log_store.create_job_log.return_value = Mock()
    mock_context.get_job_callable_command.return_value = "python test.py"

    # Mock the methods at the parent class level
    with patch("runnable.context.run_context", mock_context):
        with patch.object(GenericK8sJobExecutor, "_set_up_run_log"):
            with patch.object(MiniK8sJobExecutor, "_create_volumes"):
                with patch.object(GenericK8sJobExecutor, "submit_k8s_job") as mock_job:
                    mock_task = Mock(spec=BaseTaskType)
                    executor.submit_job(mock_task, catalog_settings=[])

                    # Should call submit_k8s_job method
                    mock_job.assert_called_once_with(mock_task)


def test_k8s_executor_submit_job_calls_parent_regular_job_implementation():
    """Test that K8sJobExecutor uses parent's submit_job logic for regular jobs"""
    from extensions.job_executor.k8s import GenericK8sJobExecutor, K8sJobExecutor
    from runnable import context as runnable_context

    config = {
        "job_spec": {"template": {"spec": {"container": {"image": "test"}}}},
        "pvc_claim_name": "test-pvc",
        "mock": True,
        # No schedule - should create regular job
    }
    executor = K8sJobExecutor(**config)

    # Mock the context
    mock_context = Mock(spec=runnable_context.JobContext)
    mock_context.run_id = "test-run-123"
    mock_context.run_log_store = Mock()
    mock_context.run_log_store.create_job_log.return_value = Mock()
    mock_context.get_job_callable_command.return_value = "python test.py"

    # Mock the methods at the parent class level
    with patch("runnable.context.run_context", mock_context):
        with patch.object(GenericK8sJobExecutor, "_set_up_run_log"):
            with patch.object(K8sJobExecutor, "_create_volumes"):
                with patch.object(GenericK8sJobExecutor, "submit_k8s_job") as mock_job:
                    mock_task = Mock(spec=BaseTaskType)
                    executor.submit_job(mock_task, catalog_settings=[])

                    # Should call submit_k8s_job method
                    mock_job.assert_called_once_with(mock_task)


def test_derived_classes_do_not_override_submit_job():
    """Test that derived classes inherit submit_job and don't override it"""
    from extensions.job_executor.k8s import (
        GenericK8sJobExecutor,
        K8sJobExecutor,
        MiniK8sJobExecutor,
    )

    # Verify that MiniK8sJobExecutor doesn't override submit_job
    assert MiniK8sJobExecutor.submit_job is GenericK8sJobExecutor.submit_job

    # Verify that K8sJobExecutor doesn't override submit_job
    assert K8sJobExecutor.submit_job is GenericK8sJobExecutor.submit_job


def test_mini_k8s_executor_schedule_defaults_to_none():
    """Test that MiniK8sJobExecutor schedule defaults to None for backward compatibility"""
    from extensions.job_executor.k8s import MiniK8sJobExecutor

    config = {"job_spec": {"template": {"spec": {"container": {"image": "test"}}}}}
    executor = MiniK8sJobExecutor(**config)
    assert executor.schedule is None


def test_k8s_executor_schedule_defaults_to_none():
    """Test that K8sJobExecutor schedule defaults to None for backward compatibility"""
    from extensions.job_executor.k8s import K8sJobExecutor

    config = {
        "job_spec": {"template": {"spec": {"container": {"image": "test"}}}},
        "pvc_claim_name": "test-pvc",
    }
    executor = K8sJobExecutor(**config)
    assert executor.schedule is None


def test_mini_k8s_executor_supports_various_cron_schedules():
    """Test that MiniK8sJobExecutor supports various valid cron expressions"""
    from extensions.job_executor.k8s import MiniK8sJobExecutor

    valid_schedules = [
        "0 2 * * *",  # Daily at 2 AM
        "0 * * * *",  # Every hour
        "0 9 * * 1",  # Monday at 9 AM
        "*/15 * * * *",  # Every 15 minutes
        "30 3 * * 0",  # Sunday at 3:30 AM
        "0 0 1 * *",  # First day of month at midnight
    ]

    for schedule in valid_schedules:
        config = {
            "job_spec": {"template": {"spec": {"container": {"image": "test"}}}},
            "schedule": schedule,
        }
        executor = MiniK8sJobExecutor(**config)
        assert executor.schedule == schedule


def test_k8s_executor_supports_various_cron_schedules():
    """Test that K8sJobExecutor supports various valid cron expressions"""
    from extensions.job_executor.k8s import K8sJobExecutor

    valid_schedules = [
        "0 2 * * *",  # Daily at 2 AM
        "0 * * * *",  # Every hour
        "0 9 * * 1",  # Monday at 9 AM
        "*/15 * * * *",  # Every 15 minutes
        "30 3 * * 0",  # Sunday at 3:30 AM
        "0 0 1 * *",  # First day of month at midnight
    ]

    for schedule in valid_schedules:
        config = {
            "job_spec": {"template": {"spec": {"container": {"image": "test"}}}},
            "pvc_claim_name": "test-pvc",
            "schedule": schedule,
        }
        executor = K8sJobExecutor(**config)
        assert executor.schedule == schedule


# Task 5: Test for Example Configuration File


def test_example_scheduled_config_is_valid():
    """Test that the example scheduled job config file is valid"""
    from pathlib import Path

    import yaml

    from extensions.job_executor.k8s import K8sJobExecutor

    config_path = Path("examples/11-jobs/k8s-scheduled-job.yaml")
    assert config_path.exists(), "Example config file should exist"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    job_executor_config = config["job-executor"]["config"]

    # Should be able to create executor with this config
    executor = K8sJobExecutor(**job_executor_config)
    assert executor.schedule == "* * * * *"
    assert executor.pvc_claim_name == "runnable"
    assert executor.namespace == "enterprise-mlops"


def test_documentation_examples_are_valid():
    """Test that cron expressions in documentation are valid"""
    examples = [
        "0 2 * * *",  # Daily at 2 AM
        "0 * * * *",  # Every hour
        "0 9 * * 1",  # Monday at 9 AM
        "*/15 * * * *",  # Every 15 minutes
        "30 3 * * 0",  # Sunday at 3:30 AM
        "0 0 1 * *",  # First day of month at midnight
    ]

    for schedule in examples:
        config = {
            "job_spec": {"template": {"spec": {"container": {"image": "test"}}}},
            "schedule": schedule,
        }
        executor = GenericK8sJobExecutor(**config)
        assert executor.schedule == schedule


def test_documentation_schedule_reference_examples():
    """Test all schedule reference examples from documentation"""
    reference_examples = [
        "* * * * *",  # Every minute
        "*/5 * * * *",  # Every 5 minutes
        "30 * * * *",  # Every hour at 30 minutes past
        "0 */6 * * *",  # Every 6 hours
        "0 12 * * *",  # Every day at noon
        "0 9 * * 1-5",  # Every weekday at 9 AM
        "0 0 * * 6",  # Every Saturday at midnight
        "0 8 1-7 * 1",  # First Monday of every month at 8 AM
    ]

    for schedule in reference_examples:
        config = {
            "job_spec": {"template": {"spec": {"container": {"image": "test"}}}},
            "schedule": schedule,
        }
        executor = GenericK8sJobExecutor(**config)
        assert executor.schedule == schedule


def test_invalid_schedule_formats_from_documentation():
    """Test that invalid schedule formats mentioned in docs are rejected"""
    invalid_schedules = [
        "0 2 * *",  # Too few fields (4 instead of 5)
        "0 0 2 * * *",  # Too many fields (6 instead of 5)
        "",  # Empty string
        "invalid",  # Single word
    ]

    for schedule in invalid_schedules:
        config = {
            "job_spec": {"template": {"spec": {"container": {"image": "test"}}}},
            "schedule": schedule,
        }
        with pytest.raises(ValidationError, match="valid cron expression"):
            GenericK8sJobExecutor(**config)


# Task 7: Integration Testing and Validation


def test_end_to_end_scheduled_job_creation(mock_kubernetes):
    """Test complete flow from configuration to CronJob creation"""
    from runnable import context as runnable_context

    config = {
        "job_spec": {
            "activeDeadlineSeconds": 3600,
            "template": {
                "spec": {
                    "activeDeadlineSeconds": 3600,
                    "restartPolicy": "Never",
                    "container": {
                        "image": "python:3.9",
                        "env": [{"name": "TEST", "value": "value"}],
                        "resources": {
                            "limits": {"cpu": "1", "memory": "1Gi"},
                            "requests": {"cpu": "500m", "memory": "512Mi"},
                        },
                    },
                }
            },
        },
        "schedule": "0 3 * * *",
        "namespace": "test-namespace",
        "mock": False,  # Set to False to test actual CronJob creation logic
    }

    executor = GenericK8sJobExecutor(**config)

    # Mock the context
    mock_context = Mock(spec=runnable_context.JobContext)
    mock_context.run_id = "integration-test-run"
    mock_context.get_job_callable_command.return_value = "python main.py"

    # Use the global mocks from the fixture
    mock_batch_api = mock_kubernetes["batch_api"]

    with patch("runnable.context.run_context", mock_context):
        mock_task = Mock(spec=BaseTaskType)
        executor.submit_k8s_job(mock_task)

        # Verify the call was made
        mock_batch_api.create_namespaced_cron_job.assert_called_once()

        # Verify the CronJob structure
        call_args = mock_batch_api.create_namespaced_cron_job.call_args
        cronjob = call_args[1]["body"]

        assert cronjob.kind == "CronJob"
        assert cronjob.api_version == "batch/v1"
        assert cronjob.metadata.name == "integration-test-run"
        assert cronjob.spec.schedule == "0 3 * * *"

        # Verify job template contains our job spec
        job_template = cronjob.spec.job_template
        assert job_template.spec.active_deadline_seconds == 3600


def test_error_handling_in_cronjob_creation(mock_kubernetes):
    """Test that CronJob creation errors are handled properly"""
    from runnable import context as runnable_context

    config = {
        "job_spec": {"template": {"spec": {"container": {"image": "test"}}}},
        "schedule": "0 2 * * *",
        "mock": False,  # Use real mode to test error handling
    }

    executor = GenericK8sJobExecutor(**config)

    # Mock the context
    mock_context = Mock(spec=runnable_context.JobContext)
    mock_context.run_id = "error-test-run"
    mock_context.get_job_callable_command.return_value = "python test.py"

    # Use the global mocks from the fixture
    mock_batch_api = mock_kubernetes["batch_api"]

    # Simulate K8s API error
    from kubernetes.client.exceptions import ApiException

    mock_batch_api.create_namespaced_cron_job.side_effect = ApiException(
        status=403, reason="Forbidden"
    )

    with patch("runnable.context.run_context", mock_context):
        mock_task = Mock(spec=BaseTaskType)

        # Should re-raise the exception
        with pytest.raises(ApiException):
            executor.submit_k8s_job(mock_task)


def test_end_to_end_regular_job_creation_without_schedule(mock_kubernetes):
    """Test complete flow for regular Job creation when schedule is not present"""
    from runnable import context as runnable_context

    config = {
        "job_spec": {
            "activeDeadlineSeconds": 3600,
            "template": {
                "spec": {
                    "activeDeadlineSeconds": 3600,
                    "restartPolicy": "Never",
                    "container": {
                        "image": "python:3.9",
                        "env": [{"name": "TEST", "value": "value"}],
                        "resources": {
                            "limits": {"cpu": "1", "memory": "1Gi"},
                            "requests": {"cpu": "500m", "memory": "512Mi"},
                        },
                    },
                }
            },
        },
        "namespace": "test-namespace",
        "mock": False,  # No schedule - should create regular Job
    }

    executor = GenericK8sJobExecutor(**config)

    # Mock the context
    mock_context = Mock(spec=runnable_context.JobContext)
    mock_context.run_id = "integration-test-regular-job"
    mock_context.get_job_callable_command.return_value = "python main.py"

    # Use the global mocks from the fixture
    mock_batch_api = mock_kubernetes["batch_api"]

    with patch("runnable.context.run_context", mock_context):
        mock_task = Mock(spec=BaseTaskType)
        executor.submit_k8s_job(mock_task)

        # Verify the regular Job creation was called, NOT CronJob
        mock_batch_api.create_namespaced_job.assert_called_once()
        mock_batch_api.create_namespaced_cron_job.assert_not_called()

        # Verify the Job structure
        call_args = mock_batch_api.create_namespaced_job.call_args
        job = call_args[1]["body"]

        assert job.kind == "Job"
        assert job.api_version == "batch/v1"
        assert job.metadata.name == "integration-test-regular-job"


def test_cronjob_with_volumes_and_tolerations(mock_kubernetes):
    """Test CronJob creation with volumes and tolerations"""
    from runnable import context as runnable_context

    config = {
        "job_spec": {
            "template": {
                "spec": {
                    "restartPolicy": "Never",
                    "container": {
                        "image": "python:3.9",
                        "volumeMounts": [
                            {"name": "data", "mountPath": "/data"},
                            {"name": "config", "mountPath": "/config"},
                        ],
                    },
                    "volumes": [
                        {"name": "data", "hostPath": {"path": "/tmp/data"}},
                        {
                            "name": "config",
                            "persistentVolumeClaim": {"claimName": "my-pvc"},
                        },
                    ],
                    "tolerations": [
                        {
                            "key": "node-role",
                            "operator": "Equal",
                            "value": "compute",
                            "effect": "NoSchedule",
                        }
                    ],
                }
            }
        },
        "schedule": "0 4 * * *",
        "mock": False,
    }

    executor = GenericK8sJobExecutor(**config)

    # Mock the context
    mock_context = Mock(spec=runnable_context.JobContext)
    mock_context.run_id = "test-volumes-tolerations"
    mock_context.get_job_callable_command.return_value = "python main.py"

    # Use the global mocks from the fixture
    mock_batch_api = mock_kubernetes["batch_api"]
    mock_client = mock_kubernetes["client"]

    with patch("runnable.context.run_context", mock_context):
        mock_task = Mock(spec=BaseTaskType)
        executor.submit_k8s_job(mock_task)

        # Verify CronJob was created
        mock_batch_api.create_namespaced_cron_job.assert_called_once()

        # Verify V1Volume and V1Toleration were called for volumes and tolerations
        assert mock_client.V1Volume.call_count >= 2  # data and config volumes
        assert mock_client.V1Toleration.call_count >= 1  # tolerations


def test_full_submit_job_flow_with_schedule():
    """Test the complete submit_job flow with schedule configured"""
    from runnable import context as runnable_context

    config = {
        "job_spec": {"template": {"spec": {"container": {"image": "test"}}}},
        "schedule": "0 2 * * *",
        "mock": True,
    }

    executor = GenericK8sJobExecutor(**config)

    # Mock the context
    mock_context = Mock(spec=runnable_context.JobContext)
    mock_context.run_id = "full-flow-test"
    mock_context.run_log_store = Mock()
    mock_context.run_log_store.create_job_log.return_value = Mock()
    mock_context.get_job_callable_command.return_value = "python test.py"

    # Test the full flow through submit_job
    with patch("runnable.context.run_context", mock_context):
        with patch.object(executor, "_set_up_run_log"):
            with patch.object(executor, "_create_volumes"):
                # Patch at class level instead of instance level
                with patch.object(
                    GenericK8sJobExecutor, "submit_k8s_job"
                ) as mock_submit:
                    mock_task = Mock(spec=BaseTaskType)
                    executor.submit_job(mock_task, catalog_settings=[])

                    # Verify submit_k8s_job was called
                    mock_submit.assert_called_once_with(mock_task)


def test_cronjob_namespace_configuration(mock_kubernetes):
    """Test that CronJob is created in the correct namespace"""
    from runnable import context as runnable_context

    config = {
        "job_spec": {"template": {"spec": {"container": {"image": "test"}}}},
        "schedule": "0 2 * * *",
        "namespace": "custom-namespace",
        "mock": False,
    }

    executor = GenericK8sJobExecutor(**config)

    # Mock the context
    mock_context = Mock(spec=runnable_context.JobContext)
    mock_context.run_id = "namespace-test"
    mock_context.get_job_callable_command.return_value = "python test.py"

    # Use the global mocks from the fixture
    mock_batch_api = mock_kubernetes["batch_api"]

    with patch("runnable.context.run_context", mock_context):
        mock_task = Mock(spec=BaseTaskType)
        executor.submit_k8s_job(mock_task)

        # Verify the call was made with correct namespace
        call_args = mock_batch_api.create_namespaced_cron_job.call_args
        assert call_args[1]["namespace"] == "custom-namespace"


def test_cronjob_mock_mode_no_api_call(mock_kubernetes):
    """Test that mock mode does not make actual K8s API calls for CronJob"""
    from runnable import context as runnable_context

    config = {
        "job_spec": {"template": {"spec": {"container": {"image": "test"}}}},
        "schedule": "0 2 * * *",
        "mock": True,  # Mock mode - should not call API
    }

    executor = GenericK8sJobExecutor(**config)

    # Mock the context
    mock_context = Mock(spec=runnable_context.JobContext)
    mock_context.run_id = "mock-mode-test"
    mock_context.get_job_callable_command.return_value = "python test.py"

    # Use the global mocks from the fixture
    mock_batch_api = mock_kubernetes["batch_api"]

    with patch("runnable.context.run_context", mock_context):
        mock_task = Mock(spec=BaseTaskType)
        executor.submit_k8s_job(mock_task)

        # Verify NO API call was made in mock mode
        mock_batch_api.create_namespaced_cron_job.assert_not_called()


def test_backward_compatibility_no_schedule_uses_regular_job(mock_kubernetes):
    """Test that when schedule is not provided, regular Job is used (backward compatibility)"""
    from runnable import context as runnable_context

    config = {
        "job_spec": {"template": {"spec": {"container": {"image": "test"}}}},
        # No schedule field - should use regular Job
        "mock": False,
    }

    executor = GenericK8sJobExecutor(**config)
    assert executor.schedule is None

    # Mock the context
    mock_context = Mock(spec=runnable_context.JobContext)
    mock_context.run_id = "backward-compat-test"
    mock_context.get_job_callable_command.return_value = "python test.py"

    # Use the global mocks from the fixture
    mock_batch_api = mock_kubernetes["batch_api"]

    with patch("runnable.context.run_context", mock_context):
        mock_task = Mock(spec=BaseTaskType)
        executor.submit_k8s_job(mock_task)

        # Verify regular Job was created, NOT CronJob
        mock_batch_api.create_namespaced_job.assert_called_once()
        mock_batch_api.create_namespaced_cron_job.assert_not_called()


def test_cronjob_with_complex_resources(mock_kubernetes):
    """Test CronJob creation with complex resource specifications"""
    from runnable import context as runnable_context

    config = {
        "job_spec": {
            "activeDeadlineSeconds": 7200,
            "backoffLimit": 3,
            "template": {
                "metadata": {
                    "generateName": "test-job-",
                    "annotations": {"description": "Test job", "app": "test-app"},
                },
                "spec": {
                    "activeDeadlineSeconds": 7200,
                    "restartPolicy": "OnFailure",
                    "container": {
                        "image": "python:3.11-slim",
                        "imagePullPolicy": "Always",
                        "env": [
                            {"name": "ENV_VAR_1", "value": "value1"},
                            {"name": "ENV_VAR_2", "value": "value2"},
                        ],
                        "resources": {
                            "limits": {"cpu": "2", "memory": "4Gi"},
                            "requests": {"cpu": "1", "memory": "2Gi"},
                        },
                    },
                },
            },
        },
        "schedule": "0 6 * * 1-5",  # Weekdays at 6 AM
        "namespace": "production",
        "mock": False,
    }

    executor = GenericK8sJobExecutor(**config)

    # Mock the context
    mock_context = Mock(spec=runnable_context.JobContext)
    mock_context.run_id = "complex-resources-test"
    mock_context.get_job_callable_command.return_value = "python app.py"

    # Use the global mocks from the fixture
    mock_batch_api = mock_kubernetes["batch_api"]

    with patch("runnable.context.run_context", mock_context):
        mock_task = Mock(spec=BaseTaskType)
        executor.submit_k8s_job(mock_task)

        # Verify CronJob creation was called
        mock_batch_api.create_namespaced_cron_job.assert_called_once()

        # Verify the CronJob has correct schedule
        call_args = mock_batch_api.create_namespaced_cron_job.call_args
        cronjob = call_args[1]["body"]
        assert cronjob.spec.schedule == "0 6 * * 1-5"

        # Verify job spec properties are preserved
        assert cronjob.spec.job_template.spec.active_deadline_seconds == 7200
