import pytest
from extensions.job_executor.k8s import GenericK8sJobExecutor, Spec
from pydantic import ValidationError
from unittest.mock import Mock, patch
from runnable.tasks import BaseTaskType


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


def test_submit_k8s_cronjob_creates_cronjob_instead_of_job():
    """Test that submit_k8s_cronjob creates a CronJob with the schedule"""
    from runnable import context as runnable_context

    config = {
        "job_spec": {"template": {"spec": {"container": {"image": "test"}}}},
        "schedule": "0 2 * * *",
        "mock": True  # Use mock mode to avoid actual K8s calls
    }
    executor = GenericK8sJobExecutor(**config)

    # Mock the context and task
    mock_task = Mock(spec=BaseTaskType)
    mock_context = Mock(spec=runnable_context.JobContext)
    mock_context.run_id = "test-run-123"
    mock_context.get_job_callable_command.return_value = "python test.py"

    # This should not raise an exception and should handle CronJob creation
    with patch("runnable.context.run_context", mock_context):
        executor.submit_k8s_cronjob(mock_task)


def test_cronjob_has_correct_schedule_and_job_template():
    """Test that the CronJob contains the correct schedule and wraps the job spec"""
    from runnable import context as runnable_context

    config = {
        "job_spec": {
            "template": {
                "spec": {
                    "container": {"image": "test-image"}
                }
            }
        },
        "schedule": "0 2 * * *",
        "mock": False  # Set to False so we can test the actual K8s API call
    }
    executor = GenericK8sJobExecutor(**config)

    # Mock the context
    mock_context = Mock(spec=runnable_context.JobContext)
    mock_context.run_id = "test-run-123"
    mock_context.get_job_callable_command.return_value = "python test.py"

    # Mock the Kubernetes client at the module level
    with patch("runnable.context.run_context", mock_context):
        with patch("extensions.job_executor.k8s.client") as mock_client:
            mock_batch_api = Mock()
            mock_client.BatchV1Api.return_value = mock_batch_api

            # Create a proper mock CronJob with nested structure
            def create_cronjob(**kwargs):
                cronjob_mock = Mock()
                cronjob_mock.metadata = Mock()
                cronjob_mock.metadata.name = kwargs.get('metadata').name if 'metadata' in kwargs else None
                cronjob_mock.spec = Mock()
                if 'spec' in kwargs:
                    cronjob_mock.spec.schedule = kwargs['spec'].schedule if hasattr(kwargs['spec'], 'schedule') else None
                    cronjob_mock.spec.job_template = kwargs['spec'].job_template if hasattr(kwargs['spec'], 'job_template') else None
                return cronjob_mock

            # Mock all the V1 objects to return simple Mocks that hold the passed data
            mock_client.V1VolumeMount = Mock(side_effect=lambda **kwargs: Mock(**kwargs))
            mock_client.V1EnvVar = Mock(side_effect=lambda **kwargs: Mock(**kwargs))
            mock_client.V1Container = Mock(side_effect=lambda **kwargs: Mock(**kwargs))
            mock_client.V1Volume = Mock(side_effect=lambda **kwargs: Mock(**kwargs))
            mock_client.V1Toleration = Mock(side_effect=lambda **kwargs: Mock(**kwargs))
            mock_client.V1PodSpec = Mock(side_effect=lambda **kwargs: Mock(**kwargs))
            mock_client.V1ObjectMeta = Mock(side_effect=lambda **kwargs: type('obj', (), kwargs))
            mock_client.V1PodTemplateSpec = Mock(side_effect=lambda **kwargs: Mock(**kwargs))
            mock_client.V1JobSpec = Mock(side_effect=lambda **kwargs: Mock(**kwargs))

            # Mock CronJob-specific objects to track the schedule
            def create_cronjob_spec(**kwargs):
                spec = Mock()
                spec.schedule = kwargs.get('schedule')
                spec.job_template = kwargs.get('job_template')
                return spec

            def create_job_template_spec(**kwargs):
                template = Mock()
                template.spec = kwargs.get('spec')
                return template

            mock_client.V1CronJobSpec = Mock(side_effect=create_cronjob_spec)
            mock_client.V1JobTemplateSpec = Mock(side_effect=create_job_template_spec)
            mock_client.V1CronJob = Mock(side_effect=create_cronjob)

            mock_task = Mock(spec=BaseTaskType)
            executor.submit_k8s_cronjob(mock_task)

            # Verify CronJob creation was called
            mock_batch_api.create_namespaced_cron_job.assert_called_once()

            # Get the CronJob that was created
            call_args = mock_batch_api.create_namespaced_cron_job.call_args
            cronjob = call_args[1]['body']  # body parameter

            assert cronjob.spec.schedule == "0 2 * * *"
            assert cronjob.spec.job_template is not None


def test_submit_job_creates_cronjob_when_schedule_present():
    """Test that submit_job calls submit_k8s_cronjob when schedule is configured"""
    from runnable import context as runnable_context

    config = {
        "job_spec": {"template": {"spec": {"container": {"image": "test"}}}},
        "schedule": "0 2 * * *",
        "mock": True
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
        with patch.object(executor, '_set_up_run_log'):
            with patch.object(executor, '_create_volumes'):
                # Use patch to mock (not wrap) the methods so they don't actually execute
                with patch.object(GenericK8sJobExecutor, 'submit_k8s_cronjob') as mock_cronjob:
                    with patch.object(GenericK8sJobExecutor, 'submit_k8s_job') as mock_job:
                        mock_task = Mock(spec=BaseTaskType)
                        executor.submit_job(mock_task, catalog_settings=[])

                        # Should call cronjob method, not regular job method
                        mock_cronjob.assert_called_once_with(mock_task)
                        mock_job.assert_not_called()


def test_submit_job_creates_regular_job_when_no_schedule():
    """Test that submit_job calls submit_k8s_job when schedule is not configured"""
    from runnable import context as runnable_context

    config = {
        "job_spec": {"template": {"spec": {"container": {"image": "test"}}}},
        "mock": True
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
        with patch.object(executor, '_set_up_run_log'):
            with patch.object(executor, '_create_volumes'):
                # Use patch to mock (not wrap) the methods so they don't actually execute
                with patch.object(GenericK8sJobExecutor, 'submit_k8s_cronjob') as mock_cronjob:
                    with patch.object(GenericK8sJobExecutor, 'submit_k8s_job') as mock_job:
                        mock_task = Mock(spec=BaseTaskType)
                        executor.submit_job(mock_task, catalog_settings=[])

                        # Should call regular job method, not cronjob method
                        mock_job.assert_called_once_with(mock_task)
                        mock_cronjob.assert_not_called()
