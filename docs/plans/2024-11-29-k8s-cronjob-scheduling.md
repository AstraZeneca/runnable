# K8s CronJob Scheduling Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add optional cron scheduling to K8sJobExecutor that creates Kubernetes CronJobs instead of Jobs when schedule is configured.

**Architecture:** Extend existing K8sJobExecutor configuration with optional `schedule` field. When present, create CronJob using existing job spec logic wrapped in CronJob template. Display scheduled job details to console without immediate execution.

**Tech Stack:** Python, Pydantic, Kubernetes Python Client, pytest

---

### Task 1: Add Schedule Configuration Field

**Files:**
- Modify: `extensions/job_executor/k8s.py:170-176` (GenericK8sJobExecutor class)
- Test: `tests/extensions/job_executor/test_k8s_scheduling.py` (new file)

**Step 1: Write the failing test for schedule field validation**

```python
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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/extensions/job_executor/test_k8s_scheduling.py::test_schedule_field_accepts_valid_cron_expression -v`
Expected: FAIL with "test_k8s_scheduling.py not found" or "schedule field not found"

**Step 3: Create test file**

Create file: `tests/extensions/job_executor/test_k8s_scheduling.py` with the test code above.

**Step 4: Run test again to verify field doesn't exist**

Run: `pytest tests/extensions/job_executor/test_k8s_scheduling.py -v`
Expected: FAIL with "schedule field not found" or similar

**Step 5: Add schedule field to GenericK8sJobExecutor**

In `extensions/job_executor/k8s.py`, modify the GenericK8sJobExecutor class:

```python
import re
from pydantic import BaseModel, ConfigDict, Field, PlainSerializer, PrivateAttr, field_validator

class GenericK8sJobExecutor(GenericJobExecutor):
    service_name: str = "k8s-job"
    config_path: Optional[str] = None
    job_spec: Spec
    mock: bool = False
    namespace: str = Field(default="default")
    schedule: Optional[str] = Field(default=None, description="Cron expression for scheduling (e.g., '0 2 * * *')")

    @field_validator('schedule')
    @classmethod
    def validate_schedule(cls, v):
        if v is not None:
            # Validate cron expression format (5 fields: minute hour day month weekday)
            if not re.match(r'^(\S+\s+){4}\S+$', v):
                raise ValueError("Schedule must be a valid cron expression with 5 fields (minute hour day month weekday)")
        return v
```

**Step 6: Run test to verify it passes**

Run: `pytest tests/extensions/job_executor/test_k8s_scheduling.py -v`
Expected: PASS (all 3 tests)

**Step 7: Commit**

```bash
git add extensions/job_executor/k8s.py tests/extensions/job_executor/test_k8s_scheduling.py
git commit -m "feat: add optional schedule field to K8sJobExecutor"
```

---

### Task 2: Implement CronJob Creation Method

**Files:**
- Modify: `extensions/job_executor/k8s.py:254-349` (add submit_k8s_cronjob method)
- Test: `tests/extensions/job_executor/test_k8s_scheduling.py` (add cronjob creation tests)

**Step 1: Write the failing test for CronJob creation**

Add to `tests/extensions/job_executor/test_k8s_scheduling.py`:

```python
from unittest.mock import Mock, patch
from runnable.tasks import BaseTaskType

def test_submit_k8s_cronjob_creates_cronjob_instead_of_job():
    """Test that submit_k8s_cronjob creates a CronJob with the schedule"""
    config = {
        "job_spec": {"template": {"spec": {"container": {"image": "test"}}}},
        "schedule": "0 2 * * *",
        "mock": True  # Use mock mode to avoid actual K8s calls
    }
    executor = GenericK8sJobExecutor(**config)

    # Mock the context and task
    mock_task = Mock(spec=BaseTaskType)
    executor._context = Mock()
    executor._context.run_id = "test-run-123"
    executor._context.get_job_callable_command.return_value = "python test.py"

    # This should not raise an exception and should handle CronJob creation
    executor.submit_k8s_cronjob(mock_task)

def test_cronjob_has_correct_schedule_and_job_template():
    """Test that the CronJob contains the correct schedule and wraps the job spec"""
    config = {
        "job_spec": {
            "template": {
                "spec": {
                    "container": {"image": "test-image"}
                }
            }
        },
        "schedule": "0 2 * * *",
        "mock": True
    }
    executor = GenericK8sJobExecutor(**config)
    executor._context = Mock()
    executor._context.run_id = "test-run-123"
    executor._context.get_job_callable_command.return_value = "python test.py"

    # Mock the Kubernetes client
    with patch.object(executor, '_client') as mock_client:
        mock_batch_api = Mock()
        mock_client.BatchV1Api.return_value = mock_batch_api

        mock_task = Mock(spec=BaseTaskType)
        executor.submit_k8s_cronjob(mock_task)

        # Verify CronJob creation was called
        mock_batch_api.create_namespaced_cron_job.assert_called_once()

        # Get the CronJob that was created
        call_args = mock_batch_api.create_namespaced_cron_job.call_args
        cronjob = call_args[1]['body']  # body parameter

        assert cronjob.spec.schedule == "0 2 * * *"
        assert cronjob.spec.job_template is not None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/extensions/job_executor/test_k8s_scheduling.py::test_submit_k8s_cronjob_creates_cronjob_instead_of_job -v`
Expected: FAIL with "submit_k8s_cronjob method not found"

**Step 3: Implement submit_k8s_cronjob method**

Add to `extensions/job_executor/k8s.py` in the GenericK8sJobExecutor class:

```python
def submit_k8s_cronjob(self, task: BaseTaskType):
    """
    Submit a Kubernetes CronJob instead of a regular Job.
    Reuses the existing job spec building logic but wraps it in a CronJob.
    """
    # Reuse existing volume mount logic
    if self.job_spec.template.spec.container.volume_mounts:
        self._volume_mounts += self.job_spec.template.spec.container.volume_mounts

    container_volume_mounts = [
        self._client.V1VolumeMount(**vol.model_dump())
        for vol in self._volume_mounts
    ]

    assert isinstance(self._context, context.JobContext)
    command = self._context.get_job_callable_command()

    container_env = [
        self._client.V1EnvVar(**env.model_dump())
        for env in self.job_spec.template.spec.container.env
    ]

    # Build container (reuse existing logic)
    base_container = self._client.V1Container(
        command=shlex.split(command),
        env=container_env,
        name="default",
        volume_mounts=container_volume_mounts,
        resources=self.job_spec.template.spec.container.resources.model_dump(
            by_alias=True, exclude_none=True
        ),
        **self.job_spec.template.spec.container.model_dump(
            exclude_none=True,
            exclude={"volume_mounts", "command", "env", "resources"},
        ),
    )

    # Build volumes (reuse existing logic)
    if self.job_spec.template.spec.volumes:
        self._volumes += self.job_spec.template.spec.volumes

    spec_volumes = [
        self._client.V1Volume(**vol.model_dump()) for vol in self._volumes
    ]

    tolerations = None
    if self.job_spec.template.spec.tolerations:
        tolerations = [
            self._client.V1Toleration(**toleration.model_dump())
            for toleration in self.job_spec.template.spec.tolerations
        ]

    # Build pod spec (reuse existing logic)
    pod_spec = self._client.V1PodSpec(
        containers=[base_container],
        volumes=spec_volumes,
        tolerations=tolerations,
        **self.job_spec.template.spec.model_dump(
            exclude_none=True, exclude={"container", "volumes", "tolerations"}
        ),
    )

    pod_template_metadata = None
    if self.job_spec.template.metadata:
        pod_template_metadata = self._client.V1ObjectMeta(
            **self.job_spec.template.metadata.model_dump(exclude_none=True)
        )

    pod_template = self._client.V1PodTemplateSpec(
        spec=pod_spec,
        metadata=pod_template_metadata,
    )

    # Build job spec (reuse existing logic)
    job_spec = client.V1JobSpec(
        template=pod_template,
        **self.job_spec.model_dump(exclude_none=True, exclude={"template"}),
    )

    # Build CronJob spec (new part)
    cronjob_spec = client.V1CronJobSpec(
        schedule=self.schedule,
        job_template=client.V1JobTemplateSpec(
            spec=job_spec
        )
    )

    # Build CronJob (new part)
    cronjob = client.V1CronJob(
        api_version="batch/v1",
        kind="CronJob",
        metadata=client.V1ObjectMeta(name=self._context.run_id),
        spec=cronjob_spec,
    )

    logger.info(f"Submitting CronJob: {cronjob.__dict__}")
    self._display_scheduled_job_info(cronjob)

    if self.mock:
        logger.info(cronjob.__dict__)
        return

    try:
        k8s_batch = self._client.BatchV1Api()
        response = k8s_batch.create_namespaced_cron_job(
            body=cronjob,
            namespace=self.namespace,
        )
        logger.debug(f"Kubernetes CronJob response: {response}")
    except Exception as e:
        logger.exception(e)
        print(e)
        raise

def _display_scheduled_job_info(self, cronjob):
    """Display information about the scheduled CronJob to the console"""
    from runnable import console

    console.print("✓ CronJob scheduled successfully")
    console.print(f"  Name: {cronjob.metadata.name}")
    console.print(f"  Namespace: {self.namespace}")
    console.print(f"  Schedule: {cronjob.spec.schedule}")
    console.print("")
    console.print("  Job Spec:")
    console.print(f"  - Image: {self.job_spec.template.spec.container.image}")
    console.print(f"  - Resources: {self.job_spec.template.spec.container.resources.model_dump()}")
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/extensions/job_executor/test_k8s_scheduling.py::test_submit_k8s_cronjob_creates_cronjob_instead_of_job -v`
Expected: PASS

**Step 5: Run all scheduling tests**

Run: `pytest tests/extensions/job_executor/test_k8s_scheduling.py -v`
Expected: PASS (all tests)

**Step 6: Commit**

```bash
git add extensions/job_executor/k8s.py tests/extensions/job_executor/test_k8s_scheduling.py
git commit -m "feat: implement CronJob creation method with schedule support"
```

---

### Task 3: Update submit_job Method to Branch on Schedule

**Files:**
- Modify: `extensions/job_executor/k8s.py:191-204` (submit_job method)
- Test: `tests/extensions/job_executor/test_k8s_scheduling.py` (add integration tests)

**Step 1: Write the failing integration test**

Add to `tests/extensions/job_executor/test_k8s_scheduling.py`:

```python
def test_submit_job_creates_cronjob_when_schedule_present():
    """Test that submit_job calls submit_k8s_cronjob when schedule is configured"""
    config = {
        "job_spec": {"template": {"spec": {"container": {"image": "test"}}}},
        "schedule": "0 2 * * *",
        "mock": True
    }
    executor = GenericK8sJobExecutor(**config)
    executor._context = Mock()
    executor._context.run_id = "test-run-123"
    executor._context.run_log_store = Mock()
    executor._context.run_log_store.create_job_log.return_value = Mock()

    # Mock the methods we'll call
    executor._set_up_run_log = Mock()
    executor._create_volumes = Mock()
    executor.submit_k8s_cronjob = Mock()
    executor.submit_k8s_job = Mock()

    mock_task = Mock(spec=BaseTaskType)
    executor.submit_job(mock_task, catalog_settings=[])

    # Should call cronjob method, not regular job method
    executor.submit_k8s_cronjob.assert_called_once_with(mock_task)
    executor.submit_k8s_job.assert_not_called()

def test_submit_job_creates_regular_job_when_no_schedule():
    """Test that submit_job calls submit_k8s_job when schedule is not configured"""
    config = {
        "job_spec": {"template": {"spec": {"container": {"image": "test"}}}},
        "mock": True
    }
    executor = GenericK8sJobExecutor(**config)
    executor._context = Mock()
    executor._context.run_id = "test-run-123"
    executor._context.run_log_store = Mock()
    executor._context.run_log_store.create_job_log.return_value = Mock()

    # Mock the methods we'll call
    executor._set_up_run_log = Mock()
    executor._create_volumes = Mock()
    executor.submit_k8s_cronjob = Mock()
    executor.submit_k8s_job = Mock()

    mock_task = Mock(spec=BaseTaskType)
    executor.submit_job(mock_task, catalog_settings=[])

    # Should call regular job method, not cronjob method
    executor.submit_k8s_job.assert_called_once_with(mock_task)
    executor.submit_k8s_cronjob.assert_not_called()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/extensions/job_executor/test_k8s_scheduling.py::test_submit_job_creates_cronjob_when_schedule_present -v`
Expected: FAIL because submit_job doesn't branch on schedule yet

**Step 3: Update submit_job method to branch on schedule**

In `extensions/job_executor/k8s.py`, modify the submit_job method:

```python
def submit_job(self, job: BaseTaskType, catalog_settings=Optional[List[str]]):
    """
    This method gets invoked by the CLI.
    """
    self._set_up_run_log()

    # Call the container job
    job_log = self._context.run_log_store.create_job_log()
    self._context.run_log_store.add_job_log(
        run_id=self._context.run_id, job_log=job_log
    )
    # create volumes and volume mounts for the job
    self._create_volumes()

    # Branch based on whether scheduling is configured
    if self.schedule:
        self.submit_k8s_cronjob(job)
    else:
        self.submit_k8s_job(job)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/extensions/job_executor/test_k8s_scheduling.py::test_submit_job_creates_cronjob_when_schedule_present -v`
Expected: PASS

**Step 5: Run all scheduling tests**

Run: `pytest tests/extensions/job_executor/test_k8s_scheduling.py -v`
Expected: PASS (all tests)

**Step 6: Test backward compatibility with existing tests**

Run: `pytest tests/extensions/job_executor/ -k "k8s" -v`
Expected: PASS (existing K8s tests still work)

**Step 7: Commit**

```bash
git add extensions/job_executor/k8s.py tests/extensions/job_executor/test_k8s_scheduling.py
git commit -m "feat: branch submit_job to create CronJob when schedule is configured"
```

---

### Task 4: Update Derived Classes to Inherit Schedule Support

**Files:**
- Verify: `extensions/job_executor/k8s.py:366-433` (MiniK8sJobExecutor class)
- Verify: `extensions/job_executor/k8s.py:434-515` (K8sJobExecutor class)
- Test: `tests/extensions/job_executor/test_k8s_scheduling.py` (add derived class tests)

**Step 1: Write tests for derived classes**

Add to `tests/extensions/job_executor/test_k8s_scheduling.py`:

```python
from extensions.job_executor.k8s import MiniK8sJobExecutor, K8sJobExecutor

def test_mini_k8s_job_executor_inherits_schedule_support():
    """Test that MiniK8sJobExecutor supports scheduling"""
    config = {
        "job_spec": {"template": {"spec": {"container": {"image": "test"}}}},
        "schedule": "0 2 * * *"
    }
    executor = MiniK8sJobExecutor(**config)
    assert executor.schedule == "0 2 * * *"

def test_k8s_job_executor_inherits_schedule_support():
    """Test that K8sJobExecutor supports scheduling"""
    config = {
        "job_spec": {"template": {"spec": {"container": {"image": "test"}}}},
        "pvc_claim_name": "test-pvc",
        "schedule": "0 2 * * *"
    }
    executor = K8sJobExecutor(**config)
    assert executor.schedule == "0 2 * * *"

def test_derived_classes_submit_job_calls_parent_implementation():
    """Test that derived classes use the parent's submit_job logic"""
    # MiniK8sJobExecutor test
    config = {
        "job_spec": {"template": {"spec": {"container": {"image": "test"}}}},
        "schedule": "0 2 * * *",
        "mock": True
    }
    executor = MiniK8sJobExecutor(**config)
    executor._context = Mock()
    executor._context.run_id = "test-run-123"
    executor._context.run_log_store = Mock()
    executor._context.run_log_store.create_job_log.return_value = Mock()

    executor._set_up_run_log = Mock()
    executor._create_volumes = Mock()
    executor.submit_k8s_cronjob = Mock()

    mock_task = Mock(spec=BaseTaskType)
    executor.submit_job(mock_task, catalog_settings=[])

    # Should call the CronJob method since schedule is set
    executor.submit_k8s_cronjob.assert_called_once_with(mock_task)
```

**Step 2: Run test to verify derived classes work**

Run: `pytest tests/extensions/job_executor/test_k8s_scheduling.py::test_mini_k8s_job_executor_inherits_schedule_support -v`
Expected: PASS (inheritance should work automatically)

**Step 3: Verify no changes needed to derived classes**

Check that MiniK8sJobExecutor and K8sJobExecutor inherit the schedule field and submit_job behavior automatically since they inherit from GenericK8sJobExecutor.

**Step 4: Run all derived class tests**

Run: `pytest tests/extensions/job_executor/test_k8s_scheduling.py -k "derived" -v`
Expected: PASS (all derived class tests)

**Step 5: Commit (if any changes were needed)**

```bash
git add tests/extensions/job_executor/test_k8s_scheduling.py
git commit -m "test: verify derived K8s executors inherit schedule support"
```

---

### Task 5: Add Example Configuration File

**Files:**
- Create: `examples/11-jobs/k8s-scheduled-job.yaml`
- Test: Manual verification of configuration

**Step 1: Create example scheduled job configuration**

Create file: `examples/11-jobs/k8s-scheduled-job.yaml`

```yaml
job-executor:
  type: "k8s-job"
  config:
    pvc_claim_name: runnable
    config_path:
    mock: false
    namespace: enterprise-mlops
    schedule: "0 2 * * *"  # Run daily at 2 AM
    jobSpec:
      activeDeadlineSeconds: 32000
      template:
        spec:
          activeDeadlineSeconds: 86400
          container:
            image: harbor.csis.astrazeneca.net/mlops/runnable:latest
            resources:
              limits:
                cpu: "1"
                memory: "2Gi"
              requests:
                cpu: "500m"
                memory: "1Gi"
```

**Step 2: Create test to verify example config is valid**

Add to `tests/extensions/job_executor/test_k8s_scheduling.py`:

```python
import yaml
from pathlib import Path

def test_example_scheduled_config_is_valid():
    """Test that the example scheduled job config file is valid"""
    config_path = Path("examples/11-jobs/k8s-scheduled-job.yaml")
    assert config_path.exists(), "Example config file should exist"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    job_executor_config = config["job-executor"]["config"]

    # Should be able to create executor with this config
    executor = K8sJobExecutor(**job_executor_config)
    assert executor.schedule == "0 2 * * *"
    assert executor.pvc_claim_name == "runnable"
```

**Step 3: Run test to verify example config works**

Run: `pytest tests/extensions/job_executor/test_k8s_scheduling.py::test_example_scheduled_config_is_valid -v`
Expected: PASS

**Step 4: Commit example configuration**

```bash
git add examples/11-jobs/k8s-scheduled-job.yaml tests/extensions/job_executor/test_k8s_scheduling.py
git commit -m "docs: add example configuration for scheduled K8s jobs"
```

---

### Task 6: Add Documentation

**Files:**
- Create: `docs/job-executors/k8s-scheduling.md`
- Modify: `docs/job-executors/k8s.md` (add scheduling section if it exists)

**Step 1: Create scheduling documentation**

Create file: `docs/job-executors/k8s-scheduling.md`

```markdown
# Kubernetes Job Scheduling

The Kubernetes job executors support optional cron-based scheduling using Kubernetes CronJobs.

## Configuration

Add a `schedule` field to your K8s job executor configuration:

```yaml
job-executor:
  type: "k8s-job"
  config:
    schedule: "0 2 * * *"  # Daily at 2 AM UTC
    # ... other configuration
```

## Schedule Format

The schedule field accepts standard cron expressions with 5 fields:

- `minute` (0-59)
- `hour` (0-23)
- `day of month` (1-31)
- `month` (1-12)
- `day of week` (0-6, Sunday=0)

### Examples

```yaml
# Every day at 2 AM
schedule: "0 2 * * *"

# Every hour
schedule: "0 * * * *"

# Every Monday at 9 AM
schedule: "0 9 * * 1"

# Every 15 minutes
schedule: "*/15 * * * *"
```

## Behavior

### When Schedule is Present
- Creates a Kubernetes CronJob instead of a regular Job
- Displays scheduled job information to console
- No immediate execution - job runs according to schedule
- Kubernetes handles the scheduling automatically

### When Schedule is Absent
- Normal behavior (immediate Job execution)
- Backward compatible with existing configurations

## Example

Complete configuration example:

```yaml
--8<-- "examples/11-jobs/k8s-scheduled-job.yaml"
```

To schedule this job:

```bash
runnable execute pipeline.yaml --config k8s-scheduled-job.yaml
```

Output:
```
✓ CronJob scheduled successfully
  Name: run-20231129-143022-123
  Namespace: enterprise-mlops
  Schedule: 0 2 * * *

  Job Spec:
  - Image: harbor.csis.astrazeneca.net/mlops/runnable:latest
  - Resources: {'limits': {'cpu': '1', 'memory': '2Gi'}}
```

## Supported Executors

All K8s job executor variants support scheduling:

- `K8sJobExecutor` - Production with PVC
- `MiniK8sJobExecutor` - Local minikube development
- `GenericK8sJobExecutor` - Base class

## Troubleshooting

### Invalid Cron Expression
```
ValidationError: Schedule must be a valid cron expression with 5 fields
```
Ensure your cron expression has exactly 5 space-separated fields.

### Permission Issues
```
Forbidden: CronJob creation not allowed in namespace
```
Ensure your Kubernetes service account has permissions to create CronJobs in the target namespace.
```

**Step 2: Add test to ensure documentation example works**

Add to `tests/extensions/job_executor/test_k8s_scheduling.py`:

```python
def test_documentation_examples_are_valid():
    """Test that cron expressions in documentation are valid"""
    examples = [
        "0 2 * * *",    # Daily at 2 AM
        "0 * * * *",    # Every hour
        "0 9 * * 1",    # Monday at 9 AM
        "*/15 * * * *"  # Every 15 minutes
    ]

    for schedule in examples:
        config = {
            "job_spec": {"template": {"spec": {"container": {"image": "test"}}}},
            "schedule": schedule
        }
        executor = GenericK8sJobExecutor(**config)
        assert executor.schedule == schedule
```

**Step 3: Run documentation tests**

Run: `pytest tests/extensions/job_executor/test_k8s_scheduling.py::test_documentation_examples_are_valid -v`
Expected: PASS

**Step 4: Commit documentation**

```bash
git add docs/job-executors/k8s-scheduling.md tests/extensions/job_executor/test_k8s_scheduling.py
git commit -m "docs: add comprehensive K8s job scheduling documentation"
```

---

### Task 7: Integration Testing and Validation

**Files:**
- Test: `tests/extensions/job_executor/test_k8s_scheduling.py` (add end-to-end tests)
- Verify: Run full test suite

**Step 1: Add comprehensive integration tests**

Add to `tests/extensions/job_executor/test_k8s_scheduling.py`:

```python
def test_end_to_end_scheduled_job_creation():
    """Test complete flow from configuration to CronJob creation"""
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
                            "requests": {"cpu": "500m", "memory": "512Mi"}
                        }
                    }
                }
            }
        },
        "schedule": "0 3 * * *",
        "namespace": "test-namespace",
        "mock": True
    }

    executor = GenericK8sJobExecutor(**config)
    executor._context = Mock()
    executor._context.run_id = "integration-test-run"
    executor._context.get_job_callable_command.return_value = "python main.py"

    with patch.object(executor, '_client') as mock_client:
        mock_batch_api = Mock()
        mock_client.BatchV1Api.return_value = mock_batch_api

        mock_task = Mock(spec=BaseTaskType)
        executor.submit_k8s_cronjob(mock_task)

        # Verify the call was made
        mock_batch_api.create_namespaced_cron_job.assert_called_once()

        # Verify the CronJob structure
        call_args = mock_batch_api.create_namespaced_cron_job.call_args
        cronjob = call_args[1]['body']

        assert cronjob.kind == "CronJob"
        assert cronjob.api_version == "batch/v1"
        assert cronjob.metadata.name == "integration-test-run"
        assert cronjob.spec.schedule == "0 3 * * *"

        # Verify job template contains our job spec
        job_template = cronjob.spec.job_template
        assert job_template.spec.active_deadline_seconds == 3600
        assert job_template.spec.template.spec.restart_policy == "Never"

def test_error_handling_in_cronjob_creation():
    """Test that CronJob creation errors are handled properly"""
    config = {
        "job_spec": {"template": {"spec": {"container": {"image": "test"}}}},
        "schedule": "0 2 * * *",
        "mock": False  # Use real mode to test error handling
    }

    executor = GenericK8sJobExecutor(**config)
    executor._context = Mock()
    executor._context.run_id = "error-test-run"
    executor._context.get_job_callable_command.return_value = "python test.py"

    with patch.object(executor, '_client') as mock_client:
        mock_batch_api = Mock()
        mock_client.BatchV1Api.return_value = mock_batch_api

        # Simulate K8s API error
        from kubernetes.client.exceptions import ApiException
        mock_batch_api.create_namespaced_cron_job.side_effect = ApiException(
            status=403, reason="Forbidden"
        )

        mock_task = Mock(spec=BaseTaskType)

        # Should re-raise the exception
        with pytest.raises(ApiException):
            executor.submit_k8s_cronjob(mock_task)
```

**Step 2: Run integration tests**

Run: `pytest tests/extensions/job_executor/test_k8s_scheduling.py::test_end_to_end_scheduled_job_creation -v`
Expected: PASS

**Step 3: Run full test suite for K8s executors**

Run: `pytest tests/extensions/job_executor/ -k "k8s" -v`
Expected: PASS (all existing and new tests)

**Step 4: Run project test suite to ensure no regressions**

Run: `pytest tests/ -x`
Expected: PASS (no breaking changes to other components)

**Step 5: Manual testing with mock mode**

Create a simple test script to verify console output:

```python
# test_console_output.py
from extensions.job_executor.k8s import GenericK8sJobExecutor
from unittest.mock import Mock

config = {
    "job_spec": {"template": {"spec": {"container": {"image": "test:latest"}}}},
    "schedule": "0 2 * * *",
    "mock": True
}

executor = GenericK8sJobExecutor(**config)
executor._context = Mock()
executor._context.run_id = "manual-test-run"
executor._context.get_job_callable_command.return_value = "python pipeline.py"
executor._context.run_log_store = Mock()
executor._context.run_log_store.create_job_log.return_value = Mock()

mock_task = Mock()
print("Testing scheduled job console output:")
executor.submit_job(mock_task, catalog_settings=[])
```

Run: `python test_console_output.py`
Expected: Console output showing scheduled job information

**Step 6: Commit integration tests**

```bash
git add tests/extensions/job_executor/test_k8s_scheduling.py
git commit -m "test: add comprehensive integration tests for K8s job scheduling"
```

---

### Task 8: Final Verification and Documentation Update

**Files:**
- Verify: All tests pass
- Update: Any main project documentation if needed

**Step 1: Run complete test suite**

Run: `pytest`
Expected: PASS (all tests including new scheduling tests)

**Step 2: Verify example configuration works**

Run: `python -c "
import yaml
from extensions.job_executor.k8s import K8sJobExecutor

with open('examples/11-jobs/k8s-scheduled-job.yaml') as f:
    config = yaml.safe_load(f)

executor = K8sJobExecutor(**config['job-executor']['config'])
print(f'Schedule: {executor.schedule}')
print('Configuration validation: PASSED')
"`
Expected: Successful configuration loading

**Step 3: Check for any missing imports**

Verify that all necessary imports are present in `extensions/job_executor/k8s.py`:

```python
import logging
import re
import shlex
from enum import Enum
from typing import Annotated, List, Optional

from kubernetes import client
from kubernetes import config as k8s_config
from pydantic import BaseModel, ConfigDict, Field, PlainSerializer, PrivateAttr, field_validator
```

**Step 4: Update main README or documentation if needed**

Check if the main project documentation mentions job executors and add a reference to the new scheduling capability if appropriate.

**Step 5: Final commit**

```bash
git add .
git commit -m "feat: complete K8s CronJob scheduling implementation

- Add optional schedule field to all K8s job executors
- Create CronJobs instead of Jobs when schedule is configured
- Maintain full backward compatibility
- Include comprehensive tests and documentation
- Add example configuration file"
```

**Step 6: Create summary of changes**

The implementation adds:

1. **Configuration**: Optional `schedule` field with cron validation
2. **CronJob Support**: Creates Kubernetes CronJobs when scheduled
3. **Console Output**: Shows scheduled job information
4. **Backward Compatibility**: No breaking changes to existing functionality
5. **Documentation**: Complete usage guide and examples
6. **Testing**: Comprehensive test coverage including edge cases

**Verification Commands:**
```bash
# Run all tests
pytest

# Test new scheduling functionality specifically
pytest tests/extensions/job_executor/test_k8s_scheduling.py -v

# Verify no regressions in existing K8s functionality
pytest tests/extensions/job_executor/ -k "k8s" -v

# Validate example configuration
python -c "import yaml; from extensions.job_executor.k8s import K8sJobExecutor; K8sJobExecutor(**yaml.safe_load(open('examples/11-jobs/k8s-scheduled-job.yaml'))['job-executor']['config'])"
```

---

## Plan Summary

This implementation adds optional cron scheduling to Kubernetes job executors through:

1. **Schedule Configuration Field** - Optional cron expression field with validation
2. **CronJob Creation Logic** - Method to create K8s CronJobs instead of Jobs
3. **Execution Flow Branching** - submit_job method branches based on schedule presence
4. **Inheritance Support** - All derived executor classes automatically support scheduling
5. **Example Configuration** - Complete working example with documentation
6. **Comprehensive Documentation** - Usage guide with examples and troubleshooting
7. **Integration Testing** - End-to-end tests ensuring reliability
8. **Final Verification** - Complete test suite validation

The implementation maintains full backward compatibility while adding powerful scheduling capabilities that leverage Kubernetes' native CronJob functionality.
