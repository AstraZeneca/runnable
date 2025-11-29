# Kubernetes Job Scheduling

Schedule jobs to run automatically using Kubernetes CronJobs with cron expressions.

!!! info "Installation Required"

    Kubernetes execution requires the optional Kubernetes dependency:
    ```bash
    pip install runnable[k8s]
    ```

## Why Use Job Scheduling?

!!! success "Scheduling Benefits"

    **Automate recurring workflows**: Schedule jobs to run on a regular cadence

    - ⏰ **Cron-based scheduling**: Use familiar cron expressions
    - 🔄 **Automatic execution**: Kubernetes handles job execution
    - 📊 **No immediate execution**: Schedule and let Kubernetes manage it
    - 🎯 **Production-ready**: Native Kubernetes CronJob support

!!! note "Trade-offs"

    - ⏱️ **No immediate results**: Job runs according to schedule, not immediately
    - 🔍 **Monitoring required**: Track scheduled job execution via Kubernetes
    - 📝 **UTC timezone**: Kubernetes CronJobs run in UTC timezone

## Basic Configuration

Add a `schedule` field to your Kubernetes job executor configuration:

```yaml
job-executor:
  type: "k8s-job"
  config:
    schedule: "0 2 * * *"  # Daily at 2 AM UTC
    pvc_claim_name: "runnable-storage"
    jobSpec:
      template:
        spec:
          container:
            image: "my-project:v1.0"
```

!!! tip "When Schedule is Present"

    - Creates a Kubernetes CronJob instead of a regular Job
    - Displays scheduled job information to console
    - No immediate execution - job runs according to schedule
    - Kubernetes handles the scheduling automatically

!!! info "When Schedule is Absent"

    Normal behavior with immediate Job execution - fully backward compatible with existing configurations.

## Schedule Format

The schedule field accepts standard cron expressions with 5 fields:

```
┌───────────── minute (0 - 59)
│ ┌───────────── hour (0 - 23)
│ │ ┌───────────── day of month (1 - 31)
│ │ │ ┌───────────── month (1 - 12)
│ │ │ │ ┌───────────── day of week (0 - 6) (Sunday = 0)
│ │ │ │ │
│ │ │ │ │
* * * * *
```

### Common Schedule Examples

```yaml
# Every day at 2 AM UTC
schedule: "0 2 * * *"

# Every hour (at minute 0)
schedule: "0 * * * *"

# Every Monday at 9 AM UTC
schedule: "0 9 * * 1"

# Every 15 minutes
schedule: "*/15 * * * *"

# Every Sunday at 3:30 AM UTC
schedule: "30 3 * * 0"

# First day of month at midnight UTC
schedule: "0 0 1 * *"
```

## Complete Example

=== "job.py"

    ```python title="examples/11-jobs/python_tasks.py"
    from examples.common.functions import hello
    from runnable import PythonJob

    def main():
        job = PythonJob(function=hello)
        job.execute()
        return job

    if __name__ == "__main__":
        main()
    ```

=== "k8s-scheduled-job.yaml"

    ```yaml title="examples/11-jobs/k8s-scheduled-job.yaml"
    --8<-- "examples/11-jobs/k8s-scheduled-job.yaml"
    ```

**Schedule the job:**

```bash
# Push your image to the registry
docker push your-registry.com/your-project:latest

# Schedule the job with Kubernetes CronJob
RUNNABLE_CONFIGURATION_FILE=examples/11-jobs/k8s-scheduled-job.yaml uv run examples/11-jobs/python_tasks.py
```

**Expected output:**

```
✓ CronJob scheduled successfully
  Name: run-20231129-143022-123
  Namespace: enterprise-mlops
  Schedule: 0 2 * * *

  Job Spec:
  - Image: harbor.csis.astrazeneca.net/mlops/runnable:latest
  - Resources: {'limits': {'cpu': '1', 'memory': '2Gi'}}
```

!!! info "No Immediate Execution"

    When scheduling is enabled, the job does not execute immediately. Kubernetes will execute the job according to the specified schedule.

## Configuration Reference

### All Kubernetes Executors Support Scheduling

The `schedule` field is available for all Kubernetes job executor variants:

=== "Production Kubernetes (k8s-job)"

    ```yaml
    job-executor:
      type: k8s-job
      config:
        schedule: "0 2 * * *"              # Optional: Cron schedule
        pvc_claim_name: "runnable-storage" # Required: PVC for data
        namespace: "default"               # Optional: K8s namespace
        jobSpec:
          template:
            spec:
              container:
                image: "my-project:v1.0"   # Required: Docker image
    ```

=== "Minikube Development (mini-k8s-job)"

    ```yaml
    job-executor:
      type: mini-k8s-job
      config:
        schedule: "0 2 * * *"              # Optional: Cron schedule
        namespace: "default"               # Optional: K8s namespace
        jobSpec:
          template:
            spec:
              container:
                image: "my-project:latest" # Required: Docker image
    ```

### Schedule Configuration Options

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `schedule` | str | No | Cron expression (5 fields) for scheduling |
| `pvc_claim_name` | str | **Yes (k8s-job)** | PVC name for data persistence |
| `jobSpec.template.spec.container.image` | str | **Yes** | Docker image for job execution |
| `namespace` | str | No | Kubernetes namespace (default: "default") |

## Managing Scheduled Jobs

### View Scheduled CronJobs

```bash
# List all CronJobs
kubectl get cronjobs -n <namespace>

# Describe a specific CronJob
kubectl describe cronjob <run-id> -n <namespace>

# View CronJob schedule
kubectl get cronjob <run-id> -n <namespace> -o jsonpath='{.spec.schedule}'
```

### View Job Executions

```bash
# List jobs created by CronJob
kubectl get jobs -n <namespace> -l cronjob=<run-id>

# View pods from scheduled jobs
kubectl get pods -n <namespace> -l job-name=<job-name>

# Check job logs
kubectl logs -n <namespace> -l job-name=<job-name>
```

### Delete Scheduled CronJob

```bash
# Delete a CronJob (stops future executions)
kubectl delete cronjob <run-id> -n <namespace>
```

## Troubleshooting

### Invalid Cron Expression

!!! warning "ValidationError: Schedule must be a valid cron expression"

    **Problem**: Cron expression format is incorrect

    **Solution**: Ensure your cron expression has exactly 5 space-separated fields

    ```yaml
    # ❌ Wrong: Too few fields
    schedule: "0 2 * *"

    # ❌ Wrong: Too many fields
    schedule: "0 0 2 * * *"

    # ✓ Correct: Exactly 5 fields
    schedule: "0 2 * * *"
    ```

### Permission Issues

!!! warning "Forbidden: CronJob creation not allowed"

    **Problem**: Service account lacks permissions to create CronJobs

    **Solution**: Ensure your Kubernetes service account has permissions

    ```bash
    # Check current permissions
    kubectl auth can-i create cronjobs -n <namespace>

    # Grant permissions (requires cluster admin)
    kubectl create clusterrolebinding cronjob-creator \
      --clusterrole=edit \
      --serviceaccount=<namespace>:<service-account>
    ```

### CronJob Not Executing

!!! warning "CronJob created but jobs not running"

    **Problem**: CronJob exists but no Job executions

    **Solution**: Check CronJob status and events

    ```bash
    # Check CronJob status
    kubectl describe cronjob <run-id> -n <namespace>

    # View recent events
    kubectl get events -n <namespace> --sort-by='.lastTimestamp'

    # Verify CronJob is not suspended
    kubectl get cronjob <run-id> -n <namespace> -o jsonpath='{.spec.suspend}'
    ```

### Timezone Considerations

!!! info "All Times are UTC"

    Kubernetes CronJobs always use UTC timezone. Convert your local time to UTC when setting schedules.

    ```python
    # Example: Run at 2 AM EST (7 AM UTC)
    schedule: "0 7 * * *"

    # Example: Run at 5 PM PST (1 AM next day UTC)
    schedule: "0 1 * * *"
    ```

## Debug Mode

Test scheduled job configuration without creating actual CronJobs:

```yaml
job-executor:
  type: k8s-job
  config:
    mock: true  # Logs CronJob spec without creating it
    schedule: "0 2 * * *"
    pvc_claim_name: "runnable-storage"
    jobSpec:
      template:
        spec:
          container:
            image: "my-project:latest"
```

## Cron Expression Reference

### Special Characters

| Character | Description | Example |
|-----------|-------------|---------|
| `*` | Any value | `* * * * *` = every minute |
| `,` | Value list | `0,30 * * * *` = every hour at :00 and :30 |
| `-` | Range of values | `0 9-17 * * *` = every hour from 9 AM to 5 PM |
| `/` | Step values | `*/5 * * * *` = every 5 minutes |

### Common Patterns

```yaml
# Every minute
schedule: "* * * * *"

# Every 5 minutes
schedule: "*/5 * * * *"

# Every hour at 30 minutes past
schedule: "30 * * * *"

# Every 6 hours
schedule: "0 */6 * * *"

# Every day at noon
schedule: "0 12 * * *"

# Every weekday at 9 AM
schedule: "0 9 * * 1-5"

# Every Saturday at midnight
schedule: "0 0 * * 6"

# First Monday of every month at 8 AM
schedule: "0 8 1-7 * 1"
```

## Best Practices

!!! tip "Scheduling Best Practices"

    **Resource Management**

    - Set appropriate resource limits for scheduled jobs
    - Consider cluster load during peak hours
    - Use different schedules to spread load

    **Monitoring**

    - Set up alerts for failed scheduled jobs
    - Monitor CronJob execution history
    - Track job completion times

    **Cleanup**

    - Configure `successfulJobsHistoryLimit` and `failedJobsHistoryLimit`
    - Delete CronJobs that are no longer needed
    - Archive old job logs

## When to Use Other Executors

Consider alternatives when you need:

!!! abstract "Immediate Execution"

    **[Kubernetes Jobs](kubernetes.md)**: For immediate job execution without scheduling

    **[Local](local.md)**: For simple development without containers

!!! note "Different Scheduling Systems"

    **External schedulers**: If you need features like:

    - Dependency-based scheduling
    - Complex conditional execution
    - Integration with existing scheduling systems

---

**Related:** [Kubernetes Jobs](kubernetes.md) | [Pipeline Argo Workflows](../pipeline-execution/argo.md) | [All Job Executors](overview.md)
