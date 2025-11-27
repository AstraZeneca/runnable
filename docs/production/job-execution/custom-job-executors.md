# Building Custom Job Executors

Execute jobs on any compute platform by understanding the two fundamental execution patterns.

## The Core Patterns

There are only two ways runnable executes jobs:

### Pattern 1: Direct Execution (Local)
Execute the job function directly in the current process - no CLI command needed.

### Pattern 2: Remote Execution (Container, K8s, Cloud)
Cross environment boundaries by using the CLI command to recreate the job context in the remote environment.

```bash
# Remote executors run this CLI command in the target environment
runnable execute-job my_job.py run_123 --config config.yaml --parameters params.yaml
```

## Implementation Examples

### Local Executor - Direct Execution
```python
def submit_job(self, job, catalog_settings):
    # Set up run log locally (storage is accessible)
    self._set_up_run_log()
    # Execute directly in same process
    self.execute_job(job, catalog_settings)

def execute_job(self, job, catalog_settings):
    # Direct function call - no CLI command
    attempt_log = job.execute_command()
    # Handle results and catalog sync
    self._sync_catalog(catalog_settings)
```

### Remote Executor - CLI Command Execution
```python
def submit_job(self, job, catalog_settings):
    # Do NOT set up run log here - storage not accessible locally!
    # Make storage accessible to remote environment
    self._setup_remote_storage_access()
    # Get the CLI command and run it remotely
    command = self._context.get_job_callable_command()
    self._execute_on_remote_platform(command)

def execute_job(self, job, catalog_settings):
    # This runs IN the remote environment
    # Now we can set up run log - storage is accessible here
    self._set_up_run_log()
    # Point to mounted/accessible storage locations
    self._use_remote_storage_locations()
    # Execute the job
    attempt_log = job.execute_command()
    # Handle results and catalog sync
    self._sync_catalog(catalog_settings)
```

## The Critical Issue: Storage Access

**⚠️ Common Bug**: Calling `self._set_up_run_log()` when storage isn't accessible locally.

**Local execution**: Storage is accessible from your machine
**Remote execution**: Storage is only accessible inside the remote environment

### Real Example: Local-Container Storage Access

The local-container executor shows the correct pattern:

```python
class LocalContainerJobExecutor(GenericJobExecutor):

    def submit_job(self, job, catalog_settings):
        # Set up run log BEFORE container (storage accessible locally)
        self._set_up_run_log()
        # Make storage accessible to container
        self._mount_volumes()
        # Run CLI command in container
        self.spin_container()

    def _mount_volumes(self):
        """Mount local storage into container"""
        # Run log store
        if self._context.run_log_store.service_name == "file-system":
            host_path = self._context.run_log_store.log_folder
            container_path = "/tmp/run_logs/"
            self._volumes[host_path] = {"bind": container_path, "mode": "rw"}

        # Catalog storage
        if self._context.catalog.service_name == "file-system":
            host_path = self._context.catalog.catalog_location
            container_path = "/tmp/catalog/"
            self._volumes[host_path] = {"bind": container_path, "mode": "rw"}

    def execute_job(self, job, catalog_settings):
        # This runs INSIDE the container
        # Point to mounted locations
        self._use_volumes()
        # Now execute
        attempt_log = job.execute_command()

    def _use_volumes(self):
        """Update context to use mounted paths"""
        if self._context.run_log_store.service_name == "file-system":
            self._context.run_log_store.log_folder = "/tmp/run_logs/"

        if self._context.catalog.service_name == "file-system":
            self._context.catalog.catalog_location = "/tmp/catalog/"
```

### For Cloud Platforms

**K8s**: Use PersistentVolumeClaims
**AWS Batch**: Mount EFS or use S3-compatible storage
**Your Platform**: However your platform provides shared storage

The pattern is always:

1. `submit_job()`: Make storage accessible to remote environment
2. `execute_job()`: Use the accessible storage locations

## Plugin Registration

Create your executor class and register it:

```python
from extensions.job_executor import GenericJobExecutor
from pydantic import Field

class MyPlatformJobExecutor(GenericJobExecutor):
    service_name: str = "my-platform"

    # Your platform config fields
    api_endpoint: str = Field(...)
    queue_name: str = Field(...)

    def submit_job(self, job, catalog_settings):
        # Your platform submission logic
        pass

    def execute_job(self, job, catalog_settings):
        # Your execution logic (runs in remote environment)
        pass
```

**Register in `pyproject.toml`:**
```toml
[project.entry-points.'job_executor']
"my-platform" = "my_package.executors:MyPlatformJobExecutor"
```

**Use in configuration:**
```yaml
job-executor:
  type: my-platform
  config:
    api_endpoint: "https://my-platform.com"
    queue_name: "production"
```

## Integration Advantage

**🔑 Key Benefit**: Custom job executors live entirely in **your codebase**, not in public repositories or external dependencies.

### Complete Control & Privacy

```python
# In your private repository
# my-company/internal-compute/executors/company_executor.py

class CompanyHPCJobExecutor(GenericJobExecutor):
    service_name: str = "company-hpc"

    # Your internal configuration
    internal_scheduler_api: str = Field(...)
    security_domain: str = Field(...)
    cost_tracking_enabled: bool = Field(default=True)
    compliance_level: str = Field(default="confidential")

    def submit_job(self, job, catalog_settings):
        # Your proprietary HPC integration
        # Company-specific security, audit trails, resource allocation
        pass

    def execute_job(self, job, catalog_settings):
        # Company-specific monitoring and logging
        # Internal cost tracking and usage metrics
        pass
```

**Integration benefits:**

- **🔒 Security**: No external dependencies or public code exposure
- **🏢 Compliance**: Implement organization-specific security and audit requirements
- **💰 Cost Control**: Direct integration with internal cost tracking and budgeting systems
- **🔧 Customization**: Build job executors for your exact compute infrastructure
- **📊 Monitoring**: Seamless integration with monitoring and alerting systems

### Reusable Templates

Teams can create internal libraries of job executors:

```python
# Internal package: company-runnable-executors
from company_runnable_executors import (
    ProductionHPCExecutor,      # Your HPC cluster setup
    DevelopmentGPUExecutor,     # Development GPU nodes
    ComplianceJobExecutor,      # SOC2/HIPAA job execution
    SpotInstanceExecutor,       # Cost-optimized cloud compute
)

# Teams use your standardized executors
job = PythonJob(function=train_model)
job.execute(configuration_file="production-hpc-config.yaml")
```

### Ecosystem Integration

```yaml
# Your company's standard job execution templates
job-executor:
  type: company-hpc
  config:
    internal_scheduler_api: "https://hpc-scheduler.company.com"
    security_domain: "ml-compute-domain"
    cost_tracking_enabled: true
    compliance_level: "confidential"
    resource_limits:
      max_cpu: "64"
      max_memory: "512Gi"
      max_gpu: "8"
    monitoring:
      metrics_endpoint: "https://company-metrics.com/jobs"
      alert_channels: ["#compute-alerts", "#ml-ops"]
    audit:
      log_level: "detailed"
      retention_days: 365
```

**This makes runnable a platform for standardizing job execution across your entire compute infrastructure** - from development laptops to production HPC clusters.

## Need Help?

**Custom job executors involve complex cloud service integrations** that require understanding both runnable's job execution model and your target platform's batch processing capabilities.

!!! question "Get Support"

    **We're here to help you succeed!** Building custom job executors involves detailed knowledge of:

    - Job context and command generation
    - Run log and job log coordination
    - Catalog synchronization patterns
    - Platform-specific job submission and monitoring

    **Don't hesitate to reach out:**

    - 📧 **Contact the team** for architecture guidance and integration support
    - 🤝 **Collaboration opportunities** - we're interested in supporting cloud platform integrations
    - 📖 **Documentation feedback** - help us improve these guides based on your implementation experience

    **Better together**: Cloud service integrations benefit from collaboration between platform experts (you) and runnable job execution experts (us).

!!! warning "Complex Integration"

    **These are sophisticated cloud integrations** that involve:

    - Understanding runnable's job execution lifecycle and context management
    - Integrating with cloud APIs that have varying reliability and rate limits
    - Handling distributed execution, networking, and failure scenarios
    - Managing container images, environment variables, and resource allocation

    **Success is much more likely with collaboration.** The existing cloud integrations required deep understanding of both runnable internals and platform specifics - leverage our experience to avoid common pitfalls.

Your success with custom job executors helps the entire runnable community!

---
