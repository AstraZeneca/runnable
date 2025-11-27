# Building Custom Pipeline Executors

Execute pipelines on any orchestration platform by understanding the two fundamental orchestration patterns.

## The Two Orchestration Patterns

Pipeline executors handle DAG execution in two fundamentally different ways:

### Pattern 1: DAG Traversal (Local, Local-Container)
**How it works:**

- Runnable **traverses the DAG** in Python
- For each node: Calls `trigger_node_execution()`
- Node execution happens locally or via individual job submission

**Use for:** Simple platforms that submit individual jobs

### Pattern 2: DAG Transpilation (Argo Workflows)
**How it works:**

- Runnable **converts the entire DAG** into the platform's native workflow format
- The **platform handles DAG traversal** (Argo's workflow engine, Airflow scheduler, etc.)
- Individual nodes use CLI pattern: `runnable execute-single-node ...`

**Use for:** Platforms with native workflow capabilities (Argo, Airflow, Prefect)

## Implementation Examples

### Pattern 1: DAG Traversal (Local Executor)
```python
class LocalExecutor(GenericPipelineExecutor):
    service_name: str = "local"

    def trigger_node_execution(self, node, map_variable=None):
        """Called for each node - execute directly"""
        # Direct execution using base class
        self.execute_node(node=node, map_variable=map_variable)

    def execute_node(self, node, map_variable=None):
        """Execute the node directly in current process"""
        self._execute_node(node=node, map_variable=map_variable)
```

**Key insight**: Runnable traverses the DAG, calls `trigger_node_execution()` for each node, which directly executes the task.

### Pattern 2: DAG Transpilation (Argo Executor)
```python
class ArgoExecutor(GenericPipelineExecutor):
    service_name: str = "argo"

    def execute_from_graph(self, dag, map_variable=None):
        """Convert entire DAG to Argo WorkflowTemplate"""
        # Transpile runnable DAG to Argo workflow YAML
        argo_workflow = self._transpile_dag_to_argo_workflow(dag)
        # Submit to Kubernetes - Argo handles the DAG traversal
        self._submit_argo_workflow(argo_workflow)

    def trigger_node_execution(self, node, map_variable=None):
        """This runs INSIDE Argo pods via CLI commands"""
        # Called by: runnable execute-single-node <node-name>
        # Set up storage access for the pod environment
        self._setup_pod_storage_access()
        # Execute the node
        self._execute_node(node=node, map_variable=map_variable)
```

**Key insight**: Argo's workflow engine traverses the DAG, calls CLI commands that invoke `trigger_node_execution()` in each pod.

## The Critical Issue: Storage Access

**Same issue as job executors**: Run logs and catalog must be accessible in the execution environment.

### Local Execution: Direct Access
```python
class LocalExecutor(GenericPipelineExecutor):
    def trigger_node_execution(self, node, map_variable=None):
        # Storage accessible locally - proceed directly
        self._execute_node(node=node, map_variable=map_variable)
```

### Remote Execution: Volume Mounting

**Local-Container Pattern** - Mount host directories into containers:
```python
class LocalContainerExecutor(GenericPipelineExecutor):
    def trigger_node_execution(self, node, map_variable=None):
        # Mount host storage into container
        self._mount_volumes()
        # Run CLI command in container
        command = f"runnable execute-single-node {node.name}"
        self._run_in_container(command)

    def _mount_volumes(self):
        # Map host paths to container paths
        if self._context.run_log_store.service_name == "file-system":
            host_logs = self._context.run_log_store.log_folder
            self._volumes[host_logs] = {"bind": "/tmp/run_logs/", "mode": "rw"}

        if self._context.catalog.service_name == "file-system":
            host_catalog = self._context.catalog.catalog_location
            self._volumes[host_catalog] = {"bind": "/tmp/catalog/", "mode": "rw"}
```

**Argo/K8s Pattern** - Use PersistentVolumeClaims:
```python
class ArgoExecutor(GenericPipelineExecutor):
    def _transpile_dag_to_argo_workflow(self, dag):
        # Add PVC mounts to every pod in the workflow
        workflow_spec = {
            "spec": {
                "volumes": [
                    {"name": "run-logs", "persistentVolumeClaim": {"claimName": "runnable-logs-pvc"}},
                    {"name": "catalog", "persistentVolumeClaim": {"claimName": "runnable-catalog-pvc"}}
                ],
                "templates": [
                    {
                        "container": {
                            "volumeMounts": [
                                {"name": "run-logs", "mountPath": "/tmp/run_logs/"},
                                {"name": "catalog", "mountPath": "/tmp/catalog/"}
                            ]
                        }
                    }
                ]
            }
        }

    def trigger_node_execution(self, node, map_variable=None):
        # This runs in Argo pod - update context to use mounted paths
        self._use_mounted_storage()
        self._execute_node(node=node, map_variable=map_variable)

    def _use_mounted_storage(self):
        # Point to PVC mount paths
        if self._context.run_log_store.service_name == "file-system":
            self._context.run_log_store.log_folder = "/tmp/run_logs/"
        if self._context.catalog.service_name == "file-system":
            self._context.catalog.catalog_location = "/tmp/catalog/"
```

**The pattern**: Make sure run logs and catalog are accessible in every execution environment (container, pod, remote job).

## Plugin Registration

Create your executor and register it:

```python
from extensions.pipeline_executor import GenericPipelineExecutor
from pydantic import Field

class MyPlatformExecutor(GenericPipelineExecutor):
    service_name: str = "my-platform"

    # Your platform config fields
    api_endpoint: str = Field(...)
    project_id: str = Field(...)

    def trigger_node_execution(self, node, map_variable=None):
        # Your platform node execution logic
        pass

    def execute_from_graph(self, dag, map_variable=None):
        # Optional: For DAG transpilation platforms only
        pass
```

**Register in `pyproject.toml`:**
```toml
[project.entry-points.'pipeline_executor']
"my-platform" = "my_package.executors:MyPlatformExecutor"
```

## Which Pattern to Choose?

**DAG Traversal** (`trigger_node_execution` only):

- **For**: Simple batch platforms (AWS Batch, SLURM, etc.)
- **How**: Runnable calls your method for each node
- **Storage**: Handle volumes/mounts in `trigger_node_execution()`

**DAG Transpilation** (both methods):

- **For**: Workflow platforms (Argo, Airflow, Prefect, etc.)
- **How**: Convert entire DAG to platform's native workflow format
- **Storage**: Handle volumes/mounts in the transpiled workflow spec

The complexity is in translating DAG semantics (parallel branches, conditionals) to your platform's workflow language.

## Integration Advantage

**🔑 Key Benefit**: Custom executors live entirely in **your codebase**, not in public repositories or external dependencies.

### Complete Control & Privacy

```python
# In your private repository
# my-company/internal-ml-platform/executors/company_executor.py

class CompanyBatchExecutor(GenericPipelineExecutor):
    service_name: str = "company-batch"

    # Your internal configuration
    internal_api_endpoint: str = Field(...)
    security_group: str = Field(...)
    compliance_tags: dict = Field(default_factory=dict)

    def trigger_node_execution(self, node, map_variable=None):
        # Your proprietary integration logic
        # Company-specific security, monitoring, cost tracking
        pass
```

**Integration benefits:**

- **🔒 Security**: No external dependencies or public code exposure
- **🏢 Compliance**: Implement organization-specific governance and audit requirements
- **💰 Cost Control**: Integrate with internal cost tracking and resource management
- **🔧 Customization**: Build reusable templates for your exact infrastructure
- **📊 Monitoring**: Integrate with dashboards and alerting systems

### Reusable Templates

Teams can create internal libraries of executors:

```python
# Internal package: company-runnable-executors
from company_runnable_executors import (
    ProductionK8sExecutor,      # Your Kubernetes setup
    StagingBatchExecutor,       # Your staging environment
    ComplianceExecutor,         # SOC2/HIPAA requirements
    CostOptimizedExecutor,      # Spot instances + cost tracking
)

# Teams use your standardized executors
class MLTrainingPipeline(Pipeline):
    def production_config(self):
        return ProductionK8sExecutor(
            namespace="ml-prod",
            resource_limits=self.get_approved_limits(),
            compliance_mode=True
        )
```

### Ecosystem Integration

```yaml
# Your company's standard configuration templates
pipeline-executor:
  type: company-batch
  config:
    internal_api_endpoint: "https://internal-batch.company.com"
    security_group: "ml-workloads-sg"
    compliance_tags:
      project: "{{PROJECT_ID}}"
      cost_center: "{{COST_CENTER}}"
      data_classification: "confidential"
    monitoring:
      dashboard_url: "https://company-monitoring.com/runnable"
      alert_channels: ["#ml-alerts", "#devops-alerts"]
```

**This makes runnable a platform for building your internal ML infrastructure**, not just using external services.

## Need Help?

**Custom pipeline executors are complex integrations** that require deep understanding of both runnable's architecture and your target platform's orchestration model.

!!! question "Get Support"

    **We're here to help you succeed!** Building custom executors involves intricate details about:

    - Graph traversal and dependency management
    - Step log coordination and error handling
    - Parameter passing and context management
    - Platform-specific workflow translation patterns

    **Don't hesitate to reach out:**

    - 📧 **Contact the team** for architecture guidance and implementation support
    - 🤝 **Collaboration opportunities** - we're interested in supporting enterprise integrations
    - 📖 **Documentation feedback** - help us improve these guides based on your experience

    **Better together**: Complex orchestration integrations benefit from collaboration between platform experts (you) and runnable architecture experts (us).

!!! warning "Highly Complex Integration"

    **These are among the most sophisticated integrations in runnable** that involve:

    - Deep understanding of runnable's graph execution engine and step lifecycle
    - Complex orchestration platform APIs and workflow specification formats
    - Distributed execution coordination, failure handling, and state management
    - Advanced container orchestration, networking, and resource management patterns

    **Success requires significant expertise in both domains.** The existing orchestration integrations (especially Argo) took substantial development effort to get right - collaboration dramatically increases your chances of success.

Your success with custom pipeline executors helps the entire runnable community!
