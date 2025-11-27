# Documentation Development

You are helping with documentation development for the Runnable framework. Focus on maintaining consistency with the existing documentation structure and style.

## Runnable Architecture Overview

Runnable uses a **plugin-based architecture** via Python entry points and stevedore for extensibility. The framework has four core concepts that are implemented as pluggable services:

### 1. Pipeline Executor
**Purpose**: Orchestrates the overall execution flow of pipelines
**Entry Point**: `[project.entry-points.'pipeline_executor']`

Available services and their key configurations:

- **`local`** (`LocalExecutor`) - Direct local execution
  - `enable_parallel: bool = False` - Enable parallel execution
  - `object_serialisation: bool = True` - Control object serialization
  - `overrides: dict[str, Any] = {}` - Node-specific configuration overrides

- **`local-container`** (`LocalContainerExecutor`) - Containerized local execution
- **`emulator`** (`Emulator`) - Simulation/testing execution
- **`argo`** (`ArgoExecutor`) - Argo Workflows execution
- **`mocked`** (`MockedExecutor`) - Mock execution for testing
- **`retry`** (`RetryExecutor`) - Retry wrapper execution

### 2. Job Executor
**Purpose**: Handles the actual execution of individual tasks/jobs
**Entry Point**: `[project.entry-points.'job_executor']`

Available services:
- **`local`** (`LocalJobExecutor`) - Direct local job execution
- **`local-container`** (`LocalContainerJobExecutor`) - Containerized job execution
- **`mini-k8s-job`** (`MiniK8sJobExecutor`) - Minimal Kubernetes job execution
- **`k8s-job`** (`K8sJobExecutor`) - Full Kubernetes job execution
- **`emulator`** (`EmulatorJobExecutor`) - Emulated job execution

### 3. Run Log Store
**Purpose**: Tracks execution metadata and provides reproducibility
**Entry Point**: `[project.entry-points.'run_log_store']`

Available services:
- **`buffered`** (`BufferRunLogstore`) - In-memory buffered logging
- **`file-system`** (`FileSystemRunLogstore`) - Local file system logging
- **`minio`** (`MinioRunLogStore`) - MinIO object storage logging
- **`chunked-fs`** (`ChunkedFileSystemRunLogStore`) - Thread-safe file system logging
- **`chunked-minio`** (`ChunkedMinioRunLogStore`) - Thread-safe MinIO logging

### 4. Catalog
**Purpose**: Manages data artifacts and inter-task communication
**Entry Point**: `[project.entry-points.'catalog']`

Available services:
- **`do-nothing`** (`DoNothingCatalog`) - No-op catalog
- **`file-system`** (`FileSystemCatalog`) - Local file system storage
- **`s3`** (`S3Catalog`) - AWS S3 storage
- **`minio`** (`MinioCatalog`) - MinIO object storage

### Additional Plugin Systems

**Secrets Management** (`[project.entry-points.'secrets']`):
- **`do-nothing`** (`DoNothingSecretManager`) - No secret management
- **`dotenv`** (`DotEnvSecrets`) - Load secrets from .env files
- **`env-secrets`** (`EnvSecretsManager`) - Environment variable secrets

**Node Types** (`[project.entry-points.'nodes']`):
- **`task`** (`TaskNode`) - Execute individual tasks
- **`fail`** (`FailNode`) - Terminal failure node
- **`success`** (`SuccessNode`) - Terminal success node
- **`parallel`** (`ParallelNode`) - Parallel execution of multiple branches
- **`map`** (`MapNode`) - Iterative execution over data
- **`stub`** (`StubNode`) - Placeholder/no-op node
- **`conditional`** (`ConditionalNode`) - Conditional execution branching

**Task Types** (`[project.entry-points.'tasks']`):
- **`python`** (`PythonTaskType`) - Execute Python functions
- **`shell`** (`ShellTaskType`) - Execute shell commands
- **`notebook`** (`NotebookTaskType`) - Execute Jupyter notebooks
- **`torch`** (`TorchTaskType`) - Execute PyTorch models

### How to Find Service Configurations

Each service is implemented as a **Pydantic model** that defines its configuration options. Here's how to discover the available configuration fields for any service:

#### Step-by-Step Configuration Discovery

1. **Find the service in pyproject.toml**: Look for the entry point mapping
   ```toml
   [project.entry-points.'pipeline_executor']
   "local" = "extensions.pipeline_executor.local:LocalExecutor"
   ```

2. **Locate the implementation file**: Navigate to the specified module path
   - Path: `extensions/pipeline_executor/local.py`
   - Class: `LocalExecutor`

3. **Read the Pydantic model**: The class defines all public configuration fields
   - Look for fields with type annotations
   - Check for `Field()` definitions with defaults
   - Note any validation rules or constraints

4. **Ignore PrivateAttr**: Fields marked with `PrivateAttr` are internal and not user-configurable

5. **Check parent classes**: Many services inherit from base classes that add additional configuration options

#### Configuration Field Types

**Public Configuration Fields** (user-configurable):
```python
class LocalExecutor(GenericPipelineExecutor):
    service_name: str = "local"                    # ← Service identifier
    enable_parallel: bool = Field(default=False)   # ← Configuration option
    object_serialisation: bool = Field(default=True) # ← Configuration option
    overrides: dict[str, Any] = {}                 # ← From parent class
```

**Private Fields** (internal only):
```python
    _is_local: bool = PrivateAttr(default=True)    # ← Internal, ignore
    _context_node: Optional[BaseNode] = PrivateAttr(default=None)  # ← Internal
```

### Service Integration Architecture

The four core concepts work together in the execution flow:

1. **Pipeline Executor** defines the workflow and coordinates execution
2. **Job Executor** runs individual tasks in the specified environment
3. **Run Log Store** captures execution metadata for each step
4. **Catalog** handles data persistence and retrieval between tasks

**Configuration Example**:
```yaml
# Example configuration showing all four concepts
pipeline_executor:
  type: local
  enable_parallel: false

job_executor:
  type: local-container
  docker_image: "python:3.11"

run_log_store:
  type: chunked-fs
  location: "./logs"

catalog:
  type: file-system
  location: "./.catalog"
```

### 🚀 CRITICAL: Custom Plugin Development and Registration

**Runnable automatically discovers and loads custom plugins** via Python entry points and stevedore. Users can extend any of the four core concepts by implementing custom services in their own projects.

#### **How to Create Custom Plugins**

1. **Implement the Interface**: Create a class that inherits from the appropriate base class
2. **Register Entry Point**: Add entry point in your project's `pyproject.toml`
3. **Runnable Auto-Discovery**: Runnable will automatically find and load your plugin at runtime

#### **Custom Plugin Registration Pattern**

**In your project's `pyproject.toml`**:
```toml
# Your custom plugins will be automatically discovered by runnable
[project.entry-points.'pipeline_executor']
"my-custom-executor" = "my_package.executors:MyCustomExecutor"

[project.entry-points.'catalog']
"my-s3-catalog" = "my_package.storage:MyS3Catalog"

[project.entry-points.'run_log_store']
"my-database-logger" = "my_package.logging:MyDatabaseLogger"

[project.entry-points.'secrets']
"my-vault-secrets" = "my_package.secrets:MyVaultSecrets"

[project.entry-points.'job_executor']
"my-k8s-executor" = "my_package.k8s:MyK8sExecutor"

[project.entry-points.'nodes']
"my-custom-node" = "my_package.nodes:MyCustomNode"

[project.entry-points.'tasks']
"my-custom-task" = "my_package.tasks:MyCustomTask"
```

#### **Custom Plugin Implementation Examples**

**Custom Pipeline Executor**:
```python
from extensions.pipeline_executor import GenericPipelineExecutor
from pydantic import Field

class MyCustomExecutor(GenericPipelineExecutor):
    """Custom executor that integrates with your infrastructure."""

    service_name: str = "my-custom-executor"

    # Your custom configuration fields
    api_endpoint: str = Field(default="https://my-service.com")
    timeout: int = Field(default=300)
    custom_setting: bool = Field(default=True)

    def trigger_node_execution(self, node, map_variable=None):
        # Your custom execution logic here
        pass
```

**Custom Catalog Implementation**:
```python
from runnable.datastore import BaseCatalog
from pydantic import Field

class MyS3Catalog(BaseCatalog):
    """Custom S3 catalog with advanced features."""

    service_name: str = "my-s3-catalog"
    service_type: str = "catalog"

    # Your configuration fields
    bucket_name: str = Field(...)
    region: str = Field(default="us-east-1")
    encryption: bool = Field(default=True)

    def get(self, name: str):
        # Your S3 get implementation
        pass

    def put(self, name: str):
        # Your S3 put implementation
        pass
```

#### **Using Custom Plugins in Configuration**

Once registered, your custom plugins are used exactly like built-in services:

```yaml
# Your configuration file
pipeline-executor:
  type: my-custom-executor    # Your custom entry point key
  config:
    api_endpoint: "https://prod-api.mycompany.com"
    timeout: 600
    custom_setting: false

catalog:
  type: my-s3-catalog         # Your custom catalog
  config:
    bucket_name: "my-prod-bucket"
    region: "eu-west-1"
    encryption: true

run-log-store:
  type: my-database-logger    # Your custom logging
  config:
    connection_string: "postgresql://..."
    table_name: "pipeline_logs"
```

#### **Plugin Discovery Process**

1. **Runtime Discovery**: Runnable uses stevedore to scan all installed packages for entry points
2. **Automatic Loading**: Custom plugins are loaded alongside built-in services
3. **Configuration Resolution**: Your custom service types become available in config files
4. **Seamless Integration**: No code changes needed in runnable core

#### **Plugin Development Guidelines**

**Required Implementation**:
- Inherit from appropriate base class (`BasePipelineExecutor`, `BaseCatalog`, etc.)
- Set `service_name` to match your entry point key
- Set `service_type` correctly (`"pipeline_executor"`, `"catalog"`, etc.)
- Implement all abstract methods from base class

**Configuration Fields**:
- Use Pydantic `Field()` for configuration options
- Provide sensible defaults where possible
- Use `PrivateAttr()` for internal state (not user-configurable)
- Add validation using Pydantic validators

**Best Practices**:
- Follow existing naming conventions
- Include comprehensive docstrings
- Add proper error handling
- Test with different configuration combinations
- Consider backward compatibility

#### **Extension Use Cases**

**Common Custom Plugins**:
- **Custom Executors**: Integration with internal orchestrators (Prefect, Airflow, Jenkins)
- **Storage Backends**: Company-specific data lakes, databases, or file systems
- **Secret Management**: Integration with HashiCorp Vault, AWS Secrets Manager, Azure Key Vault
- **Logging Systems**: Custom metrics collection, monitoring dashboards, alerting
- **Task Types**: Domain-specific execution engines (Spark jobs, ML model serving)
- **Node Types**: Custom workflow patterns specific to your business logic

**Enterprise Integration Examples**:
- Slack/Teams notification executors
- Database-backed catalogs for data lineage
- LDAP/SSO secret managers
- Custom compliance and audit loggers
- GPU cluster job executors

**🔑 Key Point**: This extensibility is fundamental to runnable's architecture. Users are not limited to built-in services - they can create fully custom implementations that integrate seamlessly with their existing infrastructure and workflows.

### Configuration File Structure and Examples

Configuration files in `examples/configs/` demonstrate how to select and configure services using the plugin architecture. All configurations follow this pattern:

#### **Standard Configuration Format**
```yaml
# Core service configurations
pipeline-executor:
  type: <service_name>  # Must match entry point key from pyproject.toml
  config:               # Optional: service-specific configuration
    <field_name>: <value>

run-log-store:
  type: <service_name>
  config:
    <field_name>: <value>

catalog:
  type: <service_name>
  config:
    <field_name>: <value>

secrets:
  type: <service_name>
  config:
    <field_name>: <value>
```

#### **Actual Configuration Examples from `examples/configs/`**

**Basic Local Development** (`default.yaml`):
```yaml
pipeline-executor:
  type: local              # Direct local execution

run-log-store:
  type: file-system        # Logs stored as local files

catalog:
  type: file-system        # Data artifacts stored locally

secrets:
  type: env-secrets        # Secrets from environment variables
```

**Containerized Execution** (`local-container.yaml`):
```yaml
pipeline-executor:
  type: local-container
  config:
    docker_image: runnable-m1:latest  # Specify Docker image
    enable_parallel: true             # Enable parallel execution

run-log-store:
  type: chunked-fs         # Thread-safe logging for parallel execution
```

**Parallel Processing** (`parallel_enabled.yaml`):
```yaml
pipeline-executor:
  type: local
  config:
    enable_parallel: true  # Enable parallel execution on local executor

run-log-store:
  type: chunked-fs         # Required for thread-safe parallel logging

catalog:
  type: file-system

secrets:
  type: env-secrets
```

**In-Memory/Testing** (`in-memory.yaml`):
```yaml
pipeline-executor:
  type: local
  config:
    object_serialisation: false  # Disable serialization for performance

run-log-store:
  type: buffered          # In-memory logging (no persistence)

catalog:
  type: do-nothing        # No data storage

secrets:
  type: do-nothing        # No secret management
```

**MinIO Object Storage** (`minio.yaml`):
```yaml
catalog:
  type: minio
  config:
    bucket: runnable/catalog     # MinIO bucket for data artifacts

run-log-store:
  type: minio
  config:
    bucket: runnable/run-logs    # MinIO bucket for execution logs
```

**Argo Workflows** (`argo-config.yaml`):
```yaml
pipeline-executor:
  type: argo
  config:
    pvc_for_runnable: runnable
    defaults:
      image: $docker_image       # Environment variable reference
      resources:
        limits:
          cpu: "1"
          memory: 1Gi
        requests:
          cpu: "0.5"
          memory: 500Mi
    argoWorkflow:
      metadata:
        generateName: "argo-"
        namespace: enterprise-mlops
      spec:
        serviceAccountName: "default-editor"

run-log-store:
  type: chunked-fs
  config:
    log_folder: /mnt/run_log_store
```

**Environment File Secrets** (`dotenv.yaml`):
```yaml
secrets:
  type: dotenv
  config:
    location: examples/secrets.env  # Path to .env file
```

**Retry Execution** (`retry-config.yaml`):
```yaml
pipeline-executor:
  type: retry
  config:
    run_id: parallel-fail    # Retry specific previous run

catalog:
  type: file-system

run-log-store:
  type: file-system
```

**Emulation Mode** (`emulate.yaml`):
```yaml
pipeline-executor:
  type: emulator           # Simulation mode for testing
```

**Chunked File System Logging** (`chunked-fs-run_log.yaml`):
```yaml
run-log-store:
  type: chunked-fs         # Thread-safe file system logging
```

#### **Configuration Mapping Rules**

1. **Service Types**: Must exactly match entry point keys in `pyproject.toml`
2. **Config Fields**: Map directly to Pydantic model fields in service implementation
3. **Defaults**: When `config` section omitted, services use their default values
4. **Environment Variables**: Use `$variable_name` syntax for parameterization

#### **Observed Configuration Patterns**

**Development Combinations**:
- `local` + `file-system` + `env-secrets`
- `local` + `buffered` + `do-nothing` (fastest, no persistence)

**Parallel Execution Requirements**:
- Any executor with `enable_parallel: true`
- Must use `chunked-fs` or `chunked-minio` for run-log-store (thread-safety)

**Container Execution**:
- `local-container` executor
- Specify `docker_image` in config
- Compatible with `chunked-fs` run-log-store

**Object Storage**:
- `minio` type for both catalog and run-log-store
- Requires `bucket` configuration parameter

**Orchestration**:
- `argo` executor with complex Kubernetes configuration
- `retry` executor for re-running previous executions

## Context

The Runnable framework documentation is built with MkDocs and follows established patterns. Your role is to extend and improve the documentation while maintaining the current structure and style.

## Key Principles

- **Maintain existing structure**: Follow the current documentation organization and layout
- **Consistent style**: Match the tone, formatting, and presentation style of existing docs
- **Code from examples**: Always use working code examples from the `examples/` directory
- **Python focus**: Prioritize Python API examples over YAML (YAML is being deprecated)
- **Progressive complexity**: Layer examples from simple to complex following the examples structure
- **Executable patterns**: All code examples must follow the executable patterns used in actual examples

## Documentation Guidelines

### Content Creation
- Use the same formatting patterns as existing documentation
- Reference working examples from `examples/` directory
- Maintain the current section organization and navigation structure
- Follow established naming conventions and terminology
- Keep the same level of detail and explanation depth

### Code Examples
When adding documentation:
1. Use existing examples from `examples/` folder
2. Show `uv run` commands as demonstrated in current docs
3. Follow the same code block formatting and syntax highlighting
4. Include the same types of explanations and context as existing sections
5. Match the balance between overview and detailed examples

#### ⚠️ CRITICAL: Correct Code Patterns
**ALL documentation code examples must follow the executable pattern used in actual examples:**

```python
from runnable import PythonJob, Pipeline, PythonTask

def main():
    # Job or pipeline setup and execution
    job = PythonJob(function=my_function)
    job.execute()
    return job  # Always return the job/pipeline object

if __name__ == "__main__":
    main()
```

**❌ NEVER use these misleading patterns in documentation:**
```python
# WRONG - Don't do this in docs
from runnable import PythonJob
job = PythonJob(function=my_function)
job.execute()
```

**✅ ALWAYS use the correct executable pattern:**
```python
# CORRECT - Always do this
from runnable import PythonJob

def main():
    job = PythonJob(function=my_function)
    job.execute()
    return job

if __name__ == "__main__":
    main()
```

### Structure Consistency
- Follow the same heading hierarchy and organization
- Use identical markdown formatting patterns
- Maintain consistent cross-referencing style
- Keep the same approach to code snippets and callouts
- Preserve the current flow from concept to implementation

## Development Approach

### When extending documentation:
1. **FIRST**: Read `runnable/sdk.py` for actual function/class signatures and parameters
2. Review existing docs to understand the current style and structure
3. Find relevant examples in `examples/` directory that match the existing pattern
4. Run examples with `uv run` to understand behavior
5. Write documentation that seamlessly fits with existing content
6. Use the same mkdocs configuration and formatting

### Areas to Focus On:
- Fill gaps in existing documentation sections
- Add missing examples that follow established patterns
- Improve clarity while maintaining the current voice and style
- Extend existing sections with additional use cases
- Add troubleshooting sections using the same format as existing ones

## Documentation Maintenance

### Quality Assurance
When working on documentation, always verify:

1. **API Accuracy**: All code examples use correct signatures from `runnable/sdk.py`
2. **Pattern Consistency**: All code examples follow the `main()` function pattern
3. **Link Integrity**: All internal links point to existing documentation sections
4. **Example Verification**: Referenced examples actually exist in the `examples/` directory
5. **Executable Code**: All code snippets can actually be run by users
6. **Navigation Alignment**: New content fits into the existing mkdocs navigation structure

## 🔑 Critical Source Files

### Primary API Reference
- **`runnable/sdk.py`** - THE source of truth for all class signatures, parameters, and methods
  - Contains `PythonJob`, `Pipeline`, `PythonTask`, and all core APIs
  - Always check actual signatures before writing sample code
  - Don't guess parameter names - read the source!

### Complete SDK API Reference

The exposed SDK concepts from `runnable/__init__.py` and their signatures from `runnable/sdk.py`:

#### **Core Execution Classes**

**Pipeline** - Main orchestration class:
```python
class Pipeline(BaseModel):
    steps: List[StepType]  # List of tasks/nodes to execute
    name: str = ""  # Optional pipeline name
    description: str = ""  # Optional description

    def execute(
        self,
        configuration_file: str = "",
        run_id: str = "",
        tag: str = "",
        parameters_file: str = "",
        log_level: str = defaults.LOG_LEVEL,
    )
```

**Job Classes** (for single task execution):
```python
class PythonJob(BaseJob):
    function: Callable = Field()  # Required: Python function to execute
    catalog: Optional[Catalog] = None
    returns: List[Union[str, TaskReturns]] = []
    secrets: List[str] = []

    def execute(
        self,
        configuration_file: str = "",
        job_id: str = "",
        tag: str = "",
        parameters_file: str = "",
        log_level: str = defaults.LOG_LEVEL,
    )

class NotebookJob(BaseJob):
    notebook: str = Field()  # Required: Path to notebook file
    optional_ploomber_args: Optional[Dict[str, Any]] = None
    # ... inherits catalog, returns, secrets from BaseJob

class ShellJob(BaseJob):
    command: str = Field()  # Required: Shell command to execute
    # ... inherits catalog, returns, secrets from BaseJob
```

#### **Task Classes** (for pipeline steps):

**PythonTask**:
```python
class PythonTask(BaseTask):
    name: str  # Required: Unique task name
    function: Callable = Field(exclude=True)  # Required: Python function

    # Flow control
    next_node: str = ""
    terminate_with_success: bool = False
    terminate_with_failure: bool = False
    on_failure: Optional[Pipeline] = None

    # Data handling
    catalog: Optional[Catalog] = None
    returns: List[Union[str, TaskReturns]] = []
    secrets: List[str] = []
    overrides: Dict[str, Any] = {}
```

**NotebookTask**:
```python
class NotebookTask(BaseTask):
    name: str  # Required: Unique task name
    notebook: str = Field()  # Required: Path to notebook
    optional_ploomber_args: Optional[Dict[str, Any]] = None
    # ... inherits all BaseTask fields
```

**ShellTask**:
```python
class ShellTask(BaseTask):
    name: str  # Required: Unique task name
    command: str = Field()  # Required: Shell command
    # ... inherits all BaseTask fields
```

#### **Node Types** (for complex workflows):

**Parallel** - Execute multiple branches concurrently:
```python
class Parallel(BaseTraversal):
    name: str  # Required: Unique node name
    branches: Dict[str, Pipeline]  # Required: Named branches to execute
```

**Map** - Iterate over data:
```python
class Map(BaseTraversal):
    name: str  # Required: Unique node name
    branch: Pipeline  # Required: Pipeline to execute per item
    iterate_on: str  # Required: Parameter name to iterate over
    iterate_as: str  # Required: Variable name for each iteration
    reducer: Optional[str] = None  # Optional: Function to reduce results
```

**Conditional** - Branching logic:
```python
class Conditional(BaseTraversal):
    name: str  # Required: Unique node name
    branches: Dict[str, Pipeline]  # Required: Named conditional branches
    parameter: str  # Required: Parameter name for condition (must be alphanumeric)
```

**Stub** - Pass-through node:
```python
class Stub(BaseTraversal):
    name: str  # Required: Unique node name
    catalog: Optional[Catalog] = None
    # Accepts arbitrary extra arguments
```

#### **Data Management Classes**

**Catalog** - File synchronization:
```python
class Catalog(BaseModel):
    get: List[str] = []  # Glob patterns to retrieve from catalog
    put: List[str] = []  # Glob patterns to store in catalog
    store_copy: bool = True  # Whether to copy files or just track hashes
```

#### **Return Type Helpers**

```python
def pickled(name: str) -> TaskReturns:
    """Mark return value as pickled Python object"""

def metric(name: str) -> TaskReturns:
    """Mark return value as a metric for tracking"""
```

#### **Terminal Nodes**

```python
class Success(BaseModel):
    name: str = "success"  # Terminal success node

class Fail(BaseModel):
    name: str = "fail"  # Terminal failure node
```

### Key Parameter Notes:
- **Required Fields**: `name` for all tasks/nodes, `function` for Python tasks, `notebook` for notebook tasks, `command` for shell tasks
- **Flow Control**: Use `>>` operator to chain tasks, or set `next_node` explicitly
- **Returns**: Can be strings (JSON), `pickled()` objects, or `metric()` values
- **Catalog**: Both Jobs and Pipeline tasks support file management via `Catalog(put=[...])`
- **Configuration**: All classes support `configuration_file`, `parameters_file`, `tag`, and `log_level` in execute methods

### Common Issues to Avoid
- ❌ Code examples that don't follow the established executable pattern
- ❌ Links to non-existent files (check with mkdocs serve for warnings)
- ❌ References to deprecated YAML patterns instead of Python API
- ❌ Inconsistent heading styles or markdown formatting
- ❌ Examples that haven't been tested with `uv run`

### Testing Documentation Changes
1. Always run `uv run mkdocs serve` to check for warnings
2. Test referenced examples with `uv run examples/path/to/example.py`
3. Verify all internal links work correctly
4. Ensure code examples are copy-pasteable and executable

### Lessons Learned from Recent Fixes
Based on recent documentation improvements, pay special attention to:

1. **Code Pattern Consistency**: The biggest issue was inline code examples that didn't match the actual executable patterns in the `examples/` directory
2. **Link Maintenance**: Several broken links existed due to:
   - References to non-existent `concepts/task.md` (should point to `concepts/building-blocks/task-types.md`)
   - Missing concept files that were referenced but never created
   - Self-referencing links that made no sense
3. **Unused Files**: Remove files that aren't in the mkdocs navigation structure
4. **Anchor Accuracy**: Ensure links to page sections use the correct anchor format
5. **README Consistency**: Keep README.md examples consistent with documentation patterns

## Key Reminders
- Preserve the current documentation structure and organization
- Match the existing writing style and tone
- Use working code from examples directory
- Follow established formatting and presentation patterns
- Maintain consistency with existing cross-references and navigation
- **CRITICAL**: All code examples must use the `main()` function pattern

## Recent Documentation Updates (2024)

### Jobs Documentation Structure
- **NEW**: Jobs now have dedicated documentation with progressive complexity structure:
  - `jobs/first-job.md` - Basic execution and concepts
  - `jobs/working-with-data.md` - Return values and data storage
  - `jobs/parameters.md` - Configuration without code changes
  - `jobs/file-storage.md` - Catalog system for file management
  - `jobs/job-types.md` - Shell, Notebook, and Python Jobs
- **Navigation**: "Core Concepts" renamed to "Pipelines" for clarity
- **Progressive Learning**: Each Jobs page builds on previous concepts

### Catalog System Understanding
- **File Storage**: Both Jobs and Pipelines support `Catalog(put=[...])` for file storage
- **No-Copy Mode**: `store_copy=False` captures MD5 hash without copying files
  - Use for large datasets (GB+ files) where copying is expensive
  - Files remain in original location, tracked by hash for integrity
  - Saves disk space and improves performance
- **Glob Patterns**: Currently supported (`"*.csv"`, `"plots/*.png"`, `"data/**/*.json"`)
- **Return Types**: `pickled()`, `metric()`, and JSON serialization for different data types

### Parameter System
- **Three-Layer Precedence**: Jobs inherit same parameter system as Pipelines
  1. `RUNNABLE_PRM_key="value"` (highest priority)
  2. `RUNNABLE_PARAMETERS_FILE="config.yaml"`
  3. `job.execute(parameters_file="config.yaml")` (lowest priority)
- **Environment Variables**: Always override YAML values

### Documentation Formatting
- **Markdown Lists**: Always add blank line between headings and lists
- **Emoji Usage**: Conservative use only for key section headers (📦, ⚙️, 📁, 🔍)
- **Code Patterns**: Only show patterns that exist in actual examples directory
