# Run Log Store Configuration

**Run logs make reproducibility simple** - they capture everything needed to understand, debug, and recreate your pipeline executions.

## Why Run Logs Matter for Reproducibility

!!! success "Complete Execution History"

    **Every run is fully documented**: Run logs capture the complete context of your pipeline execution

    - 📊 **Parameters and inputs**: What data and settings were used
    - 🔄 **Execution timeline**: When each step ran and how long it took
    - 💾 **Data lineage**: Which data artifacts were created and consumed
    - 📝 **Code snapshots**: Git commits and code versions used
    - ❌ **Failure details**: Exact error messages and stack traces
    - 🏷️ **Environment metadata**: Configuration and infrastructure used

!!! tip "Reproducibility Made Easy"

    Run logs enable you to:

    - **Debug production failures** by recreating exact conditions locally
    - **Compare experiments** across different parameter sets or code versions
    - **Audit model training** with complete training history and data lineage
    - **Resume failed pipelines** from the exact point of failure

## Available Run Log Stores

| Store Type | Environment | Best For |
|------------|-------------|----------|
| `buffered` | In-memory only | Quick testing and development |
| `file-system` | **Any environment with mounted log_folder** | Sequential execution, simple setup |
| `chunked-fs` | **Any environment with mounted log_folder** | **Parallel execution, universal choice** |
| `minio` / `chunked-minio` | Object storage | Distributed systems without shared filesystem |


## buffered

Stores run logs in-memory only. No persistence - data is lost when execution completes.

!!! warning "In-Memory Only"

    - **No persistence**: Run logs are lost after execution
    - **Testing only**: Not suitable for production or reproducibility
    - **No parallel support**: Race conditions occur with concurrent execution

**Use case**: Quick testing and debugging during development.

### Configuration

```yaml
run-log-store:
  type: buffered
```

## file-system

Stores run logs as single JSON files in the filesystem - simple and reliable for sequential execution.

!!! success "Works Everywhere with Mounted Storage"

    **Runs in any environment where log_folder is accessible**

    - 💾 **Persistent storage**: Run logs saved to mounted filesystem
    - 📁 **Simple structure**: One JSON file per pipeline run
    - 🔍 **Easy debugging**: Human-readable JSON format
    - 🏠 **Local development**: Direct filesystem access
    - 🐳 **Containers**: Works with volume mounts
    - ☸️ **Kubernetes**: Works with persistent volumes

!!! warning "Sequential Only"

    **Not suitable for parallel execution** - use `chunked-fs` for parallel workflows

### Configuration

```yaml
run-log-store:
  type: file-system
  config:
    log_folder: ".run_log_store"  # Optional: defaults to ".run_log_store"
```

### Example

=== "job.py"

    ```python
    from runnable import PythonJob
    from examples.common.functions import hello

    def main():
        job = PythonJob(function=hello)
        job.execute()
        return job

    if __name__ == "__main__":
        main()
    ```

=== "config.yaml"

    ```yaml
    run-log-store:
      type: file-system
      config:
        log_folder: ".run_log_store"
    ```

**Run with file-system logging:**
```bash
RUNNABLE_CONFIGURATION_FILE=config.yaml uv run job.py
```

**Result**: Run log stored as `.run_log_store/{run_id}.json` with complete execution metadata for reproducibility.


## chunked-fs

**Thread-safe run log store** - works everywhere with parallel execution support. The recommended choice for most use cases.

!!! success "Works Everywhere with Mounted Storage"

    **Runs in any environment where log_folder is accessible**

    - ✅ **Thread-safe**: Supports parallel execution without race conditions
    - 🏠 **Local development**: Direct filesystem access
    - 🐳 **Containers**: Works with volume mounts (Docker, local-container executor)
    - ☸️ **Kubernetes**: Works with persistent volumes (Argo, k8s-job executor)
    - ⚡ **Parallel execution**: Enable `enable_parallel: true` safely
    - 💾 **Persistent**: Full reproducibility with detailed execution history

!!! tip "Recommended Default"

    **Use `chunked-fs` unless you have specific requirements** - it provides parallel safety and works in all execution environments where the log_folder can be mounted.

### Configuration

```yaml
run-log-store:
  type: chunked-fs
  config:
    log_folder: ".run_log_store"  # Optional: defaults to ".run_log_store"
```

### Example

=== "pipeline.py"

    ```python
    from runnable import Pipeline, PythonTask, Parallel
    from examples.common.functions import hello

    def main():
        # Parallel execution safe with chunked-fs
        parallel_node = Parallel(
            name="parallel_tasks",
            branches={
                "task_a": PythonTask(function=hello, name="hello_a"),
                "task_b": PythonTask(function=hello, name="hello_b")
            }
        )

        pipeline = Pipeline(steps=[parallel_node])
        pipeline.execute()
        return pipeline

    if __name__ == "__main__":
        main()
    ```

=== "config.yaml"

    ```yaml
    run-log-store:
      type: chunked-fs

    pipeline-executor:
      type: local
      config:
        enable_parallel: true  # Safe with chunked-fs
    ```

**Run with chunked-fs logging:**
```bash
RUNNABLE_CONFIGURATION_FILE=config.yaml uv run pipeline.py
```

**Result**: Run logs stored as separate files in `.run_log_store/{run_id}/` directory:

- `RunLog.json` - Pipeline metadata and configuration
- `StepLog-{step}-{timestamp}.json` - Individual step execution details

This chunked structure enables thread-safe parallel writes while maintaining complete execution history for reproducibility.

## Object Storage (minio / chunked-minio)

For distributed systems and cloud deployments, use object storage-based run log stores:

### minio

```yaml
run-log-store:
  type: minio
  config:
    endpoint: "https://s3.amazonaws.com"
    access_key: "your-access-key"
    secret_key: "your-secret-key"
    bucket_name: "runnable-logs"
```

### chunked-minio (Recommended)

```yaml
run-log-store:
  type: chunked-minio
  config:
    endpoint: "https://s3.amazonaws.com"
    access_key: "your-access-key"
    secret_key: "your-secret-key"
    bucket_name: "runnable-logs"
```

!!! tip "Cloud Deployment"

    Use `chunked-minio` for distributed systems - it provides the same parallel execution safety as `chunked-fs` but with cloud storage scalability.

## Choosing the Right Run Log Store

!!! success "Decision Guide"

    **For most users**: Use `chunked-fs` - works in any environment with mounted storage and supports parallel execution

    **For development/testing**: Use `buffered` for quick iterations where persistence isn't needed

    **Sequential workflows**: Use `file-system` - works in any environment with mounted storage but only for sequential execution

    **Distributed systems without shared filesystem**: Use `chunked-minio` when execution environments can't mount a shared log_folder

!!! info "Filesystem vs Object Storage"

    **Filesystem stores** (`file-system`, `chunked-fs`): Work in any execution environment where the `log_folder` can be mounted

    - ✅ Local development (direct filesystem access)
    - ✅ Docker containers (volume mounts)
    - ✅ Kubernetes (persistent volumes)
    - ✅ Any containerized environment with volume mounting

    **Object storage** (`minio`, `chunked-minio`): Use when shared filesystem mounting isn't available

**Remember**: Run logs are your key to reproducibility - they capture everything needed to understand, debug, and recreate your pipeline executions.

## Custom Run Log Stores

**Need to integrate with your existing logging infrastructure?** Build custom run log stores that send execution data anywhere using Runnable's extensible architecture.

!!! success "Enterprise Integration"

    **Integrate with your existing systems**: Never be limited by built-in storage options

    - 📊 **Enterprise logging**: Send to Splunk, ELK Stack, Datadog, New Relic
    - 🏢 **Corporate databases**: Store in existing data warehouses, time-series databases
    - 🔐 **Compliance systems**: Meet audit and governance requirements
    - 🌐 **Multi-region storage**: Distribute logs across geographic regions

### Building Custom Run Log Stores

Learn how to create production-ready custom run log stores:

**[📖 Custom Run Log Stores Development Guide](custom-run-log-stores.md)**

The guide provides:

- **Complete stubbed implementation** for database and cloud storage integration
- **YAML to Pydantic configuration mapping** with validation
- **Storage system patterns** for SQL, NoSQL, and cloud storage
- **Performance optimization** for high-volume deployments

!!! example "Quick Example"

    Create a custom run log store in just 3 steps:

    1. **Implement key methods** by extending `BaseRunLogStore`
    2. **Register via entry point** in your `pyproject.toml`
    3. **Configure via YAML** for seamless integration

    ```python
    from runnable.datastore import BaseRunLogStore

    class MyDatabaseRunLogStore(BaseRunLogStore):
        service_name: str = "my-database"

        def create_run_log(self, run_id: str, **kwargs):
            # Your database integration here
            pass
    ```

**Ready to build?** See the [development guide](custom-run-log-stores.md) for complete implementation patterns.

---
