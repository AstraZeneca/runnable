# Building Custom Run Log Stores

Store execution metadata and logs in any database or cloud storage system by creating custom run log stores that integrate with Runnable's plugin architecture.

!!! tip "Real-World Examples"

    The `extensions/run_log_store/` directory contains working implementations for file system, MinIO, and chunked storage that demonstrate these patterns in production code.

    The below is a rough guideline for database and cloud storage integrations.

## Run Log Storage Workflow

Custom run log stores handle the persistent storage of execution metadata, providing durability and queryability for pipeline runs:

### Core Integration Pattern

```python
from runnable.datastore import BaseRunLogStore, RunLog, JobLog, StepLog, JsonParameter
from typing import Dict, Any, List, Optional

class DatabaseRunLogStore(BaseRunLogStore):
    service_name: str = "database"

    def create_run_log(self, run_id: str, **kwargs) -> RunLog:
        """Create a new run log entry - called at pipeline start"""
        pass

    def get_run_log_by_id(self, run_id: str, full: bool = True) -> RunLog:
        """Retrieve run log by ID - most frequently called method"""
        pass

    def put_run_log(self, run_log: RunLog):
        """Store/update complete run log - called at pipeline completion"""
        pass

    def get_run_logs(self, run_ids: List[str] = None, **kwargs) -> List[RunLog]:
        """Query multiple run logs with filters - for analysis and debugging"""
        pass

    def set_parameters(self, run_id: str, parameters: Dict[str, JsonParameter]):
        """Store pipeline parameters - called early in execution"""
        pass

    def set_run_config(self, run_id: str, run_config: Dict[str, Any]):
        """Store pipeline configuration - called during setup"""
        pass

    def create_step_log(self, run_id: str, step_log: StepLog):
        """Create step log entry - called for each pipeline step"""
        pass

    def create_job_log(self) -> JobLog:
        """Create job log entry - called for job execution"""
        pass

    def add_job_log(self, run_id: str, job_log: JobLog):
        """Add job log to run - called after job completion"""
        pass
```

**The workflow ensures**:

- **Metadata persistence**: Execution details survive beyond process lifetime
- **Query capability**: Run logs can be searched and analyzed
- **Audit trail**: Complete execution history for compliance and debugging
- **Parallel safety**: Multiple concurrent executions don't conflict

## Implementation Template

Here's a stubbed implementation template for integrating with databases or cloud storage:

```python
from typing import Dict, Any, List, Optional
from pydantic import Field
import json

from runnable.datastore import BaseRunLogStore, RunLog, JobLog, StepLog, JsonParameter
from runnable import defaults, exceptions

class CloudDatabaseRunLogStore(BaseRunLogStore):
    """Store run logs in cloud database or storage systems"""

    service_name: str = "cloud-database"

    # Configuration fields - these map to YAML config
    connection_string: str = Field(..., description="Database connection string or storage endpoint")
    table_name: str = Field(default="runnable_logs", description="Table/collection name for run logs")
    retention_days: int = Field(default=90, description="How long to keep run logs")
    enable_compression: bool = Field(default=True, description="Compress large run logs")

    def create_run_log(
        self,
        run_id: str,
        dag_hash: str = "",
        tag: str = "",
        status: str = defaults.CREATED
    ) -> RunLog:
        """Create new run log entry in storage"""

        # STEP 1: Create RunLog object
        run_log = RunLog(
            run_id=run_id,
            dag_hash=dag_hash,
            tag=tag,
            status=status
        )

        # STEP 2: Store in your database/storage system
        self._store_run_log_metadata(run_log)
        # TODO: Insert initial run log record into your database/storage

        return run_log

    def get_run_log_by_id(self, run_id: str, full: bool = True) -> RunLog:
        """Retrieve run log from storage"""

        # STEP 1: Query your storage system
        raw_data = self._fetch_run_log_data(run_id)
        # TODO: Query your database/storage for run_id

        if not raw_data:
            raise exceptions.RunLogNotFoundError(f"Run log {run_id} not found")

        # STEP 2: Convert to RunLog object
        run_log = self._deserialize_run_log(raw_data, full=full)

        return run_log

    def put_run_log(self, run_log: RunLog):
        """Store/update complete run log"""

        # STEP 1: Serialize run log data
        serialized_data = self._serialize_run_log(run_log)

        # STEP 2: Store in your database/storage system
        self._update_run_log_storage(run_log.run_id, serialized_data)
        # TODO: Update/insert complete run log in your database/storage

    def get_run_logs(
        self,
        run_ids: List[str] = None,
        tag: str = "",
        status: str = "",
        **kwargs
    ) -> List[RunLog]:
        """Query multiple run logs with filters"""

        # STEP 1: Build query based on filters
        query_conditions = self._build_query_conditions(run_ids, tag, status, **kwargs)

        # STEP 2: Execute query against your storage
        raw_results = self._query_run_logs(query_conditions)
        # TODO: Execute filtered query against your database/storage

        # STEP 3: Convert results to RunLog objects
        run_logs = [self._deserialize_run_log(data) for data in raw_results]

        return run_logs

    def _store_run_log_metadata(self, run_log: RunLog):
        """Store initial run log in your database/storage"""
        # TODO: Implement storage-specific logic
        # Examples:
        # - SQL: INSERT INTO runnable_logs (run_id, status, created_at) VALUES (...)
        # - NoSQL: collection.insert_one({"run_id": run_log.run_id, ...})
        # - Cloud Storage: upload_object(f"runs/{run_log.run_id}/metadata.json", ...)
        pass

    def _fetch_run_log_data(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Fetch run log data from your storage system"""
        # TODO: Implement retrieval logic
        # Examples:
        # - SQL: SELECT * FROM runnable_logs WHERE run_id = ?
        # - NoSQL: collection.find_one({"run_id": run_id})
        # - Cloud Storage: download_object(f"runs/{run_id}/log.json")

        return {}  # Replace with actual data

    def _update_run_log_storage(self, run_id: str, data: Dict[str, Any]):
        """Update complete run log in storage"""
        # TODO: Implement update/upsert logic
        # Handle large run logs based on your storage capabilities

        if self.enable_compression and len(json.dumps(data)) > 1000000:  # 1MB threshold
            data = self._compress_run_log_data(data)  # TODO: Implement compression

        # Store the data in your system
        pass

    def _serialize_run_log(self, run_log: RunLog) -> Dict[str, Any]:
        """Convert RunLog to storage format"""
        # Use RunLog's built-in serialization
        return run_log.model_dump()

    def _deserialize_run_log(self, data: Dict[str, Any], full: bool = True) -> RunLog:
        """Convert storage data to RunLog object"""
        # Handle decompression if needed
        if self.enable_compression and 'compressed' in data:
            data = self._decompress_run_log_data(data)  # TODO: Implement decompression

        # Create RunLog from stored data
        return RunLog(**data)

    def _build_query_conditions(self, run_ids: List[str], tag: str, status: str, **kwargs):
        """Build database query conditions"""
        # TODO: Translate filters to your storage query format
        conditions = {}

        if run_ids:
            conditions['run_id'] = {'$in': run_ids}  # MongoDB style - adapt to your DB
        if tag:
            conditions['tag'] = tag
        if status:
            conditions['status'] = status

        return conditions

    def _query_run_logs(self, conditions: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute query against your storage system"""
        # TODO: Execute filtered query
        # Examples:
        # - SQL: SELECT * FROM runnable_logs WHERE conditions
        # - NoSQL: collection.find(conditions)
        # - Cloud Storage: list_objects_with_filters(conditions)

        return []  # Replace with actual results

    def get_summary(self) -> Dict[str, Any]:
        """Return storage system summary"""
        return {
            "Type": self.service_name,
            "Connection": self.connection_string,  # May want to mask sensitive parts
            "Table": self.table_name,
            "Retention": f"{self.retention_days} days",
            "Compression": self.enable_compression
        }
```

**Key Integration Points**:

- **`create_run_log()`**: Called at pipeline start - store initial metadata
- **`get_run_log_by_id()`**: Most frequently called - optimize for fast retrieval
- **`put_run_log()`**: Called at pipeline completion - store full execution results
- **`get_run_logs()`**: For querying and analysis - support filtering and pagination

## Configuration & Plugin Registration

### YAML to Pydantic Field Mapping

Understanding how YAML configuration maps to your run log store class fields is crucial:

**Your Pydantic Class**:
```python
class CloudDatabaseRunLogStore(BaseRunLogStore):
    service_name: str = "cloud-database"
    service_type: str = "run_log_store"  # Always set to "run_log_store"

    # Required fields (must be provided in YAML)
    connection_string: str = Field(..., description="Database connection string or storage endpoint")
    table_name: str = Field(..., description="Table/collection name for run logs")

    # Optional fields with defaults
    retention_days: int = Field(default=90, description="How long to keep run logs")
    enable_compression: bool = Field(default=True, description="Compress large run logs")
    max_connections: int = Field(default=20, description="Database connection pool size")
    timeout_seconds: int = Field(default=30, description="Query timeout")

    # Thread safety support
    supports_parallel_writes: bool = Field(default=False, description="Enable for parallel execution")
```

**Maps to YAML Configuration**:
```yaml title="cloud-database-config.yaml"
run-log-store:
  type: cloud-database              # → matches service_name in your class
  config:
    # Required fields
    connection_string: "postgresql://user:pass@host:5432/db"  # → self.connection_string
    table_name: "pipeline_execution_logs"     # → self.table_name

    # Optional fields (override defaults)
    retention_days: 180             # → self.retention_days (overrides default 90)
    enable_compression: false       # → self.enable_compression (overrides default True)
    max_connections: 50             # → self.max_connections (overrides default 20)
    timeout_seconds: 60             # → self.timeout_seconds (overrides default 30)
    supports_parallel_writes: true # → self.supports_parallel_writes (enables parallel execution)
```

**In your code, access config as class attributes**:
```python
def create_run_log(self, run_id: str, **kwargs) -> RunLog:
    # Access your configuration fields directly
    connection = self._get_connection()  # Uses self.connection_string

    query = f"INSERT INTO {self.table_name} (run_id, status) VALUES (%s, %s)"
    #                     ↑ From YAML config.table_name

    with connection.cursor() as cursor:
        cursor.execute(query, (run_id, "CREATED"))

    return RunLog(run_id=run_id)

def _get_connection(self):
    """Create database connection using configuration"""
    return psycopg2.connect(
        self.connection_string,     # From YAML config.connection_string
        connect_timeout=self.timeout_seconds  # From YAML config.timeout_seconds
    )

def put_run_log(self, run_log: RunLog):
    """Store run log, optionally compressing large data"""
    data = run_log.model_dump()

    if self.enable_compression and len(json.dumps(data)) > 1000000:  # 1MB
        data = self._compress_data(data)  # Compression enabled via YAML

    # Store in table specified by YAML config
    query = f"UPDATE {self.table_name} SET data = %s WHERE run_id = %s"
    # ... store data
```

### Configuration Validation

**Pydantic automatically validates your config**:

- **Required fields**: Pipeline fails with clear error if missing from YAML
- **Type checking**: `retention_days: "invalid"` raises validation error before execution
- **Defaults applied**: Optional fields use defaults when not specified in YAML
- **Custom validation**: Add Pydantic validators for complex field validation

**Example validation error**:
```bash
ValidationError: 1 validation error for CloudDatabaseRunLogStore
connection_string
  field required (type=value_error.missing)
```

### Plugin Registration

**1. Register via entry point in `pyproject.toml`:**
```toml title="pyproject.toml"
[project.entry-points.'run_log_store']
"cloud-database" = "my_package.stores:CloudDatabaseRunLogStore"
```

**2. Runnable discovers your store automatically**

### Usage Pattern

**Pipelines use your run log store transparently**:
```python
from runnable import Pipeline, PythonTask
from examples.common.functions import hello

def main():
    pipeline = Pipeline(steps=[
        PythonTask(function=hello, name="task1"),
        PythonTask(function=hello, name="task2")
    ])
    pipeline.execute(configuration_file="cloud-database-config.yaml")
    return pipeline

if __name__ == "__main__":
    main()
```

## Storage System Considerations

### Database Integration Patterns

**Relational Databases (PostgreSQL, MySQL)**:
- Store run metadata in structured tables
- Use JSON columns for flexible step log data
- Index on run_id, status, tag, created_at for fast queries
- Consider partitioning for high-volume deployments

**NoSQL Databases (MongoDB, DynamoDB)**:
- Store complete run logs as documents
- Use compound indexes for query patterns
- Handle large documents with compression or splitting
- Leverage native JSON querying capabilities

**Cloud Storage (S3, Azure Blob, GCS)**:
- Store run logs as individual files
- Use object metadata for filtering
- Implement listing and querying via object keys
- Consider data lakes for analytics integration

### Performance Optimization

**For High-Volume Deployments**:
```python
class OptimizedRunLogStore(BaseRunLogStore):
    # Add connection pooling
    max_connections: int = Field(default=20)

    # Add caching
    cache_ttl_seconds: int = Field(default=300)

    # Add batching for writes
    batch_size: int = Field(default=100)

    # Add async operations
    async_writes: bool = Field(default=True)
```

### Parallel Execution Support

**Thread-Safe Implementation**:
```python
class ThreadSafeRunLogStore(BaseRunLogStore):
    supports_parallel_writes: bool = True  # Enable parallel pipeline execution

    # Use appropriate locking/coordination for your storage system
    def put_run_log(self, run_log: RunLog):
        # Implement thread-safe storage updates
        # Use database transactions, file locking, etc.
```

## Testing Your Custom Run Log Store

### Development Testing

```python
class CloudDatabaseRunLogStore(BaseRunLogStore):
    mock: bool = Field(default=False, description="Enable mock mode for testing")

    def _store_run_log_metadata(self, run_log: RunLog):
        if self.mock:
            # Store in memory or local files for testing
            self._mock_storage[run_log.run_id] = run_log.model_dump()
        else:
            # Real database/storage integration
            self._execute_database_insert(run_log)
```

### Test Configuration

```yaml title="mock-config.yaml"
run-log-store:
  type: cloud-database
  config:
    connection_string: "sqlite:///:memory:"  # In-memory for testing
    table_name: "test_runs"
    mock: true  # Skip real database calls
```

### Integration Testing

**Test with pipeline execution**:
```python
from runnable import Pipeline, PythonTask

def test_function():
    print("Testing custom run log store!")
    return {"test": "completed"}

def main():
    pipeline = Pipeline(steps=[
        PythonTask(function=test_function, name="test_task")
    ])
    pipeline.execute(configuration_file="mock-config.yaml")
    return pipeline

if __name__ == "__main__":
    main()
```

## Development Workflow

### 1. Start with Stubbed Template

- Copy the `CloudDatabaseRunLogStore` template above
- Replace database-specific fields with your storage system's configuration
- Keep all TODO comments initially

### 2. Test Storage Integration

- Enable mock mode to test runnable integration
- Implement basic create/get/put operations
- Verify run logs are stored and retrieved correctly

### 3. Add Query and Analytics Features

- Implement `get_run_logs()` with filtering
- Add indexing for performance
- Support pagination for large result sets

### 4. Production Hardening

- Add connection pooling and retry logic
- Implement proper error handling and logging
- Add monitoring and health checks
- Consider backup and disaster recovery

## Existing Implementation Examples

Before building your custom run log store, study the existing implementations in `extensions/run_log_store/`:

### Simple Pattern Examples
- **`file_system.py`** - Basic file-based storage pattern using `AnyPathRunLogStore`
- **`minio.py`** - Object storage integration with cloud APIs

### Advanced Pattern Examples
- **`chunked_fs.py`** - Thread-safe file system storage for parallel execution
- **`chunked_minio.py`** - Thread-safe cloud object storage
- **`generic_chunked.py`** - Base class for thread-safe implementations

### Key Patterns to Learn From

**Thread Safety for Parallel Execution**:
```python
# From chunked_fs.py - see how it handles concurrent writes
supports_parallel_writes: bool = True

def put_run_log(self, run_log: RunLog):
    # Thread-safe file operations with locking
```

**Storage Abstraction**:
```python
# From any_path.py - see the abstraction pattern
@abstractmethod
def write_to_path(self, run_log: RunLog): ...

@abstractmethod
def read_from_path(self, run_id: str) -> RunLog: ...
```

**Cloud Storage Integration**:
```python
# From minio.py - see how it handles object storage APIs
def put_run_log(self, run_log: RunLog):
    # Object storage with proper error handling
```

!!! tip "Learn from Production Code"

    These implementations show real-world patterns for:

    - **Error handling** and retry logic
    - **Serialization** and data format decisions
    - **Performance optimization** for different storage types
    - **Configuration patterns** and validation
    - **Thread safety** and parallel execution support

## Need Help?

**Custom run log stores involve complex data persistence and query patterns** that require understanding both runnable's execution metadata model and your target storage system's capabilities.

!!! question "Get Support"

    **We're here to help you succeed!** Building custom run log stores involves detailed knowledge of:

    - RunLog and StepLog data structures and serialization
    - Query patterns and performance optimization
    - Parallel execution safety and data consistency
    - Storage system integration and error handling

    **Don't hesitate to reach out:**

    - 📧 **Contact the team** for architecture guidance and data model support
    - 🤝 **Collaboration opportunities** - we're interested in supporting database and analytics integrations
    - 📖 **Documentation feedback** - help us improve these guides based on your implementation experience

    **Better together**: Data persistence integrations benefit from collaboration between storage experts (you) and runnable data model experts (us).

!!! warning "Complex Integration"

    **These are sophisticated integrations** that involve:

    - Understanding runnable's internal data models and serialization
    - Designing efficient query patterns for your storage system
    - Handling concurrent access and data consistency
    - Managing large data volumes and performance optimization

    **Success is much more likely with collaboration.** The existing implementations took significant effort to get right - leverage our experience to avoid common pitfalls.

Your success with custom run log stores helps the entire runnable community!

---
