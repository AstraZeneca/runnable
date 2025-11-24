# Data Catalog Configuration

**Catalogs manage data flow between pipeline steps** - they store and retrieve data artifacts, ensuring your workflows have access to the data they need.

## Why Data Catalogs Matter

!!! success "Seamless Data Flow"

    **Automatic data management**: Focus on your logic, not data plumbing

    - 📊 **Cross-step data sharing**: Pass data between pipeline steps automatically
    - 💾 **Artifact versioning**: Each pipeline run gets isolated data storage
    - 🔍 **Data lineage**: Track which data was used by which steps
    - 🎯 **Type safety**: Automatic serialization/deserialization with type hints
    - ♻️ **Reproducibility**: Exact data artifacts preserved for every run

## Available Catalog Stores

| Store Type | Environment | Best For |
|------------|-------------|----------|
| `do-nothing` | Any | Testing without data persistence |
| `file-system` | **Any environment with mounted storage** | Local development and single-machine production |
| `s3` / `minio` | Object storage | Distributed systems and cloud deployments |


## do-nothing

No data persistence - useful for testing pipeline logic without data storage overhead.

!!! warning "No Data Persistence"

    - **Testing only**: Data artifacts are not stored or retrieved
    - **Pipeline validation**: Verify workflow logic without data management
    - **Fast execution**: No I/O overhead for development iteration

### Configuration

```yaml
catalog:
  type: do-nothing
```

## file-system

Stores data artifacts in the local filesystem - reliable and simple for most use cases.

!!! success "Works Everywhere with Mounted Storage"

    **Runs in any environment where catalog_location is accessible**

    - 💾 **Persistent storage**: Data artifacts saved to mounted filesystem
    - 📁 **Organized structure**: Each run gets isolated directory by run_id
    - 🏠 **Local development**: Direct filesystem access
    - 🐳 **Containers**: Works with volume mounts
    - ☸️ **Kubernetes**: Works with persistent volumes

### Configuration

```yaml
catalog:
  type: file-system
  config:
    catalog_location: ".catalog"  # Optional: defaults to ".catalog"
```

### Example

=== "pipeline.py"

    ```python
    from runnable import Pipeline, PythonTask, pickled
    import pandas as pd

    def load_data():
        # Load some sample data
        data = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        return data

    def process_data(raw_data: pd.DataFrame):
        # Process the data
        processed = raw_data * 2
        return processed

    def main():
        pipeline = Pipeline(steps=[
            PythonTask(
                function=load_data,
                name="load",
                returns=[pickled("raw_data")]  # Store in catalog
            ),
            PythonTask(
                function=process_data,
                name="process",
                returns=[pickled("processed_data")]
            )
        ])

        pipeline.execute()
        return pipeline

    if __name__ == "__main__":
        main()
    ```

=== "config.yaml"

    ```yaml
    catalog:
      type: file-system
      config:
        catalog_location: ".catalog"
    ```

**Run the example:**
```bash
RUNNABLE_CONFIGURATION_FILE=config.yaml uv run pipeline.py
```

**Result**: Data artifacts stored in `.catalog/{run_id}/` with automatic serialization and data lineage tracking.

## Object Storage (s3 / minio)

For distributed systems and cloud deployments, use object storage catalogs:

!!! info "Installation Required"

    S3 storage requires the optional S3 dependency:
    ```bash
    pip install runnable[s3]
    ```

### s3

```yaml
catalog:
  type: s3
  config:
    bucket_name: "my-pipeline-data"
    prefix: "runnable-artifacts"
    aws_access_key_id: "${AWS_ACCESS_KEY_ID}"
    aws_secret_access_key: "${AWS_SECRET_ACCESS_KEY}"
    region_name: "us-west-2"
```

### minio

```yaml
catalog:
  type: minio
  config:
    endpoint: "https://minio.company.com"
    access_key: "${MINIO_ACCESS_KEY}"
    secret_key: "${MINIO_SECRET_KEY}"
    bucket_name: "pipeline-artifacts"
```

## Custom Data Catalogs

**Need to integrate with your existing data infrastructure?** Build custom catalogs that store artifacts in any system using Runnable's plugin architecture.

!!! success "Enterprise Data Integration"

    **Connect to your existing data systems**: Never be limited by built-in storage options

    - 🏢 **Data warehouses**: Store artifacts in Snowflake, BigQuery, Redshift
    - 📊 **Data lakes**: Integrate with Delta Lake, Iceberg, Hudi
    - 🗄️ **Corporate storage**: Connect to existing NFS, HDFS, object stores
    - 🔐 **Governed data**: Meet data governance and lineage requirements

### Building a Custom Catalog

Creating a custom catalog takes just 3 steps:

!!! example "Custom Catalog Implementation"

    **1. Implement the catalog interface:**
    ```python
    from runnable.catalog import BaseCatalog
    from typing import Any

    class SnowflakeCatalog(BaseCatalog):
        service_name: str = "snowflake"

        # Configuration fields
        account: str
        user: str
        password: str
        warehouse: str
        database: str = "PIPELINE_ARTIFACTS"

        def put(self, name: str, data: Any, run_id: str) -> str:
            """Store data artifact in Snowflake"""
            # Serialize data (pickle, parquet, etc.)
            serialized_data = self._serialize(data, name)

            # Create table if needed
            table_name = f"artifacts_{run_id.replace('-', '_')}"

            # Store in Snowflake
            self._execute_sql(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    artifact_name STRING,
                    data_blob BINARY,
                    created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
                )
            """)

            # Insert artifact
            self._execute_sql(f"""
                INSERT INTO {table_name} (artifact_name, data_blob)
                VALUES ('{name}', '{serialized_data}')
            """)

            return f"{table_name}.{name}"

        def get(self, name: str, run_id: str, **kwargs) -> Any:
            """Retrieve data artifact from Snowflake"""
            table_name = f"artifacts_{run_id.replace('-', '_')}"

            result = self._execute_sql(f"""
                SELECT data_blob FROM {table_name}
                WHERE artifact_name = '{name}'
                LIMIT 1
            """)

            return self._deserialize(result[0]['DATA_BLOB'], name)

        def _execute_sql(self, query: str):
            # Execute SQL using Snowflake connector
            return self.snowflake_connection.execute(query)
    ```

    **2. Register via entry point in `pyproject.toml`:**
    ```toml
    [project.entry-points.'catalog']
    "snowflake" = "my_package.catalogs:SnowflakeCatalog"
    ```

    **3. Use in your configuration:**
    ```yaml
    catalog:
      type: snowflake
      config:
        account: "mycompany.snowflakecomputing.com"
        user: "${SNOWFLAKE_USER}"
        password: "${SNOWFLAKE_PASSWORD}"
        warehouse: "ANALYTICS_WH"
        database: "PIPELINE_DATA"
    ```

### Real-World Custom Catalog Examples

!!! tip "Production Use Cases"

    **Data Lake Integration**:
    ```python
    class DeltaLakeCatalog(BaseCatalog):
        """Store artifacts in Delta Lake with versioning"""
        service_name = "delta-lake"

        def put(self, name: str, data: Any, run_id: str):
            # Write to Delta Lake with automatic versioning
            delta_table = f"artifacts.{run_id}_{name}"
            self._write_delta_table(delta_table, data)
    ```

    **Enterprise Data Warehouse**:
    ```python
    class BigQueryCatalog(BaseCatalog):
        """Store artifacts in Google BigQuery"""
        service_name = "bigquery"

        def put(self, name: str, data: Any, run_id: str):
            # Upload to BigQuery table with metadata
            table_id = f"pipeline_artifacts.{run_id}_{name}"
            self._upload_to_bq(table_id, data)
    ```

    **High-Performance Storage**:
    ```python
    class HDFSCatalog(BaseCatalog):
        """Store large artifacts in Hadoop HDFS"""
        service_name = "hdfs"

        def put(self, name: str, data: Any, run_id: str):
            # Store in HDFS with compression
            path = f"/pipeline_data/{run_id}/{name}.parquet"
            self._write_hdfs_parquet(path, data)
    ```

### Integration Patterns

!!! info "Common Integration Approaches"

    **Database storage**: Store artifacts in relational/NoSQL databases
    ```python
    def put(self, name: str, data: Any, run_id: str):
        serialized = pickle.dumps(data)
        self.db.execute(
            "INSERT INTO artifacts (run_id, name, data) VALUES (?, ?, ?)",
            (run_id, name, serialized)
        )
    ```

    **File-based systems**: Store in distributed filesystems
    ```python
    def put(self, name: str, data: Any, run_id: str):
        path = f"{self.base_path}/{run_id}/{name}.pkl"
        self.filesystem.write_bytes(path, pickle.dumps(data))
    ```

    **Hybrid storage**: Metadata in database, data in object store
    ```python
    def put(self, name: str, data: Any, run_id: str):
        # Store data in S3/GCS
        data_url = self._upload_to_storage(data, run_id, name)

        # Store metadata in database
        self._store_metadata(run_id, name, data_url, type(data).__name__)
    ```

## Choosing the Right Catalog

!!! success "Decision Guide"

    **For most users**: Use `file-system` - works in any environment with mounted storage

    **For development/testing**: Use `do-nothing` for fast iteration without data persistence

    **Distributed systems**: Use `s3`/`minio` when execution environments can't mount shared storage

    **Enterprise integration**: Build custom catalogs to integrate with existing data infrastructure

!!! info "Filesystem vs Object Storage"

    **Filesystem catalogs** (`file-system`): Work in any execution environment where the `catalog_location` can be mounted

    **Object storage** (`s3`, `minio`): Use when shared filesystem mounting isn't available or for cloud-native deployments
