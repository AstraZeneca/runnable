# 🔧 Pipeline Parameters & Environment

Configure Pipeline execution without changing code using parameters and environment variables.

## Parameter System

Pipelines share the same flexible parameter system as Jobs, with three layers of configuration precedence:

1. **Individual overrides**: `RUNNABLE_PRM_key="value"` (highest priority)
2. **Environment file**: `RUNNABLE_PARAMETERS_FILE="config.yaml"`
3. **Code-specified**: `pipeline.execute(parameters_file="config.yaml")` (lowest priority)

```python
from runnable import Pipeline, PythonTask

def process_data(batch_size=100, debug=False):
    print(f"Processing with batch_size={batch_size}, debug={debug}")
    return {"processed": True}

def main():
    pipeline = Pipeline(steps=[
        PythonTask(function=process_data, name="process")
    ])

    # Execute with parameter file
    pipeline.execute(parameters_file="config.yaml")
    return pipeline

if __name__ == "__main__":
    main()
```

**Parameter File (config.yaml):**
```yaml
batch_size: 1000
debug: true
```

## Environment Variable Overrides

Override any parameter at runtime:

```bash
# Override specific parameters
export RUNNABLE_PRM_batch_size=500
export RUNNABLE_PRM_debug=false

# Run pipeline - uses overridden values
uv run data_pipeline.py
```

## Custom Run IDs

Control pipeline execution tracking with custom identifiers:

```bash
# Set custom run ID for tracking
export RUNNABLE_RUN_ID="data-pipeline-daily-run-2024-11-20"
uv run data_processing_pipeline.py
```

**Benefits:**

- **Easy identification** in logs and run history
- **Consistent naming** across pipeline executions
- **Better debugging** when tracking specific pipeline runs
- **Integration** with external systems using predictable IDs

### Pipeline Run ID Examples

```bash
# Daily data processing
export RUNNABLE_RUN_ID="etl-daily-$(date +%Y-%m-%d)"
uv run daily_etl_pipeline.py

# Experiment tracking
export RUNNABLE_RUN_ID="experiment-feature-engineering-v2"
uv run ml_experiment_pipeline.py

# Environment-specific runs
export RUNNABLE_RUN_ID="staging-deployment-$(git rev-parse --short HEAD)"
uv run deployment_pipeline.py
```

!!! tip "Default vs Custom Run IDs"

    **Without RUNNABLE_RUN_ID:** Auto-generated names like `courtly-easley-1719`

    **With RUNNABLE_RUN_ID:** Your custom identifier `data-pipeline-daily-run-2024-11-20`

## Dynamic Parameter Files

Switch configurations without code changes:

```bash
# Development environment
export RUNNABLE_PARAMETERS_FILE="configs/dev.yaml"
uv run ml_pipeline.py

# Production environment
export RUNNABLE_PARAMETERS_FILE="configs/prod.yaml"
uv run ml_pipeline.py  # Same code, different config!
```

## Common Pipeline Patterns

### Environment-Specific Configurations

**Development (dev.yaml):**
```yaml
data_source: "s3://dev-bucket/sample-data/"
batch_size: 100
debug: true
parallel_workers: 1
```

**Production (prod.yaml):**
```yaml
data_source: "s3://prod-bucket/full-data/"
batch_size: 10000
debug: false
parallel_workers: 8
```

### Multi-Stage Pipeline Configuration

```bash
# Configure entire pipeline execution
export RUNNABLE_PARAMETERS_FILE="configs/full-pipeline.yaml"
export RUNNABLE_RUN_ID="daily-ml-pipeline-$(date +%Y%m%d)"

# Override specific stages
export RUNNABLE_PRM_training_epochs=100
export RUNNABLE_PRM_validation_split=0.2

uv run ml_training_pipeline.py
```

## Best Practices

### ✅ **Use Run IDs for Pipeline Tracking**
```bash
# Predictable naming for scheduled runs
export RUNNABLE_RUN_ID="weekly-report-$(date +%Y-week-%U)"

# Git-based versioning for deployments
export RUNNABLE_RUN_ID="deploy-$(git rev-parse --short HEAD)"

# Feature branch testing
export RUNNABLE_RUN_ID="test-$(git branch --show-current)-$(date +%s)"
```

### ✅ **Environment Variables for Deployment**
```bash
# Production deployment values
export RUNNABLE_PRM_database_url="postgresql://prod:5432/warehouse"
export RUNNABLE_PRM_s3_bucket="company-prod-data"
export RUNNABLE_PRM_notification_webhook="https://alerts.company.com/pipeline"
```

### ✅ **Layered Configuration Strategy**
```python
def main():
    pipeline = Pipeline(steps=[...])

    # 1. Base configuration in code
    pipeline.execute(parameters_file="base_config.yaml")

    # 2. Environment-specific overrides via RUNNABLE_PARAMETERS_FILE
    # 3. Runtime tweaks via RUNNABLE_PRM_* variables
    # 4. Tracking via RUNNABLE_RUN_ID

    return pipeline
```

!!! info "Shared Parameter System"

    Pipelines use the exact same parameter system as Jobs. Once you learn parameters for Jobs, you already know how to configure Pipelines!

## What's Next?

- **[Jobs vs Pipelines](jobs-vs-pipelines.md)** - When to use which approach
- **[Task Types](task-types.md)** - Different ways to define pipeline steps
