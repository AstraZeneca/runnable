# Parameters & Environment ⚙️

Configure Jobs without changing code using parameters and environment variables.

## Parameter Files

Pass configuration to Jobs using YAML files:

```python
from examples.torch.single_cpu import run_single_cpu_training
from runnable import PythonJob

def main():
    training_job = PythonJob(function=run_single_cpu_training)

    # Execute with parameters from YAML file
    training_job.execute(parameters_file="examples/torch/single_cpu_params.yaml")

    return training_job

if __name__ == "__main__":
    main()
```

??? example "See complete runnable code"
    ```python title="examples/torch/single_cpu_job.py"
    --8<-- "examples/torch/single_cpu_job.py"
    ```

**Parameter File (single_cpu_params.yaml):**
```yaml
learning_rate: 0.01
num_epochs: 50
batch_size: 32
```

**Parameter Display During Execution:**
```
Parameters available for the execution:
{
    'learning_rate': JsonParameter(value=0.01),
    'num_epochs': JsonParameter(value=50),
    'batch_size': JsonParameter(value=32)
}
```

## Environment Variable Overrides 🏆

Environment variables **always win** over YAML values:

```bash
# Override individual parameters
export RUNNABLE_PRM_learning_rate=0.05
export RUNNABLE_PRM_num_epochs=100

# Run the same Job - now uses overridden values!
uv run examples/torch/single_cpu_job.py
```

**Result:** Job uses `learning_rate=0.05` and `num_epochs=100` instead of YAML values.

## Dynamic Parameter Files

Switch parameter files without changing code:

```bash
# Development environment
export RUNNABLE_PARAMETERS_FILE="configs/dev.yaml"
uv run my_job.py

# Production environment
export RUNNABLE_PARAMETERS_FILE="configs/prod.yaml"
uv run my_job.py  # Same code, different config!
```

```python
def main():
    job = PythonJob(function=my_function)

    # No parameters_file specified - uses RUNNABLE_PARAMETERS_FILE
    job.execute()
    return job
```

## Three-Layer Parameter Precedence

Parameters are resolved in this order (highest priority wins):

1. **Individual overrides**: `RUNNABLE_PRM_key="value"`
2. **Environment file**: `RUNNABLE_PARAMETERS_FILE="config.yaml"`
3. **Code-specified**: `job.execute(parameters_file="config.yaml")`

!!! tip "Same flexibility as Pipelines"

    Jobs inherit the exact same parameter system as Pipelines. Perfect for dev/staging/prod environments without code changes!

## Common Patterns

### Environment-Specific Configurations

**Development:**
```bash
export RUNNABLE_PARAMETERS_FILE="configs/dev.yaml"
export RUNNABLE_PRM_debug=true
export RUNNABLE_PRM_sample_size=1000
```

**Production:**
```bash
export RUNNABLE_PARAMETERS_FILE="configs/prod.yaml"
export RUNNABLE_PRM_debug=false
export RUNNABLE_PRM_sample_size=1000000
```

### Complex Parameter Types

```bash
# JSON objects
export RUNNABLE_PRM_model_config='{"learning_rate": 0.01, "epochs": 100}'

# Lists
export RUNNABLE_PRM_features='["age", "income", "location"]'

# Nested configuration
export RUNNABLE_PRM_database='{"host": "prod.db.com", "port": 5432}'
```

### Testing Different Values

```bash
# Test different thresholds
export RUNNABLE_PRM_confidence_threshold=0.8
uv run analysis_job.py

export RUNNABLE_PRM_confidence_threshold=0.9
uv run analysis_job.py

# Test different data sources
export RUNNABLE_PRM_data_source="s3://bucket/test-data.csv"
uv run processing_job.py

export RUNNABLE_PRM_data_source="s3://bucket/prod-data.csv"
uv run processing_job.py
```

## Parameter File Examples

### Basic Configuration
```yaml title="config.yaml"
# Data settings
input_file: "data/sales.csv"
output_dir: "results/"

# Processing settings
batch_size: 1000
parallel_jobs: 4

# Model settings
model_type: "random_forest"
max_depth: 10
```

### Environment-Specific Files

**dev.yaml:**
```yaml
database_url: "sqlite:///dev.db"
log_level: "DEBUG"
sample_data: true
data_limit: 1000
```

**prod.yaml:**
```yaml
database_url: "postgresql://prod.db:5432/app"
log_level: "INFO"
sample_data: false
data_limit: null
```

## Parameter Validation

Functions receive parameters with full type checking:

```python
def process_data(
    input_file: str,
    batch_size: int = 100,
    debug: bool = False,
    model_config: dict = None
):
    # Parameters are validated and converted automatically
    print(f"Processing {input_file} with batch_size={batch_size}")

    if debug:
        print("Debug mode enabled")

    return {"processed": True}
```

## Best Practices

### ✅ **Environment Variables for Flexibility**
```bash
# Use env vars for values that change between environments
export RUNNABLE_PRM_database_url="postgresql://prod:5432/app"
export RUNNABLE_PRM_api_key="prod-key-123"
export RUNNABLE_PRM_debug=false
```

### ✅ **Parameter Files for Complex Config**
```yaml
# Use YAML for complex, structured configuration
model_settings:
  learning_rate: 0.01
  layers: [128, 64, 32]
  dropout: 0.2

data_pipeline:
  source: "s3://bucket/data/"
  transformations: ["normalize", "encode_categoricals"]
  validation_split: 0.2
```

### ✅ **Layered Configuration Strategy**
```bash
# Base configuration in code
job.execute(parameters_file="base_config.yaml")

# Environment-specific overrides
export RUNNABLE_PARAMETERS_FILE="prod_overrides.yaml"

# Individual tweaks
export RUNNABLE_PRM_debug=true
```

## What's Next?

Your Jobs are now fully configurable! Next topics:

- **[File Storage](file-storage.md)** - Store files created during execution
- **[Job Types](job-types.md)** - Shell and Notebook Jobs
- **[Configuration](configuration.md)** - Advanced Job options

Ready to store files from your Jobs? Continue to **[File Storage](file-storage.md)**!
