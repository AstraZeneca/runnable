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

## Converting from Argparse Scripts

**Zero-code migration**: Existing argparse functions work directly with PythonJobs! Runnable automatically converts YAML parameters to `argparse.Namespace` objects.

### Your Existing Argparse Script

```python title="single_cpu_args.py"
import argparse
import torch
# ... other imports

def run_single_cpu_training(args: argparse.Namespace):
    """Training function that expects parsed arguments."""
    print(f"Learning Rate: {args.learning_rate}, Epochs: {args.num_epochs}")
    print(f"Batch Size: {args.batch_size}")

    # Use args.learning_rate, args.num_epochs, args.batch_size
    # ... training logic

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single-CPU PyTorch Training")
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)

    args = parser.parse_args()
    run_single_cpu_training(args)
```

**Current usage:** `python single_cpu_args.py --learning_rate 0.05 --num_epochs 100`

### Direct PythonJob Integration

**No code changes needed** - just wrap your existing function:

```python title="argparse_job.py"
from my_module import run_single_cpu_training  # Your existing function!
from runnable import PythonJob

def main():
    # Use your argparse function directly - no modifications needed
    training_job = PythonJob(function=run_single_cpu_training)

    # Runnable automatically converts YAML to argparse.Namespace
    training_job.execute(parameters_file="training_params.yaml")

    return training_job

if __name__ == "__main__":
    main()
```

**Create parameter file** (matches your argparse arguments):
```yaml title="training_params.yaml"
learning_rate: 0.01
num_epochs: 50
batch_size: 32
```

**New usage:** `uv run argparse_job.py` - same function, zero code changes!

??? example "See complete working example"
    ```python title="examples/torch/single_cpu_args.py"
    --8<-- "examples/torch/single_cpu_args.py"
    ```

!!! success "Magic conversion"

    Runnable automatically creates an `argparse.Namespace` object from your YAML parameters. Your function receives exactly what it expects - no code changes required!

### Migration Benefits

**🔄 Replace command-line complexity:**
```bash
# Before: Long command lines
python script.py --learning_rate 0.05 --num_epochs 100 --batch_size 64

# After: Clean execution
uv run training_job.py
```

**🔗 Pipeline integration:**
```python
# Your argparse function becomes a pipeline step
from runnable import Pipeline, PythonTask

pipeline = Pipeline(steps=[
    PythonTask(function=preprocess_data),
    PythonTask(function=run_single_cpu_training),  # Zero changes needed!
    PythonTask(function=evaluate_model)
])
```

!!! tip "Keep both versions"

    Your original argparse script continues working unchanged. The PythonJob version gives you additional capabilities without any migration risk!

## Best Practices

### ✅ **Use the Three-Layer System**
Combine all parameter methods for maximum flexibility:

```python
# 1. Base config in code
job.execute(parameters_file="base_config.yaml")

# 2. Environment-specific file
export RUNNABLE_PARAMETERS_FILE="prod_overrides.yaml"

# 3. Individual runtime tweaks
export RUNNABLE_PRM_debug=true
```

### ✅ **Environment Variables for Deployment Values**
Use env vars for values that differ between environments:

```bash
# Production deployment
export RUNNABLE_PRM_database_url="postgresql://prod:5432/app"
export RUNNABLE_PRM_api_key="prod-key-123"
export RUNNABLE_PRM_debug=false
```

### ✅ **YAML for Complex Configuration**
Keep structured config in parameter files:

```yaml
# Complex nested configuration
model_settings:
  learning_rate: 0.01
  layers: [128, 64, 32]
  dropout: 0.2

data_pipeline:
  source: "s3://bucket/data/"
  transformations: ["normalize", "encode_categoricals"]
  validation_split: 0.2
```

## What's Next?

Your Jobs are now fully configurable! Next topics:

- **[File Storage](file-storage.md)** - Store files created during execution
- **[Job Types](job-types.md)** - Shell and Notebook Jobs
- **[Configuration](configuration.md)** - Advanced Job options

Ready to store files from your Jobs? Continue to **[File Storage](file-storage.md)**!
