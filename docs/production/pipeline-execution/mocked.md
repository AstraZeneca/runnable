# Mocked Pipeline Execution

Test and validate your pipeline structure without running the actual tasks - perfect for development, testing, and debugging workflows.

## What is Mocked Execution?

Mocked execution allows you to:

- **Validate pipeline structure** - Check that your workflow logic is correct
- **Test failure scenarios** - Simulate different outcomes without side effects
- **Debug pipeline issues** - Isolate problems by mocking successful steps
- **Speed up development** - Skip time-consuming tasks during pipeline development

!!! tip "Pipeline Validation Made Simple"

    Mocked execution runs your entire pipeline workflow but skips the actual task execution, letting you verify the logic and flow instantly.

## Getting Started

### Basic Configuration

```yaml
pipeline-executor:
  type: mocked
```

### Simple Example

=== "pipeline.py"

    ```python
    from runnable import Pipeline, PythonTask

    def slow_data_processing():
        # This would normally take hours
        print("Processing massive dataset...")
        return {"processed_records": 1000000}

    def train_model(data):
        # This would normally take even longer
        print(f"Training on {data['processed_records']} records...")
        return {"accuracy": 0.95}

    def main():
        pipeline = Pipeline(steps=[
            PythonTask(function=slow_data_processing, name="process_data"),
            PythonTask(function=train_model, name="train_model")
        ])

        # Same pipeline code - just different execution
        pipeline.execute()
        return pipeline

    if __name__ == "__main__":
        main()
    ```

=== "config.yaml"

    ```yaml
    pipeline-executor:
      type: mocked  # Skip actual execution, just validate structure
    ```

**Run with mocking:**
```bash
RUNNABLE_CONFIGURATION_FILE=config.yaml uv run pipeline.py
```

**Result**: Pipeline completes in seconds, validating the workflow without running the time-consuming tasks.

## Why Use Mocked Execution?

!!! success "Development Benefits"

    **Fast iteration**: Test pipeline changes without waiting for long-running tasks

    **Safe testing**: No side effects, no resource usage, no accidental data changes

!!! info "Testing Benefits"

    **Structure validation**: Verify your pipeline logic and flow

    **Failure scenario testing**: Test different outcomes using patches

!!! warning "Debugging Benefits"

    **Issue isolation**: Mock successful steps to focus on problematic ones

    **State recreation**: Debug failed executions by replaying with mocked setup

## Advanced Usage: Patching Tasks

Override specific tasks to test different scenarios:

=== "pipeline.py"

    ```python
    from runnable import Pipeline, ShellTask

    def main():
        pipeline = Pipeline(steps=[
            ShellTask(command="process_data.py", name="data_processing"),
            ShellTask(command="train_model.py", name="model_training"),
            ShellTask(command="deploy_model.py", name="deployment")
        ])

        pipeline.execute()
        return pipeline

    if __name__ == "__main__":
        main()
    ```

=== "test_success.yaml"

    Test the happy path by mocking a potentially failing step:

    ```yaml
    pipeline-executor:
      type: mocked
      config:
        patches:
          data_processing:
            command: "echo 'Data processing successful'"
          # model_training and deployment will be mocked (pass-through)
    ```

=== "test_failure.yaml"

    Test failure handling by simulating a failing step:

    ```yaml
    pipeline-executor:
      type: mocked
      config:
        patches:
          model_training:
            command: "exit 1"  # Simulate training failure
          # Other steps will be mocked (pass-through)
    ```

**Test different scenarios:**
```bash
# Test success scenario
RUNNABLE_CONFIGURATION_FILE=test_success.yaml uv run pipeline.py

# Test failure scenario
RUNNABLE_CONFIGURATION_FILE=test_failure.yaml uv run pipeline.py
```

## Configuration Reference

```yaml
pipeline-executor:
  type: mocked
  config:
    patches:                    # Optional: Override specific tasks
      task_name:
        command: "new_command"  # Replace task command
        # Other task configuration options available
```

### Patch Options

**For Python tasks:**
- `command`: Override the function being called

**For Shell tasks:**
- `command`: Replace the shell command

**For Notebook tasks:**
- `command`: Override notebook path
- Additional notebook-specific options available

## Common Use Cases

### 1. Development Workflow

```yaml
# Skip expensive operations during development
pipeline-executor:
  type: mocked
  config:
    patches:
      expensive_computation:
        command: "echo 'Skipped expensive step'"
```

### 2. CI/CD Testing

```yaml
# Validate pipeline structure in CI without running actual workloads
pipeline-executor:
  type: mocked
  # No patches needed - just validate structure
```

### 3. Debugging Failed Pipelines

Debug production failures locally by recreating the exact state from the failed run:

```yaml
# Mock successful steps to isolate the failing one
pipeline-executor:
  type: mocked
  config:
    patches:
      failing_step:
        command: "echo 'This step now succeeds'"
      # Don't patch the step you want to debug
```

### 4. Training & Documentation

```yaml
# Demonstrate pipeline behavior without side effects
pipeline-executor:
  type: mocked
```

## Debugging Failed Production Runs

When a pipeline fails in production, you can debug it locally using mocked execution to recreate the exact conditions.

!!! info "Prerequisites for Debugging"

    **Local machine access**: Mocked executor only runs locally

    **Failed run data**: Access to catalog data and run logs from the failed execution

    **Same codebase**: The pipeline code that failed in production

### Step-by-Step Debugging Workflow

**1. Identify the Failed Step**

First, examine the run log to find which step failed:

```bash
# Find the failed run log (usually in .run_log_store/)
ls .run_log_store/

# Or if using remote run log store, download it locally
# Example: aws s3 cp s3://my-logs/failed-run-id.json ./
```

Look for the failed step in the run log JSON:
```json
{
  "run_id": "failed-production-run",
  "status": "FAIL",
  "steps": {
    "data_processing": {"status": "SUCCESS"},
    "model_training": {"status": "FAIL"},  # <- This step failed
    "deployment": {"status": "NOT_EXECUTED"}
  }
}
```

**2. Copy Catalog Data Locally**

The failing step needs access to the exact data state from when it failed:

```bash
# Copy catalog data from failed run to local debugging location
# If using file-system catalog:
cp -r .catalog/failed-production-run ./debug-catalog/

# If using S3 catalog, download the data:
# aws s3 sync s3://my-catalog/failed-production-run/ ./debug-catalog/

# Make it available for debugging with a new run-id
cp -r ./debug-catalog/ .catalog/debug-session/
```

**3. Create Debug Configuration**

Mock all successful steps, let the failed step run with the real data:

```yaml
# debug-config.yaml
pipeline-executor:
  type: mocked
  config:
    patches:
      # Mock the successful steps (they already ran in production)
      data_processing:
        command: "echo 'Data processing - already completed in production'"

      # Don't patch model_training - let it run with real data
      # deployment step will be mocked by default since model_training might still fail

# Use local storage to access the copied catalog data
catalog:
  type: file-system
  config:
    catalog_location: ".catalog"

run-log-store:
  type: file-system
  config:
    log_folder: ".run_log_store"
```

**4. Debug the Failed Step**

Run the pipeline with the debug configuration:

```bash
# Run with specific run-id to access the copied catalog data
RUNNABLE_CONFIGURATION_FILE=debug-config.yaml uv run pipeline.py --run-id debug-session
```

**What happens:**
- `data_processing` step is mocked (skipped)
- `model_training` step runs with the actual data from production
- You can now debug the failing step locally with production data

**5. Debug with IDE/Debugger**

Since mocked executor runs locally, you can use your favorite debugging tools:

```python
def train_model(data):
    import pdb; pdb.set_trace()  # Set breakpoint

    # Debug the actual failing logic with production data
    model = train_complex_model(data)
    return model

def main():
    pipeline = Pipeline(steps=[
        PythonTask(function=load_data, name="data_processing"),
        PythonTask(function=train_model, name="model_training"),  # Will hit breakpoint
        PythonTask(function=deploy_model, name="deployment")
    ])

    pipeline.execute()
    return pipeline
```

### Example: Complete Debug Session

**Original failed pipeline:**
```python
def main():
    pipeline = Pipeline(steps=[
        PythonTask(function=load_data, name="load_data"),
        PythonTask(function=process_data, name="process_data"),
        PythonTask(function=train_model, name="train_model"),  # Failed here
        PythonTask(function=deploy_model, name="deploy")
    ])
    return pipeline
```

**Debug configuration:**
```yaml
# debug-failed-training.yaml
pipeline-executor:
  type: mocked
  config:
    patches:
      load_data:
        command: "echo 'Mocked: Data already loaded in production'"
      process_data:
        command: "echo 'Mocked: Data already processed in production'"
      # train_model - no patch, let it run with real data
      # deploy - will be mocked by default
```

**Debug session:**
```bash
# Copy production catalog data
cp -r .catalog/prod-failure-run-id .catalog/debug-training-issue

# Run debug session
RUNNABLE_CONFIGURATION_FILE=debug-failed-training.yaml uv run pipeline.py --run-id debug-training-issue

# Now you can debug train_model with the exact production data state
```

### Debug Tips

!!! tip "Debugging Best Practices"

    **Preserve original data**: Always copy catalog data, never modify the original

    **Use meaningful run-ids**: Name debug sessions clearly (e.g., `debug-training-failure-2024`)

    **Mock successful steps**: Only run the failing step and its dependencies

    **Check parameters**: Verify the failing step receives the same parameters as in production

!!! warning "Important Notes"

    **Local environment**: Ensure your local environment matches production (dependencies, versions)

    **Data access**: Make sure you have permission to access production catalog data

    **Secrets**: Production secrets may not be available locally - use debug values if needed

## Best Practices

!!! tip "When to Use Mocked Execution"

    **During development**: Validate logic without running expensive tasks

    **In testing**: Verify pipeline structure and failure handling

    **For debugging**: Isolate issues by mocking successful steps

    **In demos**: Show pipeline behavior without side effects

!!! warning "Limitations"

    **No actual output**: Mocked tasks don't produce real results

    **Limited validation**: Can't test actual task logic, only pipeline structure

    **Local only**: Mocked executor runs only on local machine

## When to Use Mocked Execution

!!! question "Choose Mocked When"

    - Developing and iterating on pipeline structure
    - Testing pipeline logic and failure scenarios
    - Debugging complex workflow issues
    - Demonstrating pipelines without side effects
    - Validating pipeline changes in CI/CD

!!! abstract "Use Other Executors When"

    - Need actual task execution and results
    - Testing production performance characteristics
    - Running final validation before deployment

---

**Related:** [Local Executor](local.md) | [Pipeline Testing Patterns](../../advanced-patterns/mocking-testing.md)
