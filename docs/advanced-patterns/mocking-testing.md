# 🎭 Mocking and Testing

Test your pipeline structure and logic without running expensive operations.

## Why mock?

**Test pipeline logic:**

- Verify workflow structure
- Test parameter flow
- Check conditional branches
- Validate failure handling

**Without the cost:**

- Skip slow ML training
- Avoid external API calls
- Test with fake data
- Debug workflow issues

## The stub pattern

Replace any task with a stub during development:

```python
from runnable import Pipeline, Stub

def main():
    # Replace expensive operations with stubs
    data_extraction = Stub(name="extract_data")         # Instead of slow API calls
    model_training = Stub(name="train_model")           # Instead of hours of training
    report_generation = Stub(name="generate_report")   # Instead of complex rendering

    # Test your pipeline structure
    pipeline = Pipeline(steps=[data_extraction, model_training, report_generation])
    pipeline.execute()  # Runs instantly, tests workflow logic
    return pipeline

if __name__ == "__main__":
    main()
```

??? example "See complete runnable code"
    ```python title="examples/01-tasks/stub.py"
    --8<-- "examples/01-tasks/stub.py"
    ```

    **Try it now:**
    ```bash
    uv run examples/01-tasks/stub.py
    ```

Stubs act like real tasks but do nothing - perfect for testing structure.

## Mock entire workflows

Replace expensive operations during testing:

```python
# Example task replacement (partial code)

# Production version
expensive_training_task = PythonTask(
    name="train_model",
    function=train_large_model,  # Takes hours
    returns=["model"]
)

# Test version
mock_training_task = Stub(
    name="train_model",
    returns=["model"]  # Returns mock data
)
```

## Configuration-based mocking

Use different configs for testing vs production:

**examples/08-mocking/mocked-config-simple.yaml:**
```yaml
catalog:
  type: file-system

run-log-store:
  type: file-system

pipeline-executor:
  type: mocked
```

**No code changes needed** - same pipeline, different behavior.

## Patching user functions

Override specific functions for testing:

**examples/08-mocking/mocked-config-unittest.yaml:**
```yaml
catalog:
  type: file-system

run-log-store:
  type: file-system

pipeline-executor:
  type: mocked
  config:
    patches:
      step 1:
        command: exit 0
```

**Advanced patching example:**
```yaml
executor:
  type: mocked
  config:
    patches:
      hello python:
        command: examples.common.functions.mocked_hello
      hello shell:
        command: echo "hello from mocked"
      hello notebook:
        command: examples/common/simple_notebook_mocked.ipynb
```

## Testing patterns with mock executor

### Test workflow structure
```python
from runnable import Pipeline, PythonTask

def main():
    # Your actual pipeline
    pipeline = Pipeline(steps=[
        PythonTask(function=extract_data, name="extract"),
        PythonTask(function=transform_data, name="transform"),
        PythonTask(function=load_data, name="load")
    ])

    # Execute with mock configuration to test structure
    pipeline.execute(configuration_file="examples/08-mocking/mocked-config-simple.yaml")
    return pipeline
```

### Test with patched functions
```yaml
# test-config.yaml - Mock specific tasks
pipeline-executor:
  type: mocked
  config:
    patches:
      extract:
        command: examples.test.functions.mock_extract_data
      transform:
        command: examples.test.functions.mock_transform_data
```

```python
def main():
    pipeline = Pipeline(steps=[
        PythonTask(function=extract_data, name="extract"),
        PythonTask(function=transform_data, name="transform"),
        PythonTask(function=load_data, name="load")
    ])

    # Test with patched functions
    pipeline.execute(configuration_file="test-config.yaml")
    return pipeline
```

### Test failure scenarios
```yaml
# failure-test-config.yaml - Mock failure conditions
pipeline-executor:
  type: mocked
  config:
    patches:
      extract:
        command: exit 1  # Simulate failure
```

```python
def main():
    pipeline = Pipeline(steps=[
        PythonTask(function=extract_data, name="extract", on_failure="handle_failure"),
        PythonTask(function=handle_failure, name="handle_failure"),
        PythonTask(function=load_data, name="load")
    ])

    # Test failure handling
    pipeline.execute(configuration_file="failure-test-config.yaml")
    return pipeline
```

### Test conditional branches
```yaml
# branch-test-config.yaml - Mock decision outcomes
pipeline-executor:
  type: mocked
  config:
    patches:
      decision_task:
        command: echo "branch_a"  # Force specific branch
```

```python
def main():
    pipeline = Pipeline(steps=[
        PythonTask(function=make_decision, name="decision_task"),
        # Conditional logic based on decision_task output
    ])

    # Test specific branch execution
    pipeline.execute(configuration_file="branch-test-config.yaml")
    return pipeline
```

## Mocking strategies

**Development:**

- Start with stubs for all tasks
- Implement one task at a time
- Test each addition independently

**Testing:**

- Mock external dependencies
- Use deterministic test data
- Test edge cases and failures

**Staging:**

- Mix real and mocked components
- Test with production-like data
- Validate performance characteristics

!!! tip "Mocking best practices"

    - **Mock boundaries**: External APIs, file systems, slow operations
    - **Keep interfaces**: Mocks should match real task signatures
    - **Test both**: Test with mocks AND real implementations
    - **Use configuration**: Switch between mock/real via config files

This completes our journey through Runnable's advanced patterns - from simple functions to sophisticated, resilient workflows.
