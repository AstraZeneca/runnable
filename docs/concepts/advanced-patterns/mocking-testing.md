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

```python linenums="1"
--8<-- "examples/01-tasks/stub.py:7:14"
```

Stubs act like real tasks but do nothing - perfect for testing structure.

## Mock entire workflows

Replace expensive operations during testing:

```python
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
executor:
  type: mocked
  override:
    train_model: mock_trained_model
```

**No code changes needed** - same pipeline, different behavior.

## Patching user functions

Override specific functions for testing:

**examples/08-mocking/mocked-config-unittest.yaml:**
```yaml
executor:
  type: mocked
  override:
    expensive_api_call: return_test_data
    slow_processing: return_mock_results
```

## Testing patterns

### Test workflow structure
```python
def test_pipeline_structure():
    # Use stubs for all tasks
    pipeline = create_pipeline_with_stubs()
    result = pipeline.execute()
    assert result.is_success()
```

### Test parameter flow
```python
def test_parameter_passing():
    # Mock first task to return known data
    # Verify second task receives correct parameters
    pipeline = create_pipeline_with_mocks()
    pipeline.execute()
    # Assertions about parameter values
```

### Test conditional paths
```python
def test_conditional_branches():
    # Mock decision function to return specific values
    # Verify correct branch executes
    for condition in ["branch_a", "branch_b"]:
        pipeline = create_test_pipeline(mock_condition=condition)
        result = pipeline.execute()
        # Assert correct branch was taken
```

### Test failure handling
```python
def test_failure_recovery():
    # Mock task to always fail
    # Verify recovery pipeline executes
    pipeline = create_pipeline_with_failing_mock()
    result = pipeline.execute()
    assert result.is_success()  # Recovery worked
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
