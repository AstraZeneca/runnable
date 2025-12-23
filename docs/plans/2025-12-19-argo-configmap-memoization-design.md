# Argo ConfigMap Memoization Design

## Overview

Extend the existing Argo executor's memoization schema to include ConfigMap cache support, completing compliance with Argo Workflows' native memoization specification while enabling persistent caching for workflow resubmission.

## Current State

The existing `Memoize` class only supports cache keys:
```python
class Memoize(BaseModelWIthConfig):
    key: str
```

Generated workflows produce incomplete memoization configuration:
```yaml
memoize:
  key: "{{workflow.parameters.run_id}}"
  # Missing required cache configuration
```

## Requirements

1. **Argo Compliance**: Match Argo Workflows' official memoization specification
2. **Persistent Caching**: ConfigMaps persist across workflow runs for resubmit functionality
3. **Automatic Management**: Argo controller creates ConfigMaps as needed
4. **Backward Compatibility**: Existing workflows continue working unchanged
5. **Workflow Isolation**: Each workflow uses its own ConfigMap for cache isolation

## Architecture

### Enhanced Schema

```python
class ConfigMapCache(BaseModelWIthConfig):
    name: str

class Cache(BaseModelWIthConfig):
    config_map: ConfigMapCache

class Memoize(BaseModelWIthConfig):
    key: str
    cache: Cache
```

### Generated YAML Structure

```yaml
memoize:
  key: "{{workflow.parameters.run_id}}"
  cache:
    configMap:
      name: "runnable-x7k9m2"
```

### ConfigMap Naming Strategy

- **Pattern**: `runnable-<6-random-chars>` per workflow
- **Generation**: Once per `ArgoExecutor` instance
- **Sharing**: Same ConfigMap used across all templates in workflow
- **Persistence**: ConfigMaps remain after workflow completion for resubmit

## Implementation Details

### ArgoExecutor Enhancement

```python
class ArgoExecutor(GenericPipelineExecutor):
    def __init__(self, ...):
        super().__init__(...)
        self._cache_name = self._generate_cache_name()

    def _generate_cache_name(self) -> str:
        import secrets
        import string
        chars = string.ascii_lowercase + string.digits
        return f"runnable-{''.join(secrets.choice(chars) for _ in range(6))}"

    @property
    def cache_name(self) -> str:
        return self._cache_name
```

### Template Generation Updates

Both `_create_container_template()` and `_create_nested_template()` methods will use:

```python
memoize=Memoize(
    key="{{workflow.parameters.run_id}}",
    cache=Cache(config_map=ConfigMapCache(name=self.cache_name))
)
```

### Step Isolation

- Single ConfigMap per workflow: `runnable-x7k9m2`
- Argo's internal memoization logic handles step differentiation
- Cache keys combined with template names provide automatic isolation
- No manual step-level cache management required

## Technical Considerations

### Error Handling

- **ConfigMap Creation**: Argo controller handles creation failures gracefully
- **Size Limits**: 1MB ConfigMap limit managed by Argo (entry eviction/overflow)
- **Permissions**: Requires `configmaps` create/read permissions for Argo service account

### Backward Compatibility

- Existing `Memoize(key="...")` usage remains valid
- No breaking changes to current API
- New cache functionality enabled automatically for all workflows

### Cache Lifecycle

- **Creation**: Argo controller creates ConfigMaps on first memoization access
- **Persistence**: ConfigMaps persist indefinitely for resubmit functionality
- **Cleanup**: Manual cleanup outside of workflow execution (if desired)

## Benefits

1. **Full Argo Compliance**: Complete implementation of Argo's memoization specification
2. **True Memoization**: Persistent cache enables cross-workflow-run optimization
3. **Seamless Integration**: No configuration changes required for existing workflows
4. **Resubmit Support**: Cached results available for workflow resubmission scenarios
5. **Workflow Isolation**: Per-workflow ConfigMaps prevent cache collision

## Testing Strategy

1. **Schema Tests**: Verify YAML serialization matches Argo specification
2. **Name Generation**: Test ConfigMap name uniqueness and format
3. **Argo Validation**: Ensure `argo lint` passes for generated workflows
4. **Integration Tests**: Verify memoization functionality in actual workflow runs
5. **Resubmit Tests**: Confirm cached results used in resubmitted workflows

## Migration Path

No migration required - enhancement is additive:
- Current workflows automatically get enhanced memoization
- No configuration changes needed
- Backward compatibility maintained

## Technical Notes

- ConfigMap names provide sufficient uniqueness for concurrent workflows
- Argo service account requires ConfigMap permissions (typically already present)
- Cache persistence aligns with resubmit workflow design patterns
- Generated cache names are deterministic per executor instance but unique across instances
