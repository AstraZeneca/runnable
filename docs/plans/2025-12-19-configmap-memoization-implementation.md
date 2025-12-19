# ConfigMap Memoization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add ConfigMap cache support to Argo memoization schema for persistent caching across workflow resubmissions

**Architecture:** Extend existing Memoize class with ConfigMapCache and Cache models, add cache name generation to ArgoExecutor, update template creation methods to include cache configuration

**Tech Stack:** Python, Pydantic, Argo Workflows YAML generation, practical testing with example pipelines

---

## Task 1: Extend Memoization Schema Classes

**Files:**
- Modify: `extensions/pipeline_executor/argo.py:114-116`

**Step 1: Implement the new schema classes**

```python
# extensions/pipeline_executor/argo.py (after line 116)
class ConfigMapCache(BaseModelWIthConfig):
    name: str


class Cache(BaseModelWIthConfig):
    config_map: ConfigMapCache


# Modify existing Memoize class (line 114-116)
class Memoize(BaseModelWIthConfig):
    key: str
    cache: Optional[Cache] = Field(default=None)
```

**Step 2: Test schema with simple verification**

Run: `uv run python -c "
from extensions.pipeline_executor.argo import Memoize, Cache, ConfigMapCache
cache = Cache(config_map=ConfigMapCache(name='test-123'))
memoize = Memoize(key='test-key', cache=cache)
print('Schema works:', memoize.model_dump())
"`
Expected: Should print schema dict with cache structure

**Step 3: Commit**

```bash
git add extensions/pipeline_executor/argo.py
git commit -m "feat: add ConfigMap cache support to Memoize schema"
```

---

## Task 2: Add Cache Name Generation to ArgoExecutor

**Files:**
- Modify: `extensions/pipeline_executor/argo.py` (ArgoExecutor class)

**Step 1: Add cache name generation methods to ArgoExecutor**

```python
# extensions/pipeline_executor/argo.py (in ArgoExecutor __init__ method)
def __init__(self, config: dict):
    # Add after existing __init__ logic
    super().__init__(config)
    # ... existing initialization ...
    self._cache_name = self._generate_cache_name()

def _generate_cache_name(self) -> str:
    """Generate a unique ConfigMap name for this workflow's cache."""
    import secrets
    import string
    chars = string.ascii_lowercase + string.digits
    suffix = ''.join(secrets.choice(chars) for _ in range(6))
    return f"runnable-{suffix}"

@property
def cache_name(self) -> str:
    """Get the ConfigMap name for this workflow's memoization cache."""
    return self._cache_name
```

**Step 2: Test cache name generation**

Run: `uv run python -c "
from extensions.pipeline_executor.argo import ArgoExecutor
import re
# Test with minimal config
config = {'pvc_for_runnable': 'test', 'defaults': {}}
executor = ArgoExecutor(config)
name = executor.cache_name
print('Cache name:', name)
assert re.match(r'^runnable-[a-z0-9]{6}$', name), f'Invalid format: {name}'
print('Format validation passed')
"`
Expected: Should print valid cache name and pass format validation

**Step 3: Commit**

```bash
git add extensions/pipeline_executor/argo.py
git commit -m "feat: add ConfigMap cache name generation to ArgoExecutor"
```

---

## Task 3: Update Template Creation Methods with Cache

**Files:**
- Modify: `extensions/pipeline_executor/argo.py:601` (_create_container_template method)
- Modify: `extensions/pipeline_executor/argo.py:676` (_create_nested_template method)

**Step 1: Update _create_container_template to include cache**

```python
# extensions/pipeline_executor/argo.py (in _create_container_template method around line 601)

# Find the existing memoize line:
# memoize=Memoize(key="{{workflow.parameters.run_id}}"),

# Replace with:
memoize=Memoize(
    key="{{workflow.parameters.run_id}}",
    cache=Cache(config_map=ConfigMapCache(name=self.cache_name))
),
```

**Step 2: Update _create_nested_template to include cache**

```python
# extensions/pipeline_executor/argo.py (in _create_nested_template method around line 676)

# Find the existing memoize line:
# memoize=Memoize(key="{{workflow.parameters.run_id}}"),

# Replace with:
memoize=Memoize(
    key="{{workflow.parameters.run_id}}",
    cache=Cache(config_map=ConfigMapCache(name=self.cache_name))
),
```

**Step 3: Commit**

```bash
git add extensions/pipeline_executor/argo.py
git commit -m "feat: add cache configuration to template memoization"
```

---

## Task 4: Test with Example Pipeline Generation

**Files:**
- Test: Generate and validate YAML from example pipelines

**Step 1: Generate YAML from python_tasks example**

Run: `RUNNABLE_CONFIGURATION_FILE=examples/configs/argo-config.yaml uv run python examples/01-tasks/python_tasks.py`
Expected: Should generate argo-pipeline.yaml without errors

**Step 2: Check cache configuration in generated YAML**

Run: `grep -A 5 -B 1 "memoize:" argo-pipeline.yaml`
Expected: Should show memoize blocks with cache.configMap.name fields

**Step 3: Validate with Argo linting**

Run: `argo lint argo-pipeline.yaml`
Expected: PASS - workflow validates successfully with Argo

**Step 4: Test multiple examples**

Run: `RUNNABLE_CONFIGURATION_FILE=examples/configs/argo-config.yaml uv run python examples/02-sequential/traversal.py && argo lint argo-pipeline.yaml`
Expected: PASS - sequential example generates valid YAML

Run: `RUNNABLE_CONFIGURATION_FILE=examples/configs/argo-config.yaml uv run python examples/06-parallel/parallel.py && argo lint argo-pipeline.yaml`
Expected: PASS - parallel example generates valid YAML

**Step 5: Commit**

```bash
git add argo-pipeline.yaml
git commit -m "test: verify example pipelines generate valid YAML with cache"
```

---

## Task 5: Verify YAML Structure Matches Argo Spec

**Files:**
- Test: Detailed verification of generated YAML structure

**Step 1: Check exact YAML structure**

Run: `uv run python -c "
import yaml
with open('argo-pipeline.yaml', 'r') as f:
    workflow = yaml.safe_load(f)

# Check that all templates have proper memoize structure
templates = workflow['spec']['templates']
for template in templates:
    if 'memoize' in template:
        memoize = template['memoize']
        print(f\"Template '{template['name']}' memoize:\")
        print(f\"  key: {memoize.get('key')}\")
        if 'cache' in memoize:
            cache = memoize['cache']
            if 'configMap' in cache:
                print(f\"  configMap name: {cache['configMap']['name']}\")
            else:
                print('  ERROR: No configMap in cache')
        else:
            print('  ERROR: No cache in memoize')
        print()
"`
Expected: All templates should show cache.configMap.name structure

**Step 2: Verify ConfigMap naming pattern**

Run: `uv run python -c "
import yaml, re
with open('argo-pipeline.yaml', 'r') as f:
    workflow = yaml.safe_load(f)

cache_names = set()
for template in workflow['spec']['templates']:
    if 'memoize' in template and 'cache' in template['memoize']:
        cache_name = template['memoize']['cache']['configMap']['name']
        cache_names.add(cache_name)

print(f'Found {len(cache_names)} unique cache names:')
for name in cache_names:
    print(f'  {name}')
    assert re.match(r'^runnable-[a-z0-9]{6}$', name), f'Invalid format: {name}'

print('All cache names follow runnable-xxxxxx pattern')
assert len(cache_names) == 1, 'All templates should use same cache name'
print('All templates share same cache name ✓')
"`
Expected: Single valid cache name used across all templates

**Step 3: Commit verification**

```bash
git add -A
git commit -m "test: verify YAML structure matches Argo memoization spec"
```

---

## Task 6: Update Documentation

**Files:**
- Modify: `docs/plans/2025-01-17-argo-retry-design.md:23-24`

**Step 1: Update existing retry design documentation**

```markdown
# docs/plans/2025-01-17-argo-retry-design.md (lines 23-24)

# Change from:
memoize:
  key: "{{workflow.parameters.run_id}}"

# To:
memoize:
  key: "{{workflow.parameters.run_id}}"
  cache:
    configMap:
      name: "runnable-abc123"  # Generated per workflow
```

**Step 2: Commit documentation updates**

```bash
git add docs/plans/2025-01-17-argo-retry-design.md
git commit -m "docs: update retry design with ConfigMap memoization"
```

---

## Task 7: Final Integration Testing

**Files:**
- Test: Complete validation across multiple pipeline types

**Step 1: Test all example pipeline types**

Run: `
for example in examples/01-tasks/python_tasks.py examples/03-parameters/passing_parameters_python.py examples/04-catalog/catalog_python.py; do
  echo "Testing $example"
  RUNNABLE_CONFIGURATION_FILE=examples/configs/argo-config.yaml uv run python $example
  argo lint argo-pipeline.yaml
  echo "✓ $example passed"
done
`
Expected: All example pipelines generate valid Argo YAML

**Step 2: Verify backward compatibility**

Run: `uv run python -c "
# Test that old-style memoize still works
from extensions.pipeline_executor.argo import Memoize
old_memoize = Memoize(key='test-key')
print('Old style still works:', old_memoize.model_dump())

# Test that new style works
from extensions.pipeline_executor.argo import Cache, ConfigMapCache
new_memoize = Memoize(
    key='test-key',
    cache=Cache(config_map=ConfigMapCache(name='runnable-abc123'))
)
print('New style works:', new_memoize.model_dump())
"`
Expected: Both old and new memoization styles work

**Step 3: Final commit**

```bash
git add -A
git commit -m "feat: complete ConfigMap memoization implementation

- Add ConfigMapCache, Cache schema classes
- Generate unique cache names per workflow
- Update template creation with cache config
- Validate against Argo specification
- Maintain backward compatibility

Enables persistent memoization across workflow resubmissions."
```

---

## Summary

This streamlined plan implements ConfigMap memoization by:

1. **Schema Extension**: Add ConfigMapCache and Cache models to support Argo's native format
2. **Cache Generation**: Generate unique `runnable-xxxxxx` names per workflow
3. **Template Updates**: Update both container and nested template creation methods
4. **Real-world Testing**: Generate and lint YAML from actual example pipelines using proper runnable commands
5. **Validation**: Verify structure matches Argo specification exactly

Testing focuses on practical pipeline generation and Argo linting using the correct runnable execution pattern with environment variables.
