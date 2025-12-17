# Argo Workflow Retry Design

## Overview

Extend the existing Argo executor to support workflow-level retry functionality through template memoization, complementing the existing RetryStrategy for step-level retries.

## Architecture

### Two-Tier Retry System
- **Step-level**: Existing RetryStrategy handles retries within a single workflow execution
- **Workflow-level**: New memoization approach handles resubmission of entire workflows

### Core Components

#### Required Parameters
- `run_id`: Now mandatory for all Argo workflows (defaults to workflow UID if not provided)
- `retry_run_id`: Optional parameter indicating this is a retry of another run
- `retry_indicator`: Optional parameter for tracking attempt numbers (defaults to empty string)

#### Template Enhancements
Every generated template receives:
```yaml
memoize:
  key: "{{workflow.parameters.run_id}}"
container:
  env:
  - name: RETRY_RUN_ID
    value: "{{workflow.parameters.run_id}}"
  - name: RETRY_INDICATOR
    value: "{{workflow.parameters.retry_indicator}}"
```

#### Workflow Parameters
```yaml
spec:
  arguments:
    parameters:
    - name: run_id          # Required
    - name: retry_run_id    # Optional, empty string default
    - name: retry_indicator # Optional, empty string default
```

## User Experience

### Initial Execution
```bash
argo submit workflow.yaml -p run_id=my-pipeline-run-001
```

### Retry Execution
```bash
argo resubmit <workflow-name> --memoized \
  -p retry_run_id=my-pipeline-run-001 \
  -p retry_indicator=2
```

## Implementation Strategy

### Integration Approach
- Extend existing `extensions/pipeline_executor/argo.py`
- Maintain backward compatibility
- Leverage existing workflow generation patterns
- Preserve existing RetryStrategy functionality

### Validation
- `retry_run_id` validation occurs naturally when first failed step executes during resubmit
- Existing run log store validation handles invalid retry references
- No additional validation needed at workflow generation time

### Environment Variables
- `RETRY_RUN_ID`: Always contains current `run_id` (provides run context)
- `RETRY_INDICATOR`: Contains attempt chain information for run log tracking

## Benefits

1. **Seamless Integration**: Works with Argo's native resubmit and memoization
2. **Granular Recovery**: Failed steps re-execute, successful steps are skipped
3. **Consistent Interface**: Uses familiar Argo parameter patterns
4. **Backward Compatible**: Existing workflows continue to work
5. **Run Log Continuity**: Proper attempt chaining through retry_indicator

## Technical Notes

- Template memoization uses shared `run_id` key across all templates
- Template names provide natural differentiation within Argo
- Environment variables present in all templates for consistency
- Validation leverages existing run log store mechanisms
