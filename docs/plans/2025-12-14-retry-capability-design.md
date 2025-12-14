# Retry Capability Design

## Overview

Add resume-from-failure capability to Runnable pipelines that works across different execution backends, allowing users to restart failed pipeline executions from the point of failure rather than rerunning successful steps.

## Core Requirements

- Resume failed pipeline executions from the failure point
- Skip successful steps, retry only failed and subsequent steps
- Work with both local executors and external systems like Argo workflows
- Allow code changes (functions, notebooks, shell scripts) while preserving DAG structure
- Continue existing run logs rather than creating new runs
- Preserve original execution flow using parameters from the original run

## Design Architecture

### Entry Points by Executor Type

**Local Executors:**
- Auto-generate run_id as normal during regular execution
- Check for `RUNNABLE_RETRY_RUN_ID` environment variable at startup
- If present, validate the specified run exists and failed, then switch to retry mode

**Argo Executors:**
- Remove auto-generation of run_id during transpilation, require explicit run_id
- Use the provided run_id for Argo's memoization system
- run_id and rerun flags are provided during Argo workflow execution (via Argo CLI/UI)
- Argo naturally skips successful steps and resumes at failure points

### DAG Structure Validation

**Structural Hash Components:**
- Node names and their dependencies
- Task types for each node
- Parameter names and return value names (ignoring actual values/content)

**Explicitly Ignored:**
- Pipeline configuration (executor settings, parallelism, etc.)
- Parameter values (use original run's parameters)
- Executable content (functions, notebooks, shell scripts)

This allows:
- Cross-environment retries (local to Argo, different configs)
- Code updates while preserving execution flow
- Same conditional logic and map iterations as original run

### Implementation Components

#### For Local Executors

1. **Startup Validation**
   - Check for `RUNNABLE_RETRY_RUN_ID` environment variable
   - Validate the specified run exists and has failed status
   - Compute and compare DAG structural hash
   - Load original run parameters and metadata

2. **Skip Logic**
   - Each node checks if it has successful attempts in the original run_log
   - Successful nodes are skipped entirely
   - Failed nodes execute normally and get new attempt numbers

3. **Attempt Tracking**
   - Continue the existing run rather than creating new run metadata
   - Only re-executed steps get new attempt records
   - Original successful attempts remain untouched

#### For Argo Executors

1. **Run ID Management**
   - Remove automatic run_id generation during transpilation
   - Transpiled workflow expects run_id as parameter
   - Run_id and rerun flag provided during Argo execution

2. **Memoization Integration**
   - Use run_id as part of Argo memoization keys
   - Let Argo's native memoization handle step skipping
   - No central validation point needed (execution is distributed)

3. **Attempt Tracking**
   - Each task container updates its own attempt number when executing
   - Only executing tasks create new attempt records

### Run Log Behavior

- **Metadata Continuity**: Retry continues the existing run rather than creating new run metadata
- **Attempt Isolation**: Only steps that actually re-execute get new attempt records
- **Parameter Preservation**: Always use parameters from the original run to ensure identical execution flow
- **Status Tracking**: Only increment attempts for steps that execute in the retry

### Validation Logic

**Local Executors:**
- Validate original run exists and failed before starting retry
- Compute and compare structural DAG hash
- Fail fast if structure doesn't match

**Argo Executors:**
- No upfront validation possible (distributed execution)
- Rely on Argo's memoization system for natural handling
- If run_id is new, Argo executes normally and establishes memoization cache

## Benefits

1. **Infrastructure Flexibility**: Same retry works locally or on Kubernetes/Argo
2. **Development Friendly**: Allow code changes while preserving execution flow
3. **Efficient**: Skip successful work, focus only on failures
4. **Consistent**: Same parameters ensure identical branching and iteration
5. **Native Integration**: Leverage each backend's strengths (local control, Argo memoization)

## Usage Examples

**Local Retry:**
```bash
# Original failed run
uv run pipeline.py

# Retry from failure
RUNNABLE_RETRY_RUN_ID=<failed-run-id> uv run pipeline.py
```

**Argo Retry:**
```bash
# Transpile pipeline (generates Argo workflow YAML)
uv run pipeline.py

# Execute with Argo (run_id and rerun provided to Argo)
argo submit workflow.yaml -p run_id=my-pipeline-run-001 -p rerun=true
```
