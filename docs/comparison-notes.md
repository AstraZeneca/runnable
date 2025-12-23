# Runnable vs Kubeflow/Metaflow - Feature Comparison

## Gaps to Discuss

### 1. Web UI/Dashboard
- **Kubeflow**: Full UI for runs, experiments, artifacts
- **Metaflow**: Metaflow UI (optional)
- **Runnable**: Has basic UI, intentionally not invested in
- **Status**: Not a priority

---

### 2. Client API for Inspection
- **Kubeflow**: Yes
- **Metaflow**: Rich API - `Flow`, `Run`, `Step` objects to query past runs
- **Runnable**: None - cannot programmatically list/query runs
- **Status**: TODO - Add simple run log query API (list runs, filter by tag, get artifacts)

---

### 3. Run Tags (Query/Filter)
- **Kubeflow**: Yes
- **Metaflow**: Multiple tags per run + query API
- **Runnable**: Single tag per run, no query API
- **Status**: Single tag sufficient. Query by tag covered in #2 Client API

---

### 4. Automatic Artifact Versioning
- **Kubeflow**: Yes
- **Metaflow**: Per-run versioning of all artifacts
- **Runnable**: Has this - catalog stores artifacts per run_id, run log tracks parameter names + SHA codes. Also supports no-copy mode (SHA tracking without copying)
- **Status**: Not a gap

---

### 5. Resume Command
- **Kubeflow**: Limited
- **Metaflow**: `python flow.py resume` - single command
- **Runnable**: `runnable retry <run_id>` CLI command + environment variable method
- **Status**: DONE - Added CLI command, documented in retry-recovery.md

---

### 6. Native Scheduling
- **Kubeflow**: Recurring runs, triggers
- **Metaflow**: Argo/Airflow/Step Functions integration
- **Runnable**: Argo CronWorkflow support via `cron_schedule` config
- **Status**: DONE - Added `cron_schedule` config option with `schedules` and `timezone`

---

### 7. Decorator-Style Resources
- **Kubeflow**: Component-level
- **Metaflow**: `@resources(gpu=1, memory=16000)` per step
- **Runnable**: Has this via `overrides={}` on tasks - same functionality, different syntax
- **Status**: Not a gap (intentional design)

---

### 8. IDE Debugging Integration
- **Kubeflow**: Limited
- **Metaflow**: PyCharm/VSCode documented
- **Runnable**: Just works - functions are native Python, no special setup needed
- **Status**: DONE - Documented in task-types.md under Python Tasks

---

### 9. Foreach with Dynamic Items
- **Kubeflow**: ParallelFor
- **Metaflow**: `foreach` generates items at runtime
- **Runnable**: Supports this - `iterate_on` can use parameters from previous steps
- **Status**: TODO - Add documentation/example showing dynamic iteration from previous step output

---

## Runnable Strengths (for reference)

- Zero domain code changes
- First-class Notebooks and Shell tasks
- Plugin architecture (everything extensible)
- Mocking executor for testing
- On-failure handlers
- Custom reducers in Map
- Metrics as return types
- Step-level overrides
- Cross-environment retry/debugging
