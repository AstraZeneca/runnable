# Telemetry Design for Runnable

**Date:** 2025-01-30
**Status:** Approved
**Scope:** Phase 1 - Pipeline and Task level telemetry with FastAPI streaming support

## Overview

Add optional telemetry to runnable that emits OpenTelemetry spans during execution. This provides real-time visibility into workflow execution while preserving existing run log functionality for detailed post-hoc review.

## Goals

- Execution times for pipelines and tasks
- Input/output parameters (truncated to ~256 bytes)
- Execution logs as span events
- Function calls as they happen through the workflow
- Real-time streaming to UI via FastAPI SSE
- Self-hosted OpenTelemetry backend support (Jaeger, Tempo, etc.)

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          Your Infrastructure                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐         ┌──────────────┐         ┌──────────────┐    │
│  │   Your UI    │◄────────│   Backend    │◄────────│  Collector   │    │
│  │  (React/etc) │  query  │ (Jaeger/Tempo│  store  │   (OTEL)     │    │
│  └──────────────┘         │   /Grafana)  │         └──────────────┘    │
│         │                 └──────────────┘                ▲            │
│         │                                                 │            │
│         │ SSE stream                           emit spans │            │
│         ▼                                                 │            │
│  ┌──────────────┐         ┌──────────────────────────────┴──┐         │
│  │   FastAPI    │────────▶│         runnable pipeline       │         │
│  │   Service    │ execute │  (logfire-api emits to collector)│         │
│  └──────────────┘         └─────────────────────────────────┘         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

1. **Built into core with automatic kill switch**: Uses `logfire-api` shim package - if `logfire` is not installed, all calls are no-ops. No manual enable/disable needed.

2. **Dual output via custom SpanProcessor**: Spans are always exported to collector (if configured) AND streamed to SSE queue (if FastAPI connection is active).

3. **Additive, not replacing**: Telemetry runs alongside existing run log store. Run logs remain the source of truth for detailed review.

## Span Hierarchy

```
HTTP Request (FastAPI middleware - if present)
└── pipeline:{name}                             ← runnable/context.py
    ├── task:{name}                             ← runnable/tasks.py
    │   └── events: started, completed/failed
    ├── task:{name}
    │   └── events: started, completed/failed
    └── ...
```

### Span Attributes

**Pipeline span:**
- `pipeline_name`: Name of the pipeline
- `run_id`: Unique execution ID
- `executor`: Executor class name

**Task span:**
- `task_name`: Name of the task
- `task_type`: python, notebook, shell
- `command`: Function path or command
- `inputs`: Truncated input parameters (JSON, max 256 bytes)
- `outputs`: Truncated output parameters (JSON, max 256 bytes)
- `status`: success/fail
- `error`: Error message (on failure)

## Implementation

### New File: `runnable/telemetry.py`

```python
"""
Telemetry support for runnable pipelines.

Uses logfire-api for zero-dependency instrumentation.
If logfire is installed, spans are emitted. If not, all calls are no-ops.

For real-time streaming (e.g., FastAPI SSE), use StreamingSpanProcessor.
"""
import json
from contextvars import ContextVar
from queue import Queue
from typing import Any

import logfire_api as logfire

# Optional OTEL imports for streaming processor
try:
    from opentelemetry.sdk.trace import SpanProcessor, ReadableSpan
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False

# Context var for active stream queue (set by FastAPI when SSE is active)
_stream_queue: ContextVar[Queue | None] = ContextVar("stream_queue", default=None)


def truncate_value(value: Any, max_bytes: int = 256) -> str:
    """Truncate serialized value to max_bytes."""
    try:
        serialized = json.dumps(value, default=str)
        if len(serialized) > max_bytes:
            return serialized[:max_bytes - 3] + "..."
        return serialized
    except Exception:
        return f"<unserializable: {type(value).__name__}>"


def set_stream_queue(q: Queue | None):
    """Set the queue for streaming spans (called by FastAPI)."""
    _stream_queue.set(q)


def get_stream_queue() -> Queue | None:
    """Get the current stream queue."""
    return _stream_queue.get()


if OTEL_AVAILABLE:
    class StreamingSpanProcessor(SpanProcessor):
        """
        SpanProcessor that:
        1. Always forwards to base processor (collector export)
        2. Also pushes to stream queue if SSE is active
        """
        def __init__(self, base_processor: SpanProcessor | None = None):
            self.base_processor = base_processor

        def on_start(self, span, parent_context=None):
            if self.base_processor:
                self.base_processor.on_start(span, parent_context)

            q = _stream_queue.get()
            if q is not None:
                q.put_nowait({
                    "type": "span_start",
                    "name": span.name,
                    "span_id": format(span.context.span_id, "016x"),
                })

        def on_end(self, span: ReadableSpan):
            if self.base_processor:
                self.base_processor.on_end(span)

            q = _stream_queue.get()
            if q is not None:
                q.put_nowait({
                    "type": "span_end",
                    "name": span.name,
                    "span_id": format(span.context.span_id, "016x"),
                    "status": span.status.status_code.name,
                    "duration_ms": (span.end_time - span.start_time) / 1_000_000,
                    "attributes": dict(span.attributes) if span.attributes else {},
                })

        def shutdown(self):
            if self.base_processor:
                self.base_processor.shutdown()

        def force_flush(self, timeout_millis=None):
            if self.base_processor:
                self.base_processor.force_flush(timeout_millis)
```

### Integration: `runnable/context.py`

```python
import logfire_api as logfire

class PipelineContext(BaseContext):
    def execute(self, ...):
        with logfire.span(
            "pipeline:{pipeline_name}",
            pipeline_name=self.pipeline.name,
            run_id=self.run_id,
            executor=self.pipeline_executor.__class__.__name__,
        ):
            logfire.info("Pipeline started")

            try:
                self.pipeline_executor._set_up_run_log(exists_ok=False)
                self.pipeline_executor.execute_graph(dag=self.pipeline.dag)

                logfire.info("Pipeline completed", status="success")
            except Exception as e:
                logfire.error("Pipeline failed", error=str(e))
                raise
```

### Integration: `runnable/tasks.py`

```python
import logfire_api as logfire
from runnable.telemetry import truncate_value

class PythonTaskType(BaseTaskType):
    def execute_command(self, map_variable: MapVariableType = None) -> StepAttempt:
        attempt_log = StepAttempt(status=defaults.FAIL, ...)
        task_name = self._context.pipeline_executor._context_node.name

        with logfire.span(
            "task:{task_name}",
            task_name=task_name,
            task_type=self.task_type,
            command=self.command,
        ):
            with self.execution_context(map_variable=map_variable) as params:
                logfire.info("Task started", inputs=truncate_value(params))

                try:
                    # ... existing function call logic ...
                    user_set_parameters = f(**filtered_parameters)

                    if self.returns:
                        # ... existing return handling ...
                        logfire.info("Task completed",
                                    outputs=truncate_value(output_parameters),
                                    status="success")

                    attempt_log.status = defaults.SUCCESS

                except Exception as e:
                    logfire.error("Task failed", error=str(e)[:256])
                    # ... existing error handling ...

        return attempt_log
```

Same pattern applies to `NotebookTaskType` and `ShellTaskType`.

## FastAPI Integration Example

Example code for users to integrate runnable with FastAPI and SSE streaming:

```python
import asyncio
import json
import os
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from runnable import Pipeline
from runnable.telemetry import set_stream_queue

app = FastAPI()
executor = ThreadPoolExecutor(max_workers=4)

# Pipeline registry - knows the design-level config
PIPELINE_REGISTRY = {
    "ml-training": {
        "builder": build_ml_training_pipeline,
        "parameters_file": "configs/ml-training.yaml",
    },
}


class WorkflowRequest(BaseModel):
    pipeline_name: str
    user_parameters: dict = {}  # Runtime overrides -> RUNNABLE_PRM_*


def run_pipeline(pipeline: Pipeline, parameters_file: str | None, user_params: dict):
    """Runs in thread pool (synchronous execution)."""
    # Set user parameters as environment variables
    for key, value in user_params.items():
        env_key = f"RUNNABLE_PRM_{key}"
        if isinstance(value, (dict, list)):
            os.environ[env_key] = json.dumps(value)
        else:
            os.environ[env_key] = str(value)

    try:
        return pipeline.execute(parameters_file=parameters_file)
    finally:
        # Clean up env vars
        for key in user_params:
            env_key = f"RUNNABLE_PRM_{key}"
            if env_key in os.environ:
                del os.environ[env_key]


@app.post("/run-workflow")
async def run_workflow(request: WorkflowRequest):
    span_queue = Queue()
    set_stream_queue(span_queue)

    pipeline_def = PIPELINE_REGISTRY[request.pipeline_name]
    pipeline = pipeline_def["builder"]()
    parameters_file = pipeline_def["parameters_file"]

    async def event_stream():
        loop = asyncio.get_event_loop()

        future = loop.run_in_executor(
            executor,
            run_pipeline,
            pipeline,
            parameters_file,
            request.user_parameters,
        )

        while not future.done():
            try:
                span_data = span_queue.get_nowait()
                yield f"data: {json.dumps(span_data)}\n\n"
            except Empty:
                await asyncio.sleep(0.05)

        # Drain remaining spans
        while True:
            try:
                span_data = span_queue.get_nowait()
                yield f"data: {json.dumps(span_data)}\n\n"
            except Empty:
                break

        try:
            result = future.result()
            yield f"data: {json.dumps({'type': 'complete', 'status': 'success'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'complete', 'status': 'error', 'error': str(e)})}\n\n"
        finally:
            set_stream_queue(None)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )
```

## Dependencies

**pyproject.toml:**

```toml
[project]
dependencies = [
    # ... existing deps ...
    "logfire-api>=1.0.0",  # Zero-dep telemetry shim
]

[project.optional-dependencies]
telemetry = [
    "logfire>=1.0.0",
    "opentelemetry-sdk>=1.20",
    "opentelemetry-exporter-otlp>=1.20",
]
```

**User installation:**

```bash
# Basic - logfire-api is always there, but no-ops without logfire
uv add runnable

# With telemetry support (self-hosted OTEL backend)
uv add runnable[telemetry]

# With Logfire hosted
uv add runnable logfire
```

## Configuration

Standard OTEL environment variables - no runnable-specific config needed:

| Variable | Description | Default |
|----------|-------------|---------|
| `OTEL_EXPORTER_OTLP_ENDPOINT` | Collector endpoint | `http://localhost:4317` |
| `OTEL_SERVICE_NAME` | Service name in traces | `runnable` |
| `LOGFIRE_TOKEN` | If using Logfire hosted | - |

## Files to Modify

| Component | File | Description |
|-----------|------|-------------|
| Telemetry module | `runnable/telemetry.py` | New file - StreamingSpanProcessor, helpers |
| Pipeline span | `runnable/context.py` | Wrap `PipelineContext.execute()` |
| Task spans | `runnable/tasks.py` | Wrap each task type's `execute_command()` |
| Dependency | `pyproject.toml` | Add `logfire-api`, optional `[telemetry]` |
| Example | `examples/fastapi-telemetry/` | FastAPI + SSE streaming example |

## Phase 2 (Future)

- Graph traversal spans (parallel branches, map iterations, conditionals)
- Argo remote execution trace propagation
- Catalog operation spans
