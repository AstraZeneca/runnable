# FastAPI Telemetry Streaming Example

This example demonstrates how to integrate runnable pipelines with FastAPI,
providing real-time Server-Sent Events (SSE) streaming of pipeline execution events.

## Architecture Overview

The system provides dual-output telemetry:

1. **logfire/OpenTelemetry** - Structured spans for observability backends
2. **SSE Queue** - Real-time event streaming for web UI

```
┌─────────────────────────────────────────────────────────────────┐
│                         FastAPI Server                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  POST /run-workflow                                             │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────┐     ┌──────────────┐     ┌─────────────────┐  │
│  │ event_queue │◄────│ Thread Pool  │     │ SSE Response    │  │
│  │ (Queue)     │     │ Executor     │     │ (to client)     │  │
│  └──────┬──────┘     └──────┬───────┘     └───────▲─────────┘  │
│         │                   │                     │             │
│         │            set_stream_queue(q)          │             │
│         │                   │                     │             │
│         │            pipeline.execute()           │             │
│         │                   │                     │             │
│         │            ┌──────┴──────┐              │             │
│         │            │ Task Code   │              │             │
│         │            ├─────────────┤              │             │
│         │            │ logfire     │──► OTEL     │             │
│         │            │ .span()     │   Collector  │             │
│         │            ├─────────────┤              │             │
│         ◄────────────│_emit_event()│              │             │
│         │            └─────────────┘              │             │
│         │                                         │             │
│         └──► event_stream() polls ────────────────┘             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Prerequisites

```bash
# Install dependencies
uv add fastapi uvicorn

# Optional: For OpenTelemetry export
uv add 'logfire[fastapi]'
```

## Running the Example

```bash
# Start the FastAPI server
uv run uvicorn examples.fastapi-telemetry.main:app --reload

# In another terminal, trigger a workflow
curl -N -X POST http://localhost:8000/run-workflow \
  -H "Content-Type: application/json" \
  -d '{"pipeline_name": "example"}'
```

## How It Works

### 1. Pipeline Registry

Pipelines are registered using `functools.partial` to map names to builder functions:

```python
from functools import partial
from pipelines import example_pipeline

PIPELINE_REGISTRY = {
    "example": partial(example_pipeline),
}
```

### 2. Request Flow

1. **POST /run-workflow** receives `{"pipeline_name": "example"}`
2. Creates an `event_queue` for this request
3. Spawns pipeline execution in thread pool (runnable SDK is synchronous)
4. Returns `StreamingResponse` that polls the queue

### 3. Queue Wiring

The queue is set as a thread-local context variable:

```python
def execute_pipeline(pipeline_name, parameters, event_queue):
    # Wire up queue for this thread
    set_stream_queue(event_queue)

    event_queue.put({"type": "pipeline_started", "name": pipeline_name})

    pipeline = PIPELINE_REGISTRY[pipeline_name]()
    pipeline.execute()  # Tasks emit events via _emit_event()

    event_queue.put({"type": "pipeline_completed", "status": "success"})

    set_stream_queue(None)  # Cleanup
```

### 4. Task Event Emission

Tasks in `runnable/tasks.py` emit events to the queue if one is set:

```python
class BaseTaskType:
    def _emit_event(self, event):
        q = get_stream_queue()
        if q is not None:
            q.put_nowait(event)

class PythonTaskType(BaseTaskType):
    def execute_command(self, ...):
        with logfire.span("task:{name}", ...):  # OTEL span
            self._emit_event({"type": "task_started", ...})  # SSE event
            # ... execute ...
            self._emit_event({"type": "task_completed", ...})
```

## SSE Event Format

Events are JSON objects streamed as Server-Sent Events:

```
data: {"type": "pipeline_started", "name": "example"}

data: {"type": "task_started", "name": "examples.fastapi-telemetry.pipelines.compute"}

data: {"type": "task_completed", "name": "examples.fastapi-telemetry.pipelines.compute"}

data: {"type": "task_started", "name": "examples.fastapi-telemetry.pipelines.finalize"}

data: {"type": "task_completed", "name": "examples.fastapi-telemetry.pipelines.finalize"}

data: {"type": "pipeline_completed", "status": "success"}
```

### Event Types

| Type | Description |
|------|-------------|
| `pipeline_started` | Pipeline execution began |
| `task_started` | A task began execution |
| `task_completed` | A task completed successfully |
| `task_error` | A task failed (includes `error` field) |
| `pipeline_completed` | Pipeline finished (includes `status`: success/error) |

## OpenTelemetry Integration (Optional)

To also export spans to an OTEL collector, configure logfire in `main.py`:

```python
import logfire

# Console output only (development)
logfire.configure(
    send_to_logfire=False,
    console=logfire.ConsoleOptions(verbose=True),
)

# Or send to OTEL collector
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

logfire.configure(
    send_to_logfire=False,
    additional_span_processors=[
        BatchSpanProcessor(
            OTLPSpanExporter(endpoint="http://localhost:4317")
        )
    ],
)
```

This gives you structured spans in your observability backend:

```
pipeline:example                              [2.5s]
├── task:pipelines.compute                    [1.0s]
│   ├── inputs: {}
│   └── outputs: {"result": "<object>"}
└── task:pipelines.finalize                   [0.5s]
    ├── inputs: {"result": "<object>"}
    └── outputs: {"final": "<object>"}
```

## Dual Output Summary

| Aspect | logfire/OTEL | SSE Queue |
|--------|--------------|-----------|
| Purpose | Observability/tracing | Real-time UI updates |
| Destination | OTEL collector | HTTP client |
| Persistence | Yes (stored) | No (ephemeral) |
| Structure | Hierarchical spans | Flat events |
| Dependency | Optional | Always available |

Both outputs happen simultaneously from the same task execution code.
