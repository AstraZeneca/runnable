# Telemetry & Observability

Runnable provides built-in telemetry for monitoring pipeline and task execution.
The telemetry system supports two output modes:

1. **OpenTelemetry/logfire** - Structured spans for observability backends (Jaeger, Datadog, etc.)
2. **SSE Streaming** - Real-time events for web UI integration

## How It Works

Every task execution emits telemetry:

```python
# In runnable/tasks.py
class PythonTaskType(BaseTaskType):
    def execute_command(self, ...):
        # OpenTelemetry span
        with logfire.span("task:{task_name}", task_name=self.command):
            logfire.info("Task started", inputs=...)

            # SSE event (if queue is set)
            self._emit_event({"type": "task_started", "name": self.command})

            # ... execute task ...

            self._emit_event({"type": "task_completed", "name": self.command})
            logfire.info("Task completed", outputs=...)
```

## OpenTelemetry Integration

Runnable uses [logfire-api](https://github.com/pydantic/logfire) as a zero-dependency
shim. When logfire is not installed, all telemetry calls are no-ops.

### Installation

```bash
uv add logfire
```

### Configuration

There are two ways to configure telemetry:

1. **Environment Variables** (recommended for containers)
2. **Programmatic** (for local development or custom setups)

## Environment Variable Configuration

For containerized execution (local-container, Argo, Kubernetes), set environment
variables in your container or config. Runnable auto-configures logfire at import time.

| Variable | Description |
|----------|-------------|
| `RUNNABLE_TELEMETRY_CONSOLE` | Set to `true` for console span output |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OTLP collector endpoint (e.g., `http://collector:4317`) |
| `LOGFIRE_TOKEN` | Logfire cloud token (enables cloud export) |

Example in a container config:

```yaml
# argo-config.yaml or k8s config
executor:
  type: local-container
  config:
    docker_image: my-image:latest
    environment:
      RUNNABLE_TELEMETRY_CONSOLE: "true"
      OTEL_EXPORTER_OTLP_ENDPOINT: "http://otel-collector:4317"
```

When the container runs `runnable execute_single_node ...`, telemetry is
automatically configured before any task execution.

## Programmatic Configuration

For local development or custom setups, configure logfire in your application:

```python
import logfire

# Console output (development)
logfire.configure(
    send_to_logfire=False,
    console=logfire.ConsoleOptions(
        colors="auto",
        span_style="indented",
        verbose=True,
    ),
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

### Span Hierarchy

Pipeline execution creates nested spans:

```
pipeline:example                              [2.5s]
├── task:my_module.compute                    [1.0s]
│   ├── inputs: {"x": 10}
│   └── outputs: {"result": 20}
└── task:my_module.finalize                   [0.5s]
    ├── inputs: {"result": 20}
    └── outputs: {"final": "Result: 20"}
```

## SSE Streaming for Web UI

For real-time updates in a web interface, use the SSE streaming mechanism.

### Setting Up the Queue

```python
from queue import Queue
from runnable import set_stream_queue

# Create queue for this request
event_queue = Queue()

# Wire up the queue (must be in same thread as pipeline execution)
set_stream_queue(event_queue)

# Execute pipeline - tasks will emit events to queue
pipeline.execute()

# Cleanup
set_stream_queue(None)
```

### Event Types

| Type | Fields | Description |
|------|--------|-------------|
| `pipeline_started` | `name` | Pipeline execution began |
| `task_started` | `name` | A task began execution |
| `task_completed` | `name` | A task completed successfully |
| `task_error` | `name`, `error` | A task failed |
| `pipeline_completed` | `status` | Pipeline finished (success/error) |

### Local Telemetry Example

See [examples/telemetry-local/simple_telemetry_test.py](https://github.com/AstraZeneca/runnable/tree/main/examples/telemetry-local)
for a complete working example of local telemetry with console output.

```python
from runnable import Pipeline, PythonTask, pickled
import logfire

def step_one(x: int = 5) -> int:
    result = x * 2
    return result

def step_two(doubled: int) -> str:
    result = f"Final result: {doubled}"
    return result

def main():
    pipeline = Pipeline(
        steps=[
            PythonTask(
                function=step_one,
                name="step_one",
                returns=[pickled("doubled")]
            ),
            PythonTask(
                function=step_two,
                name="step_two",
                returns=[pickled("final_result")]
            ),
        ]
    )

    pipeline.execute()
    return pipeline

if __name__ == "__main__":
    # Configure console telemetry output
    logfire.configure(
        send_to_logfire=False,
        console=logfire.ConsoleOptions(
            colors="auto",
            span_style="indented",
            verbose=True,
        ),
    )

    main()
```

Run with: `uv run examples/telemetry-local/simple_telemetry_test.py`

### FastAPI Integration

For FastAPI applications, you can integrate telemetry with both traditional batch pipelines and async streaming workflows:

**Traditional Pipeline with SSE Events**:
```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from queue import Queue
from runnable import set_stream_queue
import json

@app.post("/run-pipeline")
def run_pipeline_with_events():
    event_queue = Queue()
    set_stream_queue(event_queue)

    def event_stream():
        try:
            # Execute pipeline in background
            pipeline.execute()

            # Stream events to client
            while not event_queue.empty():
                event = event_queue.get_nowait()
                yield f"data: {json.dumps(event)}\n\n"
        finally:
            set_stream_queue(None)

    return StreamingResponse(event_stream(), media_type="text/event-stream")
```

**Async Streaming Workflows**:
For real-time streaming with AsyncPipeline, see [Async & Streaming](../advanced-patterns/async-streaming.md) and the [FastAPI LLM examples](https://github.com/AstraZeneca/runnable/tree/main/examples/fastapi_llm).

## Telemetry Attributes

Each task span includes:

| Attribute | Description |
|-----------|-------------|
| `task_name` | Full module path (e.g., `my_module.compute`) |
| `task_type` | Type of task (`python`, `notebook`, `shell`) |
| `inputs` | Serialized input parameters (truncated to 256 bytes) |
| `outputs` | Serialized output parameters (truncated to 256 bytes) |
| `status` | Execution status (`success` or error message) |

## Dual Output Architecture

The telemetry system supports simultaneous output to both OpenTelemetry and SSE:

```
Task Execution
      │
      ├──► logfire.span()     ──► OpenTelemetry Collector ──► Jaeger/Datadog/etc.
      │
      └──► _emit_event()      ──► Queue ──► SSE ──► Web UI
```

Both outputs happen from the same code path, ensuring consistency between
observability data and real-time UI updates.
