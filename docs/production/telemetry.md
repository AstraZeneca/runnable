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

### FastAPI Example

See [examples/fastapi-telemetry](https://github.com/AstraZeneca/runnable/tree/main/examples/fastapi-telemetry)
for a complete example of SSE streaming with FastAPI.

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from queue import Queue
from runnable import set_stream_queue

app = FastAPI()

@app.post("/run-workflow")
async def run_workflow(request: WorkflowRequest):
    event_queue = Queue()

    async def run_in_background():
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, execute_pipeline, request.pipeline_name, event_queue
        )

    async def event_stream():
        task = asyncio.create_task(run_in_background())

        while True:
            await asyncio.sleep(0.1)
            while not event_queue.empty():
                event = event_queue.get_nowait()
                yield f"data: {json.dumps(event)}\n\n"

                if event.get("type") == "pipeline_completed":
                    return

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def execute_pipeline(pipeline_name: str, event_queue: Queue):
    set_stream_queue(event_queue)

    event_queue.put({"type": "pipeline_started", "name": pipeline_name})

    pipeline = get_pipeline(pipeline_name)
    pipeline.execute()

    event_queue.put({"type": "pipeline_completed", "status": "success"})

    set_stream_queue(None)
```

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
