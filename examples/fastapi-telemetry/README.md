# FastAPI Telemetry Streaming Example

This example demonstrates how to integrate runnable with FastAPI using logfire's
built-in FastAPI instrumentation plus real-time SSE streaming.

## Prerequisites

```bash
# Install runnable with telemetry support
uv add runnable[telemetry]

# Install FastAPI with logfire integration
uv add 'logfire[fastapi]' uvicorn
```

## Running the Example

```bash
# Start the FastAPI server
uv run uvicorn examples.fastapi-telemetry.main:app --reload

# In another terminal, trigger a workflow
curl -N -X POST http://localhost:8000/run-workflow \
  -H "Content-Type: application/json" \
  -d '{"pipeline_name": "example"}'

# List available pipelines
curl http://localhost:8000/pipelines
```

## How It Works

1. `logfire.instrument_fastapi(app)` instruments all HTTP requests
2. FastAPI receives a workflow request - this creates a parent span
3. A `Queue` is set up for streaming spans via `set_stream_queue()`
4. The pipeline runs in a thread pool (runnable is synchronous)
5. Pipeline/task spans are automatically children of the HTTP request span
6. `StreamingSpanProcessor` pushes span events to the queue
7. FastAPI streams the events back to the client via SSE

## Span Hierarchy

```
HTTP POST /run-workflow (from logfire.instrument_fastapi)
└── pipeline:
    ├── task:compute
    └── task:finalize
```

## SSE Event Format

Events are JSON objects with a `type` field:

```json
{"type": "span_start", "name": "pipeline:", "span_id": "..."}
{"type": "span_end", "name": "task:compute", "status": "OK", "duration_ms": 1000}
{"type": "complete", "status": "success"}
```
