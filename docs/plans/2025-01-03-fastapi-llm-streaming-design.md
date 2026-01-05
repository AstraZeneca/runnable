# FastAPI LLM Streaming Design

## Overview

Add a new FastAPI example demonstrating native async streaming from pipeline functions to SSE clients, mocking an LLM call with token-by-token streaming.

## Context

- **Existing telemetry example**: Uses sync pipelines + thread pool executor + Queue for SSE streaming
- **New async capability**: `AsyncPipeline` + `AsyncPythonTask` support native async execution
- **Goal**: Stream domain data (LLM tokens) directly from async functions to clients

## Key Design Decisions

### Telemetry vs LLM Streaming: Separate Concerns

| Concern | Purpose | Mechanism | Scope |
|---------|---------|-----------|-------|
| Telemetry | Infrastructure observability | `logfire.span()`, `set_stream_queue()` | All executors |
| LLM Streaming | Domain data to clients | `event_callback` parameter | Local async only |

These are independent paths that do not interfere with each other.

### Streaming Pattern: AsyncGenerator Pass-through

The `execute_streaming()` method returns an `AsyncGenerator` that any async code can consume:

```python
async for event in pipeline.execute_streaming():
    # FastAPI SSE, WebSocket, CLI, Jupyter, tests, etc.
```

This is the standard Python async iteration protocol - not FastAPI-specific.

### Implementation: Callback → Queue → AsyncGenerator Wrapper

Minimal changes (~20 lines) by wrapping existing callback mechanism:

```python
async def execute_streaming(self):
    queue = asyncio.Queue()

    async def run():
        await self.execute(event_callback=queue.put_nowait)
        await queue.put(None)  # sentinel

    task = asyncio.create_task(run())
    while (event := await queue.get()) is not None:
        yield event
```

The queue is an internal implementation detail. The public API is a clean AsyncGenerator.

## Architecture

### New Example Structure

```
examples/fastapi-llm/
├── main.py          # FastAPI app with SSE endpoint
├── pipelines.py     # AsyncPipeline definitions
└── llm_mock.py      # Mock LLM async generator functions
```

### Data Flow

```
Client POST /chat
    ↓
FastAPI async endpoint
    ↓
pipeline.execute_streaming()
    ↓
AsyncPipeline with event_callback
    ↓
AsyncPythonTask runs mock_llm_stream()
    ↓
AsyncGenerator yields {"type": "chunk", "text": "..."}
    ↓
event_callback pushes to internal queue
    ↓
execute_streaming() yields from queue
    ↓
SSE streams to client
```

### Event Structure

Simple typed events (inspired by Anthropic's streaming API):

```python
{"type": "status", "status": "thinking"}   # Progress
{"type": "chunk", "text": "Hello"}         # Token
{"type": "chunk", "text": " world"}        # Token
{"type": "done", "full_text": "Hello world"}  # Completion
```

## Components

### 1. Mock LLM Function (`llm_mock.py`)

```python
async def mock_llm_stream(prompt: str) -> AsyncGenerator[dict, None]:
    yield {"type": "status", "status": "thinking"}
    await asyncio.sleep(0.3)

    response = f"Response to: {prompt[:50]}..."

    yield {"type": "status", "status": "generating"}

    for word in response.split():
        yield {"type": "chunk", "text": word + " "}
        await asyncio.sleep(0.05)

    yield {"type": "done", "full_text": response}
```

### 2. Pipeline Definition (`pipelines.py`)

```python
def chat_pipeline(prompt: str) -> AsyncPipeline:
    return AsyncPipeline(
        name="chat",
        steps=[
            AsyncPythonTask(
                name="llm_response",
                function=mock_llm_stream,
                returns=["full_text"],
            )
        ],
    )
```

### 3. FastAPI Endpoint (`main.py`)

```python
@app.post("/chat")
async def chat(request: ChatRequest):
    async def event_stream():
        pipeline = chat_pipeline(request.prompt)
        async for event in pipeline.execute_streaming():
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
```

### 4. SDK Changes (`runnable/sdk.py`)

Add `execute_streaming()` method to `AsyncPipeline`:

```python
async def execute_streaming(self, ...):
    """Execute pipeline and yield events as AsyncGenerator."""
    queue = asyncio.Queue()

    async def run():
        await self.execute(..., event_callback=queue.put_nowait)
        await queue.put(None)

    task = asyncio.create_task(run())
    while (event := await queue.get()) is not None:
        yield event
```

## Changes Required

### SDK (`runnable/sdk.py`)

- Add `execute_streaming()` method to `AsyncPipeline` class

### Context (`runnable/context.py`)

- Add `event_callback` parameter threading through `AsyncPipelineContext.execute_async()`

### Executor (`extensions/pipeline_executor/__init__.py`)

- Add `event_callback` parameter to async execution methods

### New Files

- `examples/fastapi-llm/main.py`
- `examples/fastapi-llm/pipelines.py`
- `examples/fastapi-llm/llm_mock.py`

## Testing

- Run FastAPI server: `uvicorn examples.fastapi_llm.main:app --reload`
- Test with curl: `curl -N -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{"prompt": "Hello"}'`
- Verify events stream token-by-token

## Non-Goals

- Real LLM integration (mock only)
- Telemetry queue changes (separate concern)
- Non-local executor support for streaming
