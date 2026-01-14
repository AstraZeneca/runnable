# 🧪 Async & Streaming Execution (Experimental)

Execute async functions and stream results in real-time with runnable's experimental async capabilities.

!!! warning "Experimental Feature"

    **Async and streaming capabilities are experimental** and under active development. Features may change, and stability is not guaranteed for production use.

## Overview

**Runnable provides experimental async and streaming support** for specialized use cases requiring real-time processing:

- **Batch Processing**: Core production-ready data pipelines with full reproducibility and orchestration
- **Experimental Streaming**: AsyncGenerator support for LLM inference, APIs, and streaming data

**AsyncPipeline and AsyncPythonTask** are experimental features that enable streaming workflows while maintaining runnable's core features: parameter management, catalog system, reproducibility, and multi-environment execution.

!!! warning "Local Execution Only"

    Async capabilities are currently **only supported for local pipeline execution**. AsyncPipeline and AsyncPythonTask cannot be used with containerized (`local-container`) or Kubernetes (`argo`) pipeline executors.

    ```python
    # This works ✓
    pipeline.execute(configuration_file="configs/default.yaml")  # Uses local executor

    # This won't work ✗
    pipeline.execute(configuration_file="configs/local-container.yaml")
    pipeline.execute(configuration_file="configs/argo-config.yaml")
    ```

## When to Use Async Execution

- **LLM and AI model inference** with streaming responses
- **Real-time data processing** that produces intermediate results
- **Long-running async operations** that benefit from streaming feedback
- **FastAPI integration** with Server-Sent Events (SSE)
- **WebSocket streaming** and event-driven architectures

## AsyncPythonTask

Execute async functions with native `await` support and optional streaming:

### Basic Async Function

```python
from runnable import AsyncPythonTask
import asyncio

async def fetch_data():
    await asyncio.sleep(2)  # Simulate async operation
    return {"status": "complete", "data": [1, 2, 3]}

async def main():
    task = AsyncPythonTask(
        name="fetch_task",
        function=fetch_data,
        returns=["result"]
    )

    # Convert to pipeline and execute with streaming
    pipeline = task.as_async_pipeline()

    # Stream events from the async pipeline
    async for event in pipeline.execute_streaming():
        print(f"Event: {event}")

    return pipeline

if __name__ == "__main__":
    asyncio.run(main())
```

### Streaming with AsyncGenerator

For real-time streaming, use `AsyncGenerator` functions that yield events:

```python
from runnable import AsyncPythonTask
import asyncio
from typing import AsyncGenerator

async def mock_llm_stream(prompt: str) -> AsyncGenerator[dict, None]:
    """Stream LLM response token by token."""

    # Initial status
    yield {"type": "status", "status": "thinking"}
    await asyncio.sleep(0.3)

    # Simulate streaming response
    response = f"Response to: {prompt}"
    words = response.split()

    yield {"type": "status", "status": "generating"}

    # Stream word by word
    for word in words:
        yield {"type": "chunk", "text": word + " "}
        await asyncio.sleep(0.05)

    # Final event with complete response
    yield {"type": "done", "full_text": response}

async def main():
    task = AsyncPythonTask(
        name="llm_stream",
        function=mock_llm_stream,
        returns=["full_text"],
        stream_end_type="done"  # Which event contains final values
    )

    pipeline = task.as_async_pipeline()

    # Stream events in real-time
    async for event in pipeline.execute_streaming():
        print(f"Streaming event: {event}")

    return pipeline

if __name__ == "__main__":
    asyncio.run(main())
```

## AsyncPipeline

Execute sequences of async tasks with streaming support:

### Multi-Step Async Pipeline

```python
from runnable import AsyncPipeline, AsyncPythonTask
import asyncio

async def process_input(text: str):
    await asyncio.sleep(0.5)
    return text.upper()

async def analyze_text(processed_text: str):
    await asyncio.sleep(0.3)
    word_count = len(processed_text.split())
    return {"word_count": word_count, "processed": processed_text}

async def main():
    pipeline = AsyncPipeline(
        name="text_processing",
        steps=[
            AsyncPythonTask(
                name="process",
                function=process_input,
                returns=["processed_text"]
            ),
            AsyncPythonTask(
                name="analyze",
                function=analyze_text,
                returns=["analysis"]
            )
        ]
    )

    # Stream events from the multi-step pipeline
    async for event in pipeline.execute_streaming():
        print(f"Pipeline event: {event}")

    return pipeline

if __name__ == "__main__":
    asyncio.run(main())
```

### Streaming Execution

Access events from AsyncGenerator functions in real-time:

```python
from runnable import AsyncPipeline, AsyncPythonTask

async def streaming_analysis(data: str) -> AsyncGenerator[dict, None]:
    yield {"type": "status", "message": "Starting analysis"}

    # Simulate processing steps
    steps = ["tokenizing", "parsing", "analyzing", "summarizing"]
    for step in steps:
        yield {"type": "progress", "step": step}
        await asyncio.sleep(0.2)

    yield {"type": "done", "summary": f"Analyzed: {data[:50]}..."}

async def main():
    pipeline = AsyncPipeline(
        steps=[
            AsyncPythonTask(
                name="stream_analysis",
                function=streaming_analysis,
                returns=["summary"]
            )
        ]
    )

    # Stream real-time events from AsyncGenerator function
    async for event in pipeline.execute_streaming():
        print(f"Real-time event: {event}")

    return pipeline

if __name__ == "__main__":
    asyncio.run(main())
```

## FastAPI Integration

Create streaming APIs with Server-Sent Events (SSE):

### Basic FastAPI Setup

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json
import os

app = FastAPI()

class ChatRequest(BaseModel):
    prompt: str

@app.post("/chat")
async def chat_stream(request: ChatRequest):
    """Stream LLM response via Server-Sent Events."""

    async def event_stream():
        # Set pipeline parameter
        os.environ["RUNNABLE_PRM_prompt"] = request.prompt

        try:
            # Your async pipeline from previous examples
            pipeline = create_chat_pipeline()

            # Stream events directly to client
            async for event in pipeline.execute_streaming():
                yield f"data: {json.dumps(event)}\n\n"
        finally:
            os.environ.pop("RUNNABLE_PRM_prompt", None)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )

def create_chat_pipeline():
    return AsyncPythonTask(
        name="llm_response",
        function=mock_llm_stream,
        returns=["full_text"]
    ).as_async_pipeline()
```

## Advanced Patterns

### Parallel Async Execution

Combine async tasks with parallel execution:

```python
from runnable import AsyncPipeline, AsyncPythonTask, Parallel

async def translate_spanish(text: str) -> AsyncGenerator[dict, None]:
    yield {"type": "status", "message": "Translating to Spanish"}
    await asyncio.sleep(0.5)
    yield {"type": "done", "spanish_translation": f"ES: {text}"}

async def translate_french(text: str) -> AsyncGenerator[dict, None]:
    yield {"type": "status", "message": "Translating to French"}
    await asyncio.sleep(0.4)
    yield {"type": "done", "french_translation": f"FR: {text}"}

async def main():
    pipeline = AsyncPipeline(
        steps=[
            # First generate text
            AsyncPythonTask(
                name="generate",
                function=mock_llm_stream,
                returns=["full_text"]
            ),
            # Then translate in parallel
            Parallel(
                name="translate_parallel",
                branches={
                    "spanish": AsyncPythonTask(
                        name="spanish",
                        function=translate_spanish,
                        returns=["spanish_translation"]
                    ).as_async_pipeline(),
                    "french": AsyncPythonTask(
                        name="french",
                        function=translate_french,
                        returns=["french_translation"]
                    ).as_async_pipeline()
                }
            )
        ]
    )

    # Stream events from LLM generation and parallel translations
    async for event in pipeline.execute_streaming():
        print(f"Parallel execution event: {event}")

    return pipeline

if __name__ == "__main__":
    asyncio.run(main())
```

### Map with Async Streaming

Process multiple items with async streaming:

```python
from runnable import AsyncPipeline, AsyncPythonTask, Map

async def process_item(text: str, language: str) -> AsyncGenerator[dict, None]:
    yield {"type": "status", "message": f"Processing {language}"}
    await asyncio.sleep(0.3)
    yield {"type": "done", "result": f"[{language.upper()}] {text}"}

async def main():
    # Branch that processes one language
    process_branch = AsyncPythonTask(
        name="process_language",
        function=process_item,
        returns=["result"]
    ).as_async_pipeline()

    pipeline = AsyncPipeline(
        steps=[
            AsyncPythonTask(
                name="generate_text",
                function=mock_llm_stream,
                returns=["full_text"]
            ),
            Map(
                name="process_languages",
                iterate_on="languages",  # Parameter: ["spanish", "french", "german"]
                iterate_as="language",   # Current language in loop
                branch=process_branch
            )
        ]
    )

    # Stream events from text generation and map processing
    async for event in pipeline.execute_streaming():
        print(f"Map streaming event: {event}")

    return pipeline

if __name__ == "__main__":
    asyncio.run(main())
```

## Event Types and Stream Configuration

### Standard Event Types

AsyncGenerator functions should yield events with a `type` field:

- **`status`**: Processing status updates
- **`progress`**: Step-by-step progress information
- **`chunk`**: Incremental data (text tokens, partial results)
- **`done`**: Final completion with return values
- **`error`**: Error information

### Stream End Configuration

Control which event contains the final return values:

```python
AsyncPythonTask(
    name="custom_stream",
    function=my_async_generator,
    returns=["result"],
    stream_end_type="complete"  # Look for type="complete" events
)
```

The framework extracts return values from the specified event type, using all keys except `type` as the returned data.

## Best Practices

### AsyncGenerator Patterns

```python
async def well_structured_stream(input_data: str) -> AsyncGenerator[dict, None]:
    """Best practice streaming function structure."""

    try:
        # Always start with status
        yield {"type": "status", "status": "starting", "input": input_data}

        # Processing with progress updates
        steps = ["step1", "step2", "step3"]
        for i, step in enumerate(steps):
            yield {"type": "progress", "step": step, "completed": i, "total": len(steps)}
            await asyncio.sleep(0.1)  # Your actual work here

        # Stream incremental results if applicable
        result_parts = ["part1", "part2", "part3"]
        full_result = []

        for part in result_parts:
            yield {"type": "chunk", "data": part}
            full_result.append(part)
            await asyncio.sleep(0.05)

        # Always end with complete data
        yield {
            "type": "done",
            "final_result": " ".join(full_result),
            "metadata": {"processing_time": "estimated"}
        }

    except Exception as e:
        yield {"type": "error", "error": str(e)}
        raise
```

### Error Handling

Handle errors gracefully in async pipelines:

```python
async def safe_async_operation(data: str) -> AsyncGenerator[dict, None]:
    try:
        yield {"type": "status", "status": "processing"}

        if not data:
            raise ValueError("Empty input data")

        # Your processing
        await asyncio.sleep(0.5)
        result = data.upper()

        yield {"type": "done", "result": result}

    except Exception as e:
        yield {"type": "error", "error": str(e), "failed_input": data}
        # Re-raise to fail the pipeline step
        raise
```

### FastAPI Best Practices

```python
@app.post("/stream-endpoint")
async def stream_endpoint(request: RequestModel):
    async def event_stream():
        # Always use try/finally for cleanup
        os.environ["RUNNABLE_PRM_input"] = request.input

        try:
            pipeline = create_streaming_pipeline()
            async for event in pipeline.execute_streaming(
                run_id=f"req-{datetime.now().isoformat()}"
            ):
                # Validate events before sending
                if isinstance(event, dict) and "type" in event:
                    yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            # Send error event to client
            error_event = {"type": "error", "error": str(e)}
            yield f"data: {json.dumps(error_event)}\n\n"
        finally:
            # Always clean up parameters
            os.environ.pop("RUNNABLE_PRM_input", None)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream"
    )
```

## Complete Example

See the [FastAPI LLM streaming example](https://github.com/AstraZeneca/runnable/tree/main/examples/fastapi_llm) for a complete working implementation with:

- Multiple async pipeline patterns
- FastAPI SSE integration
- Parallel and map async execution
- Conditional async workflows
- Production-ready error handling

```bash
# Run the example
cd examples/fastapi_llm
uv run uvicorn main:app --reload

# Test streaming
curl -N -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, how are you?"}'
```

## Key Differences from Sync Execution

| Feature | Sync Pipeline | AsyncPipeline |
|---------|---------------|---------------|
| **Execution** | `pipeline.execute()` | `async for event in pipeline.execute_streaming()` |
| **Streaming** | Not available | ✅ Native streaming support |
| **Task Type** | `PythonTask` | `AsyncPythonTask` |
| **Functions** | Regular functions | `async def` or `AsyncGenerator` |
| **Use Cases** | Batch processing | Real-time streaming, LLMs, APIs |

The async capabilities enable entirely new patterns like real-time LLM streaming, progressive data processing, and seamless FastAPI integration while maintaining runnable's core principles of reproducibility and configuration management.
