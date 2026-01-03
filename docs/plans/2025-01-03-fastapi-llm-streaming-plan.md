# FastAPI LLM Streaming Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `execute_streaming()` to AsyncPipeline and create a FastAPI example demonstrating LLM token streaming.

**Architecture:** Add `_event_callback` attribute to executor, set at execution start. Task types access callback via executor. Wrap with AsyncGenerator at SDK level.

**Tech Stack:** Python asyncio, FastAPI, SSE (Server-Sent Events), AsyncGenerator

---

## Task 1: Add _event_callback attribute to BasePipelineExecutor

**Files:**

- Modify: `runnable/executor.py:184-188`

**Step 1: Add the _event_callback PrivateAttr**

After line 188 (`_context_node`), add:

```python
_event_callback: Optional[Callable[[dict], None]] = PrivateAttr(default=None)
```

**Step 2: Update imports**

At top of file, update the typing import:

```python
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional
```

**Step 3: Run tests**

Run: `uv run pytest tests/ -k "executor" -v --tb=short`
Expected: Tests pass (no behavior change yet)

**Step 4: Commit**

```bash
git add runnable/executor.py
git commit -m "feat: add _event_callback attribute to BasePipelineExecutor"
```

---

## Task 2: Use executor's event_callback in TaskNode.execute_async

**Files:**

- Modify: `extensions/nodes/task.py:101-144`

**Step 1: Update execute_async to use executor's callback**

Replace the `execute_async` method:

```python
async def execute_async(
    self,
    map_variable: MapVariableType = None,
    attempt_number: int = 1,
    mock: bool = False,
) -> StepLog:
    """Async task execution with fallback to sync."""
    step_log = self._context.run_log_store.get_step_log(
        self._get_step_log_name(map_variable), self._context.run_id
    )

    if not mock:
        # Get event_callback from executor
        event_callback = self._context.pipeline_executor._event_callback

        # Try async first, fall back to sync
        try:
            attempt_log = await self.executable.execute_command_async(
                map_variable=map_variable,
                event_callback=event_callback,
            )
        except NotImplementedError:
            # Task doesn't support async, fall back to sync
            attempt_log = self.executable.execute_command(map_variable=map_variable)

        attempt_log.attempt_number = attempt_number
        attempt_log.retry_indicator = self._context.retry_indicator
    else:
        attempt_log = datastore.StepAttempt(
            status=defaults.SUCCESS,
            start_time=str(datetime.now()),
            end_time=str(datetime.now()),
            attempt_number=attempt_number,
            retry_indicator=self._context.retry_indicator,
        )

    # Add code identities to the attempt
    self._context.pipeline_executor.add_code_identities(
        node=self, attempt_log=attempt_log
    )

    logger.info(f"attempt_log: {attempt_log}")
    logger.info(f"Step {self.name} completed with status: {attempt_log.status}")

    step_log.status = attempt_log.status
    step_log.attempts.append(attempt_log)

    return step_log
```

**Step 2: Run tests**

Run: `uv run pytest tests/ -k "async" -v --tb=short`
Expected: All async tests pass

**Step 3: Commit**

```bash
git add extensions/nodes/task.py
git commit -m "feat: TaskNode.execute_async uses executor's event_callback"
```

---

## Task 3: Add execute_streaming to AsyncPipeline

**Files:**

- Modify: `runnable/sdk.py` (after line 1116, end of AsyncPipeline.execute)

**Step 1: Add asyncio import**

At top of file, add to imports:

```python
import asyncio
```

**Step 2: Add execute_streaming method**

After the `execute` method in `AsyncPipeline` class, add:

```python
async def execute_streaming(
    self,
    configuration_file: str = "",
    run_id: str = "",
    tag: str = "",
    parameters_file: str = "",
    log_level: str = defaults.LOG_LEVEL,
):
    """
    Execute the async pipeline and yield events as an AsyncGenerator.

    This method allows streaming events from AsyncGenerator functions
    directly to the caller, enabling patterns like SSE streaming.

    Usage:
        async for event in pipeline.execute_streaming():
            print(event)

    Yields:
        dict: Events yielded by AsyncGenerator functions in the pipeline.
    """
    from runnable import context

    logger.setLevel(log_level)

    service_configurations = context.ServiceConfigurations(
        configuration_file=configuration_file,
        execution_context=context.ExecutionContext.PIPELINE,
    )

    configurations = {
        "dag": self.return_dag(),
        "parameters_file": parameters_file,
        "tag": tag,
        "run_id": run_id,
        "configuration_file": configuration_file,
        **service_configurations.services,
    }

    run_context = context.AsyncPipelineContext.model_validate(configurations)
    context.set_run_context(run_context)

    # Use asyncio.Queue to bridge callback to AsyncGenerator
    queue: asyncio.Queue = asyncio.Queue()

    # Set the callback on the executor
    run_context.pipeline_executor._event_callback = queue.put_nowait

    async def run_pipeline():
        try:
            await run_context.execute_async()
        finally:
            await queue.put(None)  # Sentinel to signal completion

    # Start pipeline execution in background
    task = asyncio.create_task(run_pipeline())

    # Yield events as they arrive
    while True:
        event = await queue.get()
        if event is None:
            break
        yield event

    # Ensure task completed (will raise if there was an exception)
    await task
```

**Step 3: Run tests**

Run: `uv run pytest tests/test_pipeline_examples.py -k "async" -v`
Expected: Async tests pass

**Step 4: Commit**

```bash
git add runnable/sdk.py
git commit -m "feat: add execute_streaming method to AsyncPipeline"
```

---

## Task 4: Create mock LLM function

**Files:**

- Create: `examples/fastapi_llm/__init__.py`
- Create: `examples/fastapi_llm/llm_mock.py`

**Step 1: Create package init**

Create `examples/fastapi_llm/__init__.py`:

```python
"""FastAPI LLM streaming example."""
```

**Step 2: Create llm_mock.py**

Create `examples/fastapi_llm/llm_mock.py`:

```python
"""Mock LLM functions for demonstrating async streaming."""

import asyncio
from typing import AsyncGenerator


async def mock_llm_stream(prompt: str) -> AsyncGenerator[dict, None]:
    """
    Mock LLM that streams responses token-by-token.

    This demonstrates the AsyncGenerator pattern for streaming events
    through the pipeline to SSE clients.

    Args:
        prompt: The user's input prompt

    Yields:
        dict: Events in the format:
            - {"type": "status", "status": "thinking"} - Processing status
            - {"type": "chunk", "text": "..."} - Token/word chunk
            - {"type": "done", "full_text": "..."} - Completion with full response
    """
    # Simulate initial "thinking" delay
    yield {"type": "status", "status": "thinking"}
    await asyncio.sleep(0.3)

    # Build response based on prompt
    prompt_preview = prompt[:50] + ("..." if len(prompt) > 50 else "")
    response = (
        f"I received your prompt: '{prompt_preview}'. "
        "Here is my simulated response with multiple words to demonstrate "
        "token-by-token streaming in action."
    )

    yield {"type": "status", "status": "generating"}
    await asyncio.sleep(0.1)

    # Stream word by word
    words = response.split()
    for i, word in enumerate(words):
        text = word + (" " if i < len(words) - 1 else "")
        yield {"type": "chunk", "text": text}
        await asyncio.sleep(0.05)  # 50ms per token

    yield {"type": "done", "full_text": response}
```

**Step 3: Commit**

```bash
git add examples/fastapi_llm/
git commit -m "feat: add mock LLM streaming functions"
```

---

## Task 5: Create pipeline definitions

**Files:**

- Create: `examples/fastapi_llm/pipelines.py`

**Step 1: Create pipelines.py**

```python
"""Pipeline definitions for FastAPI LLM streaming example."""

from runnable import AsyncPipeline, AsyncPythonTask

from examples.fastapi_llm.llm_mock import mock_llm_stream


def chat_pipeline() -> AsyncPipeline:
    """
    Single-task pipeline that streams LLM response.

    The mock_llm_stream function is an AsyncGenerator that yields
    events which flow through the pipeline to the SSE client.
    """
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

**Step 2: Commit**

```bash
git add examples/fastapi_llm/pipelines.py
git commit -m "feat: add AsyncPipeline definition for LLM streaming"
```

---

## Task 6: Create FastAPI application

**Files:**

- Create: `examples/fastapi_llm/main.py`

**Step 1: Create main.py**

```python
"""
FastAPI LLM streaming example.

Demonstrates native async streaming from AsyncPipeline functions to SSE clients.

Start the server:
    uv run uvicorn examples.fastapi_llm.main:app --reload

Test with curl:
    curl -N -X POST http://localhost:8000/chat \
        -H "Content-Type: application/json" \
        -d '{"prompt": "Hello, how are you?"}'
"""

import json
import os

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from examples.fastapi_llm.pipelines import chat_pipeline

app = FastAPI(title="Async LLM Streaming Demo")


class ChatRequest(BaseModel):
    prompt: str


@app.get("/")
async def root():
    """Health check and usage info."""
    return {
        "status": "ok",
        "endpoints": {
            "/chat": "POST - Stream LLM response via SSE",
        },
    }


@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Stream LLM response via Server-Sent Events.

    The AsyncPipeline's execute_streaming() yields events from the
    AsyncGenerator function, which are streamed directly to the client.
    """

    async def event_stream():
        # Set prompt as environment variable for pipeline parameter
        os.environ["RUNNABLE_PRM_prompt"] = request.prompt

        try:
            pipeline = chat_pipeline()
            async for event in pipeline.execute_streaming():
                yield f"data: {json.dumps(event)}\n\n"
        finally:
            os.environ.pop("RUNNABLE_PRM_prompt", None)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**Step 2: Commit**

```bash
git add examples/fastapi_llm/main.py
git commit -m "feat: add FastAPI app with SSE streaming endpoint"
```

---

## Task 7: Manual integration test

**Step 1: Start the server**

Run in one terminal:
```bash
uv run uvicorn examples.fastapi_llm.main:app --reload
```
Expected: Server starts on http://localhost:8000

**Step 2: Test streaming endpoint**

In another terminal:
```bash
curl -N -X POST http://localhost:8000/chat \
    -H "Content-Type: application/json" \
    -d '{"prompt": "Hello, how are you?"}'
```

Expected output (streamed line by line):
```
data: {"type": "status", "status": "thinking"}

data: {"type": "status", "status": "generating"}

data: {"type": "chunk", "text": "I "}

data: {"type": "chunk", "text": "received "}

... (more chunks)

data: {"type": "done", "full_text": "I received your prompt: ..."}
```

**Step 3: Stop server**

Ctrl+C to stop the server

---

## Task 8: Add README and final commit

**Files:**

- Create: `examples/fastapi_llm/README.md`

**Step 1: Create README**

```markdown
# FastAPI LLM Streaming Example

Demonstrates native async streaming from `AsyncPipeline` functions to SSE clients.

## Key Concepts

- **AsyncPipeline**: Pipeline that executes async functions natively
- **AsyncPythonTask**: Task wrapper for async functions
- **execute_streaming()**: Returns AsyncGenerator for event streaming
- **AsyncGenerator functions**: Yield events that flow to SSE clients

## Running

```bash
uv run uvicorn examples.fastapi_llm.main:app --reload
```

## Testing

```bash
curl -N -X POST http://localhost:8000/chat \
    -H "Content-Type: application/json" \
    -d '{"prompt": "Hello!"}'
```

## Event Types

- `{"type": "status", "status": "thinking"}` - Processing status
- `{"type": "chunk", "text": "..."}` - Token/word chunk
- `{"type": "done", "full_text": "..."}` - Completion
```

**Step 2: Commit**

```bash
git add examples/fastapi_llm/README.md
git commit -m "docs: add README for FastAPI LLM streaming example"
```

---

## Task 9: Run all tests and push

**Step 1: Run all tests**

```bash
uv run pytest tests/ -v --tb=short
```
Expected: All tests pass

**Step 2: Push**

```bash
git push origin async-execution
```
Expected: Push succeeds
