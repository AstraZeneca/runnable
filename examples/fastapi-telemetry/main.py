"""
FastAPI integration example with telemetry streaming using logfire.

Start the server:
    uv run uvicorn examples.fastapi-telemetry.main:app --reload

Test with curl:
    curl -N -X POST http://localhost:8000/run-workflow \
        -H "Content-Type: application/json" \
        -d '{"pipeline_name": "example"}'

Or list available pipelines:
    curl http://localhost:8000/pipelines
"""

import asyncio
import json
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from queue import Empty, Queue

import logfire
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from runnable import Pipeline, set_stream_queue

# Import pipeline registry - use relative import since hyphen in directory name
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pipelines import PIPELINE_REGISTRY

# Configure logfire with console output
logfire.configure(
    send_to_logfire=False,
    console=logfire.ConsoleOptions(
        colors="auto",
        span_style="indented",
        verbose=True,
    ),
)

# Create FastAPI app
app = FastAPI(title="Runnable Telemetry Demo")

# Instrument FastAPI with logfire - creates spans for HTTP requests
logfire.instrument_fastapi(app)

executor = ThreadPoolExecutor(max_workers=4)


class WorkflowRequest(BaseModel):
    pipeline_name: str
    user_parameters: dict = {}


def run_pipeline(pipeline: Pipeline, user_params: dict):
    """Run pipeline in thread pool with user parameters as env vars."""
    # Set user parameters as environment variables (RUNNABLE_PRM_*)
    for key, value in user_params.items():
        env_key = f"RUNNABLE_PRM_{key}"
        if isinstance(value, (dict, list)):
            os.environ[env_key] = json.dumps(value)
        else:
            os.environ[env_key] = str(value)

    try:
        return pipeline.execute()
    finally:
        # Clean up env vars
        for key in user_params:
            env_key = f"RUNNABLE_PRM_{key}"
            if env_key in os.environ:
                del os.environ[env_key]


@app.post("/run-workflow")
async def run_workflow(request: WorkflowRequest):
    """
    Run a workflow with SSE streaming of telemetry.

    The HTTP request span (from logfire.instrument_fastapi) is the parent.
    Pipeline and task spans are automatically children of this request.
    """
    if request.pipeline_name not in PIPELINE_REGISTRY:
        return {"error": f"Unknown pipeline: {request.pipeline_name}"}

    span_queue: Queue = Queue()
    set_stream_queue(span_queue)

    pipeline_def = PIPELINE_REGISTRY[request.pipeline_name]
    pipeline = pipeline_def["builder"]()

    async def event_stream():
        loop = asyncio.get_event_loop()

        # Use partial to bind arguments since run_in_executor
        # only passes positional args after the function
        task_fn = partial(run_pipeline, pipeline, request.user_parameters)
        future = loop.run_in_executor(executor, task_fn)

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
            future.result()
            yield f"data: {json.dumps({'type': 'complete', 'status': 'success'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'complete', 'status': 'error', 'error': str(e)})}\n\n"
        finally:
            set_stream_queue(None)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/pipelines")
async def list_pipelines():
    """List available pipelines."""
    return {"pipelines": list(PIPELINE_REGISTRY.keys())}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
