"""
FastAPI integration example with telemetry streaming.

Start the server:
    uv run uvicorn examples.fastapi-telemetry.main:app --reload

Test with curl:
    curl -N -X POST http://localhost:8000/run-workflow \
        -H "Content-Type: application/json" \
        -d '{"pipeline_name": "example"}'
"""

import asyncio
import json
import sys
from pathlib import Path
from queue import Empty, Queue

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Import pipeline registry
sys.path.insert(0, str(Path(__file__).parent))
from pipelines import PIPELINE_REGISTRY

app = FastAPI(title="Runnable Telemetry Demo")


class WorkflowRequest(BaseModel):
    pipeline_name: str
    user_parameters: dict = {}


@app.get("/pipelines")
async def list_pipelines():
    """List available pipelines."""
    return {"pipelines": list(PIPELINE_REGISTRY.keys())}


@app.post("/run-workflow")
async def run_workflow(request: WorkflowRequest):
    """
    Run a workflow with SSE streaming of events.

    The pipeline runs in a background thread while events are
    streamed back to the client via SSE.
    """
    event_queue: Queue = Queue()

    async def run_pipeline_in_background():
        """Run the pipeline in a thread pool executor."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            execute_pipeline,
            request.pipeline_name,
            request.user_parameters,
            event_queue,
        )

    async def event_stream():
        # Start pipeline execution in background
        task = asyncio.create_task(run_pipeline_in_background())

        # Stream events from queue
        while True:
            try:
                # Check queue with small timeout
                await asyncio.sleep(0.1)
                while not event_queue.empty():
                    event = event_queue.get_nowait()
                    yield f"data: {json.dumps(event)}\n\n"

                    # Exit on completion
                    if event.get("type") == "pipeline_completed":
                        return

            except Empty:
                continue

            # Check if task failed
            if task.done() and task.exception():
                yield f"data: {json.dumps({'type': 'error', 'message': str(task.exception())})}\n\n"
                return

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def execute_pipeline(pipeline_name: str, parameters: dict, event_queue: Queue):
    """
    Execute a pipeline and push events to the queue.

    This runs in a thread pool executor (synchronous context).
    """
    import os

    from runnable import set_stream_queue

    # Set queue for this thread - task code will push events here
    set_stream_queue(event_queue)

    # Expose parameters as RUNNABLE_PRM_* environment variables
    param_env_keys = []
    for key, value in parameters.items():
        env_key = f"RUNNABLE_PRM_{key}"
        param_env_keys.append(env_key)
        if isinstance(value, (dict, list)):
            os.environ[env_key] = json.dumps(value)
        else:
            os.environ[env_key] = str(value)

    event_queue.put({"type": "pipeline_started", "name": pipeline_name})

    try:
        # Get pipeline builder from registry
        pipeline_builder = PIPELINE_REGISTRY.get(pipeline_name)
        if not pipeline_builder:
            event_queue.put({
                "type": "pipeline_completed",
                "status": "error",
                "error": f"Unknown pipeline: {pipeline_name}"
            })
            return

        # Build and execute the pipeline
        pipeline = pipeline_builder()
        pipeline.execute()

        event_queue.put({"type": "pipeline_completed", "status": "success"})

    except Exception as e:
        event_queue.put({
            "type": "pipeline_completed",
            "status": "error",
            "error": str(e)
        })
    finally:
        # Cleanup: remove parameter env vars and stream queue
        for env_key in param_env_keys:
            os.environ.pop(env_key, None)
        set_stream_queue(None)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
