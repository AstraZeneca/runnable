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
from datetime import datetime

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from examples.fastapi_llm.pipelines import chat_and_summarize_pipeline, chat_pipeline
from runnable import names

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
        # Generate unique run_id for this request
        now = datetime.now()
        run_id = (
            f"{names.get_random_name()}-{now.hour:02}{now.minute:02}{now.second:02}"
        )

        # Set prompt as environment variable for pipeline parameter
        os.environ["RUNNABLE_PRM_prompt"] = request.prompt

        try:
            pipeline = chat_pipeline()
            async for event in pipeline.execute_streaming(run_id=run_id):
                yield f"data: {json.dumps(event)}\n\n"
        finally:
            os.environ.pop("RUNNABLE_PRM_prompt", None)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/chat-and-summarize")
async def chat_and_summarize(request: ChatRequest):
    """
    Stream LLM response via Server-Sent Events.

    The AsyncPipeline's execute_streaming() yields events from the
    AsyncGenerator function, which are streamed directly to the client.
    """

    async def event_stream():
        # Generate unique run_id for this request
        now = datetime.now()
        run_id = (
            f"{names.get_random_name()}-{now.hour:02}{now.minute:02}{now.second:02}"
        )

        # Set prompt as environment variable for pipeline parameter
        os.environ["RUNNABLE_PRM_prompt"] = request.prompt

        try:
            pipeline = chat_and_summarize_pipeline()
            async for event in pipeline.execute_streaming(run_id=run_id):
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
