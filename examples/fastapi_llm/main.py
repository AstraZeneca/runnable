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

from examples.fastapi_llm.pipelines import (
    chat_and_summarize_pipeline,
    chat_and_translate_pipeline,
    chat_conditional_translate_pipeline,
    chat_map_translate_pipeline,
    chat_pipeline,
)
from runnable import names

app = FastAPI(title="Async LLM Streaming Demo")


class ChatRequest(BaseModel):
    prompt: str


class ChatMapRequest(BaseModel):
    prompt: str
    languages: list[str]  # List of languages to translate to


class ChatConditionalRequest(BaseModel):
    prompt: str
    language: str  # Single language preference


@app.get("/")
async def root():
    """Health check and usage info."""
    return {
        "status": "ok",
        "endpoints": {
            "/chat": "POST - Stream LLM response via SSE",
            "/chat-and-summarize": "POST - Stream LLM response and summarize",
            "/chat-and-translate": "POST - Stream LLM response and translate in parallel",
            "/chat-map-translate": "POST - Stream LLM response and translate to multiple languages (map)",
            "/chat-conditional-translate": "POST - Stream LLM response and translate to preferred language (conditional)",
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


@app.post("/chat-and-translate")
async def chat_and_translate(request: ChatRequest):
    """
    Stream LLM response and parallel translations via Server-Sent Events.

    The AsyncPipeline's execute_streaming() yields events from the
    AsyncGenerator function and parallel translation tasks.
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
            pipeline = chat_and_translate_pipeline()
            async for event in pipeline.execute_streaming(run_id=run_id):
                yield f"data: {json.dumps(event)}\n\n"
        finally:
            os.environ.pop("RUNNABLE_PRM_prompt", None)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/chat-map-translate")
async def chat_map_translate(request: ChatMapRequest):
    """
    Stream LLM response and map translations via Server-Sent Events.

    The Map pipeline iterates over the requested languages and translates
    the LLM response to each language.
    """
    # Validate languages
    supported_languages = {"spanish", "french"}
    invalid_languages = set(request.languages) - supported_languages
    if invalid_languages:
        from fastapi import HTTPException

        raise HTTPException(
            status_code=400,
            detail=f"Unsupported languages: {invalid_languages}. Supported: {supported_languages}",
        )

    async def event_stream():
        # Generate unique run_id for this request
        now = datetime.now()
        run_id = (
            f"{names.get_random_name()}-{now.hour:02}{now.minute:02}{now.second:02}"
        )

        # Set parameters as environment variables
        os.environ["RUNNABLE_PRM_prompt"] = request.prompt
        # For list parameters, we need to pass them differently
        # Map expects the parameter to be available as a list in the pipeline context
        import json

        os.environ["RUNNABLE_PRM_languages"] = json.dumps(request.languages)

        try:
            pipeline = chat_map_translate_pipeline()
            async for event in pipeline.execute_streaming(run_id=run_id):
                yield f"data: {json.dumps(event)}\n\n"
        finally:
            os.environ.pop("RUNNABLE_PRM_prompt", None)
            os.environ.pop("RUNNABLE_PRM_languages", None)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/chat-conditional-translate")
async def chat_conditional_translate(request: ChatConditionalRequest):
    """
    Stream LLM response and conditional translation via Server-Sent Events.

    The Conditional pipeline chooses the translation branch based on
    the user's language preference.
    """
    # Validate language
    supported_languages = {"spanish", "french"}
    if request.language not in supported_languages:
        from fastapi import HTTPException

        raise HTTPException(
            status_code=400,
            detail=f"Unsupported language: {request.language}. Supported: {supported_languages}",
        )

    async def event_stream():
        # Generate unique run_id for this request
        now = datetime.now()
        run_id = (
            f"{names.get_random_name()}-{now.hour:02}{now.minute:02}{now.second:02}"
        )

        # Set parameters as environment variables
        os.environ["RUNNABLE_PRM_prompt"] = request.prompt
        os.environ["RUNNABLE_PRM_language"] = request.language

        try:
            pipeline = chat_conditional_translate_pipeline()
            async for event in pipeline.execute_streaming(run_id=run_id):
                yield f"data: {json.dumps(event)}\n\n"
        finally:
            os.environ.pop("RUNNABLE_PRM_prompt", None)
            os.environ.pop("RUNNABLE_PRM_language", None)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
