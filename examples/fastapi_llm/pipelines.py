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
