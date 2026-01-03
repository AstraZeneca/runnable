"""Pipeline definitions for FastAPI LLM streaming example."""

from examples.fastapi_llm.llm_mock import mock_llm_stream, mock_summarize
from runnable import AsyncPipeline, AsyncPythonTask


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


def chat_and_summarize_pipeline() -> AsyncPipeline:
    """
    Two-task pipeline that streams LLM response and then summarizes it.

    The first task streams the LLM response, and the second task
    summarizes the full text after streaming is complete.
    """
    return AsyncPipeline(
        name="chat_and_summarize",
        steps=[
            AsyncPythonTask(
                name="llm_response",
                function=mock_llm_stream,
                returns=["full_text"],
            ),
            AsyncPythonTask(
                name="summarize",
                function=mock_summarize,
                returns=["summary"],
            ),
        ],
    )
