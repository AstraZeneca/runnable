"""Pipeline definitions for FastAPI LLM streaming example."""

from examples.fastapi_llm.llm_mock import (
    mock_llm_stream,
    mock_summarize,
    mock_translate_french,
    mock_translate_language,
    mock_translate_spanish,
)
from runnable import AsyncPipeline, AsyncPythonTask, Conditional, Map, Parallel


def chat_pipeline() -> AsyncPipeline:
    """
    Single-task pipeline that streams LLM response.

    The mock_llm_stream function is an AsyncGenerator that yields
    events which flow through the pipeline to the SSE client.
    """
    return AsyncPythonTask(
        name="llm_response",
        function=mock_llm_stream,
        returns=["full_text"],
    ).as_async_pipeline()


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


def chat_and_translate_pipeline() -> AsyncPipeline:
    """
    Three-task pipeline that streams LLM response and then translates it in parallel.

    The first task streams the LLM response, then two parallel tasks
    translate the full text to Spanish and French simultaneously.
    """
    return AsyncPipeline(
        name="chat_and_translate",
        steps=[
            AsyncPythonTask(
                name="llm_response",
                function=mock_llm_stream,
                returns=["full_text"],
            ),
            Parallel(
                name="translate_parallel",
                branches={
                    "spanish": AsyncPythonTask(
                        name="spanish_translation",
                        function=mock_translate_spanish,
                        returns=["spanish_translation"],
                    ).as_async_pipeline(),
                    "french": AsyncPythonTask(
                        name="french_translation",
                        function=mock_translate_french,
                        returns=["french_translation"],
                    ).as_async_pipeline(),
                },
            ),
        ],
    )


def chat_map_translate_pipeline() -> AsyncPipeline:
    """
    Map pipeline that streams LLM response and translates to multiple languages.

    The first task streams the LLM response, then a Map node iterates over
    the requested languages and translates to each one.
    """
    # Translation branch for the Map - translates to whatever language is current
    translation_branch = AsyncPythonTask(
        name="translate_to_language",
        function=mock_translate_language,
        returns=["translation_result"],
    ).as_async_pipeline()

    return AsyncPipeline(
        steps=[
            AsyncPythonTask(
                name="llm_response",
                function=mock_llm_stream,
                returns=["full_text"],
            ),
            Map(
                name="translate_map",
                iterate_on="languages",  # User-provided list of languages
                iterate_as="language",  # Current language in the loop
                branch=translation_branch,
            ),
        ],
    )


def chat_conditional_translate_pipeline() -> AsyncPipeline:
    """
    Conditional pipeline that streams LLM response and translates to user's preferred language.

    The first task streams the LLM response, then a Conditional node
    chooses the translation branch based on user's language preference.
    """
    # Spanish translation branch
    spanish_branch = AsyncPythonTask(
        name="spanish_translation",
        function=mock_translate_spanish,
        returns=["spanish_translation"],
    ).as_async_pipeline()

    # French translation branch
    french_branch = AsyncPythonTask(
        name="french_translation",
        function=mock_translate_french,
        returns=["french_translation"],
    ).as_async_pipeline()

    return AsyncPipeline(
        steps=[
            AsyncPythonTask(
                name="llm_response",
                function=mock_llm_stream,
                returns=["full_text"],
            ),
            Conditional(
                name="translate_conditional",
                parameter="language",  # User's language preference
                branches={
                    "spanish": spanish_branch,
                    "french": french_branch,
                },
            ),
        ],
    )
