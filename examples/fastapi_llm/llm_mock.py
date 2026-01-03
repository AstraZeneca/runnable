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


async def mock_summarize(full_text: str) -> AsyncGenerator[dict, None]:
    """
    Mock summarization function.

    Args:
        full_text: The full text to summarize.

    Returns:
        str: A mock summary of the input text.
    """
    await asyncio.sleep(0.2)  # Simulate processing delay
    yield {"type": "status", "status": "summarizing"}
    yield {"type": "done", "summary": f"Summary: {full_text[:50]}..."}
