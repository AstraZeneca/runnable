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


async def mock_translate_spanish(full_text: str) -> AsyncGenerator[dict, None]:
    """
    Mock Spanish translation function.

    Args:
        full_text: The full text to translate.

    Yields:
        dict: Events in the format:
            - {"type": "status", "status": "translating"} - Processing status
            - {"type": "chunk", "text": "..."} - Translated word chunk
            - {"type": "done", "spanish_translation": "..."} - Complete translation
    """
    await asyncio.sleep(0.3)  # Simulate processing delay
    yield {"type": "status", "status": "translating"}

    # Simple mock translation - replace key English words with Spanish
    spanish_words = {
        "Hello": "Hola", "Hi": "Hola", "hello": "hola", "hi": "hola",
        "how": "cómo", "are": "estás", "you": "tú", "I": "Yo",
        "received": "recibí", "your": "tu", "prompt": "pregunta",
        "Here": "Aquí", "is": "es", "my": "mi", "response": "respuesta",
        "with": "con", "multiple": "múltiples", "words": "palabras",
        "to": "para", "demonstrate": "demostrar", "in": "en", "action": "acción",
        "simulated": "simulada", "streaming": "transmisión", "token": "token"
    }

    words = full_text.split()
    translated_words = []

    for i, word in enumerate(words):
        # Remove punctuation for translation lookup
        clean_word = word.strip('.,!?";:')
        translated = spanish_words.get(clean_word, clean_word)

        # Restore punctuation
        if word != clean_word:
            translated += word[len(clean_word):]

        translated_words.append(translated)
        text = translated + (" " if i < len(words) - 1 else "")
        yield {"type": "chunk", "text": text}
        await asyncio.sleep(0.05)  # 50ms per word

    spanish_translation = " ".join(translated_words)
    yield {"type": "done", "spanish_translation": spanish_translation}


async def mock_translate_french(full_text: str) -> AsyncGenerator[dict, None]:
    """
    Mock French translation function.

    Args:
        full_text: The full text to translate.

    Yields:
        dict: Events in the format:
            - {"type": "status", "status": "translating"} - Processing status
            - {"type": "chunk", "text": "..."} - Translated word chunk
            - {"type": "done", "french_translation": "..."} - Complete translation
    """
    await asyncio.sleep(0.25)  # Simulate processing delay
    yield {"type": "status", "status": "translating"}

    # Simple mock translation - replace key English words with French
    french_words = {
        "Hello": "Bonjour", "Hi": "Salut", "hello": "bonjour", "hi": "salut",
        "how": "comment", "are": "allez", "you": "vous", "I": "Je",
        "received": "reçu", "your": "votre", "prompt": "question",
        "Here": "Voici", "is": "est", "my": "ma", "response": "réponse",
        "with": "avec", "multiple": "multiples", "words": "mots",
        "to": "pour", "demonstrate": "démontrer", "in": "en", "action": "action",
        "simulated": "simulée", "streaming": "diffusion", "token": "jeton"
    }

    words = full_text.split()
    translated_words = []

    for i, word in enumerate(words):
        # Remove punctuation for translation lookup
        clean_word = word.strip('.,!?";:')
        translated = french_words.get(clean_word, clean_word)

        # Restore punctuation
        if word != clean_word:
            translated += word[len(clean_word):]

        translated_words.append(translated)
        text = translated + (" " if i < len(words) - 1 else "")
        yield {"type": "chunk", "text": text}
        await asyncio.sleep(0.04)  # 40ms per word (slightly faster than Spanish)

    french_translation = " ".join(translated_words)
    yield {"type": "done", "french_translation": french_translation}


async def mock_translate_language(full_text: str, language: str) -> AsyncGenerator[dict, None]:
    """
    Generic language translation function that routes to specific language handlers.

    Args:
        full_text: The full text to translate.
        language: The target language ("spanish" or "french").

    Yields:
        dict: Events in the format appropriate for the target language.
    """
    if language == "spanish":
        async for event in mock_translate_spanish(full_text):
            yield event
    elif language == "french":
        async for event in mock_translate_french(full_text):
            yield event
    else:
        # Fallback for unsupported languages
        await asyncio.sleep(0.1)
        yield {"type": "status", "status": f"translating to {language}"}
        yield {"type": "done", f"{language}_translation": f"[Unsupported language: {language}] {full_text}"}
