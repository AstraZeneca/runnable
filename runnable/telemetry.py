"""
Telemetry support for runnable pipelines.

Uses logfire-api for zero-dependency instrumentation.
If logfire is installed, spans are emitted. If not, all calls are no-ops.

For real-time streaming (e.g., FastAPI SSE), use StreamingSpanProcessor.
"""

import json
from contextvars import ContextVar
from queue import Queue
from typing import Any, Optional

import logfire_api as logfire  # noqa: F401 - re-exported for convenience

# Context var for active stream queue (set by FastAPI when SSE is active)
_stream_queue: ContextVar[Optional[Queue]] = ContextVar("stream_queue", default=None)


def truncate_value(value: Any, max_bytes: int = 256) -> str:
    """
    Truncate serialized value to max_bytes.

    Args:
        value: Any JSON-serializable value
        max_bytes: Maximum length of the returned string

    Returns:
        JSON string, truncated with "..." if too long
    """
    try:
        serialized = json.dumps(value, default=str)
        if len(serialized) > max_bytes:
            return serialized[: max_bytes - 3] + "..."
        return serialized
    except Exception:
        return f"<unserializable: {type(value).__name__}>"


def set_stream_queue(q: Optional[Queue]) -> None:
    """
    Set the queue for streaming spans.

    Called by FastAPI endpoint to enable real-time span streaming.

    Args:
        q: Queue to push span data to, or None to disable streaming
    """
    _stream_queue.set(q)


def get_stream_queue() -> Optional[Queue]:
    """
    Get the current stream queue.

    Returns:
        The active Queue if SSE streaming is enabled, None otherwise
    """
    return _stream_queue.get()
