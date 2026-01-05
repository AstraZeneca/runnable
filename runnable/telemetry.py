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


def truncate_value(value: Any, max_bytes: int = 256) -> Any:
    """
    Truncate a single serialized value to max_bytes.

    Args:
        value: Any JSON-serializable value
        max_bytes: Maximum length for string representation

    Returns:
        The value (possibly truncated if string representation exceeds max_bytes)
    """
    try:
        serialized = json.dumps(value, default=str)
        if len(serialized) > max_bytes:
            # Return truncated string representation
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


# Optional OTEL imports for streaming processor
try:
    from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    SpanProcessor = object  # type: ignore
    ReadableSpan = object  # type: ignore


if OTEL_AVAILABLE:

    class StreamingSpanProcessor(SpanProcessor):
        """
        SpanProcessor that:
        1. Always forwards to base processor (collector export) if provided
        2. Also pushes to stream queue if SSE is active

        This enables dual output: persistent collector storage AND
        real-time streaming to UI.
        """

        def __init__(self, base_processor: Optional[SpanProcessor] = None):
            """
            Initialize the streaming processor.

            Args:
                base_processor: Optional underlying processor for collector export
            """
            self.base_processor = base_processor

        def on_start(self, span, parent_context=None):
            """Called when a span starts."""
            if self.base_processor:
                self.base_processor.on_start(span, parent_context)

            q = _stream_queue.get()
            if q is not None:
                q.put_nowait(
                    {
                        "type": "span_start",
                        "name": span.name,
                        "span_id": format(span.context.span_id, "016x"),
                    }
                )

        def on_end(self, span: ReadableSpan):
            """Called when a span ends."""
            if self.base_processor:
                self.base_processor.on_end(span)

            q = _stream_queue.get()
            if q is not None:
                q.put_nowait(
                    {
                        "type": "span_end",
                        "name": span.name,
                        "span_id": format(span.context.span_id, "016x"),
                        "status": span.status.status_code.name,
                        "duration_ms": (span.end_time - span.start_time)  # type: ignore
                        / 1_000_000,  # ty: ignore
                        "attributes": dict(span.attributes) if span.attributes else {},
                    }
                )

        def shutdown(self):
            """Shutdown the processor."""
            if self.base_processor:
                self.base_processor.shutdown()

        def force_flush(self, timeout_millis=None):
            """Force flush any pending spans."""
            if self.base_processor:
                self.base_processor.force_flush(timeout_millis)  # ty: ignore

else:
    # Placeholder when OTEL is not installed
    StreamingSpanProcessor = None  # type: ignore
