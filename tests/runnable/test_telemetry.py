import pytest
from queue import Queue


class TestTruncateValue:
    """Tests for truncate_value helper function."""

    def test_truncate_short_string(self):
        from runnable.telemetry import truncate_value

        result = truncate_value({"key": "value"})
        assert result == '{"key": "value"}'

    def test_truncate_long_string(self):
        from runnable.telemetry import truncate_value

        long_value = {"data": "x" * 500}
        result = truncate_value(long_value, max_bytes=50)
        assert len(result) == 50
        assert result.endswith("...")

    def test_truncate_object_uses_str_fallback(self):
        from runnable.telemetry import truncate_value

        class CustomObject:
            pass

        # Objects are serialized via default=str, so they become string repr
        result = truncate_value(CustomObject())
        assert "CustomObject" in result
        assert result.startswith('"')  # It's a JSON string

    def test_truncate_with_default_max_bytes(self):
        from runnable.telemetry import truncate_value

        # Default is 256 bytes
        long_value = {"data": "x" * 1000}
        result = truncate_value(long_value)
        assert len(result) == 256


class TestStreamQueue:
    """Tests for stream queue context var helpers."""

    def test_set_and_get_stream_queue(self):
        from runnable.telemetry import set_stream_queue, get_stream_queue

        # Initially None
        assert get_stream_queue() is None

        # Set a queue
        q = Queue()
        set_stream_queue(q)
        assert get_stream_queue() is q

        # Clear it
        set_stream_queue(None)
        assert get_stream_queue() is None


class TestStreamingSpanProcessor:
    """Tests for StreamingSpanProcessor."""

    def test_otel_available_flag(self):
        """OTEL_AVAILABLE should reflect whether opentelemetry is installed."""
        from runnable.telemetry import OTEL_AVAILABLE

        # Should be a boolean
        assert isinstance(OTEL_AVAILABLE, bool)

    def test_processor_available_when_otel_installed(self):
        """StreamingSpanProcessor should be available when OTEL is installed."""
        from runnable.telemetry import OTEL_AVAILABLE

        if not OTEL_AVAILABLE:
            pytest.skip("OpenTelemetry not installed")

        from runnable.telemetry import StreamingSpanProcessor

        assert StreamingSpanProcessor is not None

    def test_processor_pushes_to_queue_on_span_end(self):
        """Processor should push span data to queue when SSE is active."""
        from runnable.telemetry import OTEL_AVAILABLE

        if not OTEL_AVAILABLE:
            pytest.skip("OpenTelemetry not installed")

        from opentelemetry.sdk.trace import TracerProvider

        from runnable.telemetry import (
            StreamingSpanProcessor,
            set_stream_queue,
        )

        # Setup
        q = Queue()
        set_stream_queue(q)

        processor = StreamingSpanProcessor(base_processor=None)
        provider = TracerProvider()
        provider.add_span_processor(processor)

        tracer = provider.get_tracer("test")

        # Create a span
        with tracer.start_as_current_span("test-span") as span:
            span.set_attribute("test_attr", "test_value")

        # Verify queue received span data
        assert not q.empty()

        # Should have span_start and span_end
        events = []
        while not q.empty():
            events.append(q.get_nowait())

        assert len(events) == 2
        assert events[0]["type"] == "span_start"
        assert events[0]["name"] == "test-span"
        assert events[1]["type"] == "span_end"
        assert events[1]["name"] == "test-span"
        assert "duration_ms" in events[1]

        # Cleanup
        set_stream_queue(None)

    def test_processor_no_queue_no_error(self):
        """Processor should not error when no queue is set."""
        from runnable.telemetry import OTEL_AVAILABLE

        if not OTEL_AVAILABLE:
            pytest.skip("OpenTelemetry not installed")

        from opentelemetry.sdk.trace import TracerProvider

        from runnable.telemetry import (
            StreamingSpanProcessor,
            set_stream_queue,
        )

        # Ensure no queue is set
        set_stream_queue(None)

        processor = StreamingSpanProcessor(base_processor=None)
        provider = TracerProvider()
        provider.add_span_processor(processor)

        tracer = provider.get_tracer("test")

        # Should not raise
        with tracer.start_as_current_span("test-span"):
            pass
