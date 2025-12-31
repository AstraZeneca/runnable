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
