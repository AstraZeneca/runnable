# Telemetry Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add OpenTelemetry-based telemetry to runnable for pipeline and task observability with FastAPI SSE streaming support.

**Architecture:** Uses `logfire-api` as a zero-dependency shim that no-ops when logfire is not installed. A custom `StreamingSpanProcessor` enables dual output: collector export AND real-time SSE streaming to UI when FastAPI is the caller.

**Tech Stack:** logfire-api, opentelemetry-sdk (optional), opentelemetry-exporter-otlp (optional)

**Design Document:** `docs/plans/2025-01-30-telemetry-design.md`

---

## Task 1: Add logfire-api Dependency

**Files:**
- Modify: `pyproject.toml:10-20` (dependencies section)

**Step 1: Add logfire-api to core dependencies**

In `pyproject.toml`, add `logfire-api` to the dependencies list:

```toml
dependencies = [
    "pydantic>=2.10.3",
    "ruamel-yaml>=0.18.6",
    "stevedore>=5.4.0",
    "rich>=13.9.4",
    "dill>=0.3.9",
    "setuptools>=75.6.0",
    "python-dotenv>=1.0.1",
    "typer>=0.17.3",
    "cloudpathlib>=0.20.0",
    "logfire-api>=2.0.0",
]
```

**Step 2: Add telemetry optional dependency group**

After the existing optional dependencies (around line 38), add:

```toml
telemetry = [
    "logfire>=2.0.0",
    "opentelemetry-sdk>=1.20.0",
    "opentelemetry-exporter-otlp>=1.20.0",
]
```

**Step 3: Sync dependencies**

Run: `uv sync --all-extras`
Expected: Dependencies install successfully

**Step 4: Verify logfire-api is available**

Run: `uv run python -c "import logfire_api; print('OK')"`
Expected: `OK`

**Step 5: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "feat: add logfire-api dependency for telemetry support"
```

---

## Task 2: Create Telemetry Module with Helpers

**Files:**
- Create: `runnable/telemetry.py`
- Test: `tests/runnable/test_telemetry.py`

**Step 1: Write the failing tests for helpers**

Create `tests/runnable/test_telemetry.py`:

```python
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

    def test_truncate_unserializable(self):
        from runnable.telemetry import truncate_value

        class Unserializable:
            pass

        result = truncate_value(Unserializable())
        assert "<unserializable:" in result

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
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/runnable/test_telemetry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'runnable.telemetry'`

**Step 3: Write the telemetry module**

Create `runnable/telemetry.py`:

```python
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
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/runnable/test_telemetry.py -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add runnable/telemetry.py tests/runnable/test_telemetry.py
git commit -m "feat: add telemetry module with helpers"
```

---

## Task 3: Add StreamingSpanProcessor

**Files:**
- Modify: `runnable/telemetry.py`
- Modify: `tests/runnable/test_telemetry.py`

**Step 1: Write the failing tests for StreamingSpanProcessor**

Add to `tests/runnable/test_telemetry.py`:

```python
class TestStreamingSpanProcessor:
    """Tests for StreamingSpanProcessor."""

    def test_processor_available_when_otel_installed(self):
        """StreamingSpanProcessor should be available when OTEL is installed."""
        try:
            from runnable.telemetry import StreamingSpanProcessor, OTEL_AVAILABLE
            if OTEL_AVAILABLE:
                assert StreamingSpanProcessor is not None
        except ImportError:
            pytest.skip("OpenTelemetry not installed")

    def test_processor_pushes_to_queue_on_span_end(self):
        """Processor should push span data to queue when SSE is active."""
        try:
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.trace import StatusCode
        except ImportError:
            pytest.skip("OpenTelemetry not installed")

        from runnable.telemetry import (
            StreamingSpanProcessor,
            set_stream_queue,
            get_stream_queue,
            OTEL_AVAILABLE,
        )

        if not OTEL_AVAILABLE:
            pytest.skip("OpenTelemetry not installed")

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
        try:
            from opentelemetry.sdk.trace import TracerProvider
        except ImportError:
            pytest.skip("OpenTelemetry not installed")

        from runnable.telemetry import (
            StreamingSpanProcessor,
            set_stream_queue,
            OTEL_AVAILABLE,
        )

        if not OTEL_AVAILABLE:
            pytest.skip("OpenTelemetry not installed")

        # Ensure no queue is set
        set_stream_queue(None)

        processor = StreamingSpanProcessor(base_processor=None)
        provider = TracerProvider()
        provider.add_span_processor(processor)

        tracer = provider.get_tracer("test")

        # Should not raise
        with tracer.start_as_current_span("test-span"):
            pass
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/runnable/test_telemetry.py::TestStreamingSpanProcessor -v`
Expected: FAIL with `ImportError: cannot import name 'StreamingSpanProcessor'`

**Step 3: Add StreamingSpanProcessor to telemetry module**

Add to `runnable/telemetry.py` at the end:

```python
# Optional OTEL imports for streaming processor
try:
    from opentelemetry.sdk.trace import SpanProcessor, ReadableSpan
    from opentelemetry.trace import StatusCode

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
                        "duration_ms": (span.end_time - span.start_time) / 1_000_000,
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
                self.base_processor.force_flush(timeout_millis)

else:
    # Placeholder when OTEL is not installed
    StreamingSpanProcessor = None  # type: ignore
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/runnable/test_telemetry.py -v`
Expected: All tests PASS (some may skip if OTEL not installed)

**Step 5: Install telemetry extras and re-run**

Run: `uv sync --extra telemetry && uv run pytest tests/runnable/test_telemetry.py -v`
Expected: All tests PASS including StreamingSpanProcessor tests

**Step 6: Commit**

```bash
git add runnable/telemetry.py tests/runnable/test_telemetry.py
git commit -m "feat: add StreamingSpanProcessor for SSE streaming"
```

---

## Task 4: Add Telemetry to PipelineContext.execute()

**Files:**
- Modify: `runnable/context.py:1-10` (imports)
- Modify: `runnable/context.py:416-451` (PipelineContext.execute method)
- Modify: `tests/runnable/test_context.py`

**Step 1: Write the failing test**

Add to `tests/runnable/test_context.py`:

```python
class TestPipelineContextTelemetry:
    """Test telemetry integration in PipelineContext."""

    def test_execute_emits_pipeline_span(self, monkeypatch):
        """Test that execute() emits a pipeline span via logfire."""
        from unittest.mock import MagicMock, patch

        # Mock logfire.span
        mock_span_ctx = MagicMock()
        mock_span_ctx.__enter__ = MagicMock(return_value=None)
        mock_span_ctx.__exit__ = MagicMock(return_value=None)

        with patch("runnable.context.logfire") as mock_logfire:
            mock_logfire.span.return_value = mock_span_ctx
            mock_logfire.info = MagicMock()
            mock_logfire.error = MagicMock()

            # We can't easily test full execution, but we can verify the import works
            from runnable.context import PipelineContext
            import runnable.context as ctx_module

            # Verify logfire is imported in the module
            assert hasattr(ctx_module, "logfire")
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/runnable/test_context.py::TestPipelineContextTelemetry -v`
Expected: FAIL with `AttributeError: module 'runnable.context' has no attribute 'logfire'`

**Step 3: Add logfire import to context.py**

At the top of `runnable/context.py`, after the existing imports (around line 10), add:

```python
import logfire_api as logfire
```

**Step 4: Modify PipelineContext.execute() method**

Replace the `execute` method in `PipelineContext` (lines 416-451) with:

```python
    def execute(self):
        assert self.dag is not None

        with logfire.span(
            "pipeline:{pipeline_name}",
            pipeline_name=self.dag.name if hasattr(self.dag, "name") else "unnamed",
            run_id=self.run_id,
            executor=self.pipeline_executor.__class__.__name__,
        ):
            logfire.info("Pipeline execution started")

            console.print("Working with context:")
            console.print(run_context)
            console.rule(style="[dark orange]")

            # Prepare for graph execution
            if self.pipeline_executor._should_setup_run_log_at_traversal:
                self.pipeline_executor._set_up_run_log(exists_ok=False)

            try:
                self.pipeline_executor.execute_graph(dag=self.dag)
                if not self.pipeline_executor._should_setup_run_log_at_traversal:
                    # non local executors just traverse the graph and do nothing
                    logfire.info("Pipeline submitted", status="submitted")
                    return {}

                run_log = run_context.run_log_store.get_run_log_by_id(
                    run_id=run_context.run_id, full=False
                )

                if run_log.status == defaults.SUCCESS:
                    console.print(
                        "Pipeline executed successfully!", style=defaults.success_style
                    )
                    logfire.info("Pipeline completed", status="success")
                else:
                    console.print("Pipeline execution failed.", style=defaults.error_style)
                    logfire.error("Pipeline failed", status="failed")
                    raise exceptions.ExecutionFailedError(run_context.run_id)
            except Exception as e:  # noqa: E722
                console.print(e, style=defaults.error_style)
                logfire.error("Pipeline failed with exception", error=str(e)[:256])
                raise

            if self.pipeline_executor._should_setup_run_log_at_traversal:
                return run_context.run_log_store.get_run_log_by_id(
                    run_id=run_context.run_id
                )
```

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/runnable/test_context.py::TestPipelineContextTelemetry -v`
Expected: PASS

**Step 6: Run all context tests to ensure no regressions**

Run: `uv run pytest tests/runnable/test_context.py -v`
Expected: All tests PASS

**Step 7: Commit**

```bash
git add runnable/context.py tests/runnable/test_context.py
git commit -m "feat: add telemetry spans to PipelineContext.execute"
```

---

## Task 5: Add Telemetry to JobContext.execute()

**Files:**
- Modify: `runnable/context.py:503-523` (JobContext.execute method)

**Step 1: Modify JobContext.execute() method**

Replace the `execute` method in `JobContext` (lines 503-523) with:

```python
    def execute(self):
        with logfire.span(
            "job:{job_name}",
            job_name=self.job_definition_file,
            run_id=self.run_id,
            executor=self.job_executor.__class__.__name__,
        ):
            logfire.info("Job execution started")

            console.print("Working with context:")
            console.print(run_context)
            console.rule(style="[dark orange]")

            try:
                self.job_executor.submit_job(
                    job=self.job, catalog_settings=self.catalog_settings
                )
                logfire.info("Job submitted", status="submitted")
            except Exception as e:
                logfire.error("Job failed", error=str(e)[:256])
                raise
            finally:
                console.print(f"Job execution completed for run id: {self.run_id}")

            logger.info(
                "Executing the job from the user. We are still in the caller's compute environment"
            )

            if self.job_executor._should_setup_run_log_at_traversal:
                return run_context.run_log_store.get_run_log_by_id(
                    run_id=run_context.run_id
                )
```

**Step 2: Run all context tests**

Run: `uv run pytest tests/runnable/test_context.py -v`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add runnable/context.py
git commit -m "feat: add telemetry spans to JobContext.execute"
```

---

## Task 6: Add Telemetry to PythonTaskType

**Files:**
- Modify: `runnable/tasks.py:1-31` (imports)
- Modify: `runnable/tasks.py:332-422` (PythonTaskType.execute_command)
- Modify: `tests/runnable/test_tasks.py`

**Step 1: Write the failing test**

Add to `tests/runnable/test_tasks.py`:

```python
class TestTaskTelemetry:
    """Test telemetry integration in task types."""

    def test_python_task_emits_span(self, mock_context):
        """Test that PythonTaskType emits telemetry span."""
        from unittest.mock import MagicMock, patch

        mock_span_ctx = MagicMock()
        mock_span_ctx.__enter__ = MagicMock(return_value=None)
        mock_span_ctx.__exit__ = MagicMock(return_value=None)

        with patch("runnable.tasks.logfire") as mock_logfire:
            mock_logfire.span.return_value = mock_span_ctx
            mock_logfire.info = MagicMock()
            mock_logfire.error = MagicMock()

            # Setup mock for _context_node
            mock_ctx, _ = mock_context
            mock_node = MagicMock()
            mock_node.name = "test_task"
            mock_executor = MagicMock()
            mock_executor._context_node = mock_node
            mock_ctx.pipeline_executor = mock_executor

            task = PythonTaskType(
                command="math.sqrt",
                returns=[TaskReturns(name="result", kind="json")]
            )

            with patch("runnable.tasks.importlib.import_module") as mock_import:
                mock_module = MagicMock()
                mock_module.sqrt = lambda x: x**0.5
                mock_import.return_value = mock_module

                task.execute_command()

            # Verify span was created
            mock_logfire.span.assert_called_once()
            call_args = mock_logfire.span.call_args
            assert "task:" in call_args[0][0]
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/runnable/test_tasks.py::TestTaskTelemetry -v`
Expected: FAIL (logfire not imported in tasks.py)

**Step 3: Add imports to tasks.py**

At the top of `runnable/tasks.py`, after the existing imports (around line 30), add:

```python
import logfire_api as logfire
from runnable.telemetry import truncate_value
```

**Step 4: Modify PythonTaskType.execute_command()**

Replace the `execute_command` method in `PythonTaskType` (lines 332-422) with:

```python
    def execute_command(
        self,
        map_variable: MapVariableType = None,
    ) -> StepAttempt:
        """Execute the notebook as defined by the command."""
        attempt_log = StepAttempt(
            status=defaults.FAIL,
            start_time=str(datetime.now()),
            retry_indicator=self._context.retry_indicator,
        )

        # Get task name from context
        task_name = "unknown"
        if hasattr(self._context, "pipeline_executor") and hasattr(
            self._context.pipeline_executor, "_context_node"
        ):
            node = self._context.pipeline_executor._context_node
            if node:
                task_name = node.name

        with logfire.span(
            "task:{task_name}",
            task_name=task_name,
            task_type=self.task_type,
            command=self.command,
        ):
            with (
                self.execution_context(map_variable=map_variable) as params,
                self.expose_secrets() as _,
            ):
                logfire.info(
                    "Task started",
                    inputs=truncate_value(
                        {k: v.get_value() if hasattr(v, "get_value") else str(v) for k, v in params.items()}
                    ),
                )

                module, func = utils.get_module_and_attr_names(self.command)
                sys.path.insert(0, os.getcwd())  # Need to add the current directory to path
                imported_module = importlib.import_module(module)
                f = getattr(imported_module, func)

                try:
                    try:
                        filtered_parameters = parameters.filter_arguments_for_func(
                            f, params.copy(), map_variable
                        )
                        logger.info(
                            f"Calling {func} from {module} with {filtered_parameters}"
                        )
                        with redirect_output(console=task_console) as (
                            buffer,
                            stderr_buffer,
                        ):
                            user_set_parameters = f(
                                **filtered_parameters
                            )  # This is a tuple or single value
                    except Exception as e:
                        raise exceptions.CommandCallError(
                            f"Function call: {self.command} did not succeed.\n"
                        ) from e
                    finally:
                        attempt_log.input_parameters = params.copy()
                        if map_variable:
                            attempt_log.input_parameters.update(
                                {
                                    k: JsonParameter(value=v, kind="json")
                                    for k, v in map_variable.items()
                                }
                            )

                    if self.returns:
                        if not isinstance(user_set_parameters, tuple):  # make it a tuple
                            user_set_parameters = (user_set_parameters,)

                        if len(user_set_parameters) != len(self.returns):
                            raise ValueError(
                                "Returns task signature does not match the function returns"
                            )

                        output_parameters: Dict[str, Parameter] = {}
                        metrics: Dict[str, Parameter] = {}

                        for i, task_return in enumerate(self.returns):
                            output_parameter = task_return_to_parameter(
                                task_return=task_return,
                                value=user_set_parameters[i],
                            )

                            if task_return.kind == "metric":
                                metrics[task_return.name] = output_parameter

                            param_name = task_return.name
                            if map_variable:
                                for _, v in map_variable.items():
                                    param_name = f"{v}_{param_name}"

                            output_parameters[param_name] = output_parameter

                        attempt_log.output_parameters = output_parameters
                        attempt_log.user_defined_metrics = metrics
                        params.update(output_parameters)

                        logfire.info(
                            "Task completed",
                            outputs=truncate_value(
                                {k: v.get_value() if hasattr(v, "get_value") else str(v) for k, v in output_parameters.items()}
                            ),
                            status="success",
                        )
                    else:
                        logfire.info("Task completed", status="success")

                    attempt_log.status = defaults.SUCCESS
                except Exception as _e:
                    msg = f"Call to the function {self.command} did not succeed.\n"
                    attempt_log.message = msg
                    task_console.print_exception(show_locals=False)
                    task_console.log(_e, style=defaults.error_style)
                    logfire.error("Task failed", error=str(_e)[:256])

        attempt_log.end_time = str(datetime.now())

        return attempt_log
```

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/runnable/test_tasks.py::TestTaskTelemetry -v`
Expected: PASS

**Step 6: Run all task tests**

Run: `uv run pytest tests/runnable/test_tasks.py -v`
Expected: All tests PASS

**Step 7: Commit**

```bash
git add runnable/tasks.py tests/runnable/test_tasks.py
git commit -m "feat: add telemetry spans to PythonTaskType"
```

---

## Task 7: Add Telemetry to NotebookTaskType

**Files:**
- Modify: `runnable/tasks.py:506-617` (NotebookTaskType.execute_command)

**Step 1: Modify NotebookTaskType.execute_command()**

Replace the `execute_command` method in `NotebookTaskType` with telemetry-instrumented version. The key changes are:

1. Add task name extraction at the start
2. Wrap execution in `logfire.span()`
3. Add `logfire.info()` for start/completion
4. Add `logfire.error()` for failures

```python
    def execute_command(
        self,
        map_variable: MapVariableType = None,
    ) -> StepAttempt:
        """Execute the python notebook as defined by the command."""
        attempt_log = StepAttempt(
            status=defaults.FAIL,
            start_time=str(datetime.now()),
            retry_indicator=self._context.retry_indicator,
        )

        # Get task name from context
        task_name = "unknown"
        if hasattr(self._context, "pipeline_executor") and hasattr(
            self._context.pipeline_executor, "_context_node"
        ):
            node = self._context.pipeline_executor._context_node
            if node:
                task_name = node.name

        with logfire.span(
            "task:{task_name}",
            task_name=task_name,
            task_type=self.task_type,
            command=self.command,
        ):
            try:
                import ploomber_engine as pm
                from ploomber_engine.ipython import PloomberClient

                notebook_output_path = self.get_notebook_output_path(
                    map_variable=map_variable
                )

                with (
                    self.execution_context(
                        map_variable=map_variable, allow_complex=False
                    ) as params,
                    self.expose_secrets() as _,
                ):
                    logfire.info(
                        "Notebook task started",
                        inputs=truncate_value(
                            {k: v.get_value() if hasattr(v, "get_value") else str(v) for k, v in params.items()}
                        ),
                        notebook=self.command,
                    )

                    attempt_log.input_parameters = params.copy()
                    copy_params = copy.deepcopy(params)

                    if map_variable:
                        for key, value in map_variable.items():
                            copy_params[key] = JsonParameter(kind="json", value=value)

                    # Remove any {v}_unreduced parameters from the parameters
                    unprocessed_params = [
                        k for k, v in copy_params.items() if not v.reduced
                    ]

                    for key in list(copy_params.keys()):
                        if any(key.endswith(f"_{k}") for k in unprocessed_params):
                            del copy_params[key]

                    notebook_params = {k: v.get_value() for k, v in copy_params.items()}

                    ploomber_optional_args = self.optional_ploomber_args

                    kwds = {
                        "input_path": self.command,
                        "output_path": notebook_output_path,
                        "parameters": notebook_params,
                        "log_output": True,
                        "progress_bar": False,
                    }
                    kwds.update(ploomber_optional_args)

                    with redirect_output(console=task_console) as (buffer, stderr_buffer):
                        pm.execute_notebook(**kwds)

                    context.run_context.catalog.put(name=notebook_output_path)

                    client = PloomberClient.from_path(path=notebook_output_path)
                    namespace = client.get_namespace()

                    output_parameters: Dict[str, Parameter] = {}
                    try:
                        for task_return in self.returns:
                            param_name = Template(task_return.name).safe_substitute(
                                map_variable  # type: ignore
                            )

                            if map_variable:
                                for _, v in map_variable.items():
                                    param_name = f"{v}_{param_name}"

                            output_parameters[param_name] = task_return_to_parameter(
                                task_return=task_return,
                                value=namespace[task_return.name],
                            )
                    except PicklingError as e:
                        logger.exception("Notebooks cannot return objects")
                        logger.exception(e)
                        logfire.error("Notebook pickling error", error=str(e)[:256])
                        raise

                    if output_parameters:
                        attempt_log.output_parameters = output_parameters
                        params.update(output_parameters)
                        logfire.info(
                            "Notebook task completed",
                            outputs=truncate_value(
                                {k: v.get_value() if hasattr(v, "get_value") else str(v) for k, v in output_parameters.items()}
                            ),
                            status="success",
                        )
                    else:
                        logfire.info("Notebook task completed", status="success")

                    attempt_log.status = defaults.SUCCESS

            except (ImportError, Exception) as e:
                msg = (
                    f"Call to the notebook command {self.command} did not succeed.\n"
                    "Ensure that you have installed runnable with notebook extras"
                )
                logger.exception(msg)
                logger.exception(e)
                logfire.error("Notebook task failed", error=str(e)[:256])

                attempt_log.status = defaults.FAIL

        attempt_log.end_time = str(datetime.now())

        return attempt_log
```

**Step 2: Run task tests**

Run: `uv run pytest tests/runnable/test_tasks.py -v`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add runnable/tasks.py
git commit -m "feat: add telemetry spans to NotebookTaskType"
```

---

## Task 8: Add Telemetry to ShellTaskType

**Files:**
- Modify: `runnable/tasks.py:686-822` (ShellTaskType.execute_command)

**Step 1: Modify ShellTaskType.execute_command()**

Add telemetry instrumentation following the same pattern as PythonTaskType:

```python
    def execute_command(
        self,
        map_variable: MapVariableType = None,
    ) -> StepAttempt:
        """Execute the shell command as defined by the command."""
        attempt_log = StepAttempt(
            status=defaults.FAIL,
            start_time=str(datetime.now()),
            retry_indicator=self._context.retry_indicator,
        )

        # Get task name from context
        task_name = "unknown"
        if hasattr(self._context, "pipeline_executor") and hasattr(
            self._context.pipeline_executor, "_context_node"
        ):
            node = self._context.pipeline_executor._context_node
            if node:
                task_name = node.name

        subprocess_env = {}

        # Expose RUNNABLE environment variables to be passed to the subprocess.
        for key, value in os.environ.items():
            if key.startswith("RUNNABLE_"):
                subprocess_env[key] = value

        # Expose map variable as environment variables
        if map_variable:
            for key, value in map_variable.items():  # type: ignore
                subprocess_env[key] = str(value)

        # Expose secrets as environment variables
        if self.secrets:
            for key in self.secrets:
                secret_value = context.run_context.secrets.get(key)
                subprocess_env[key] = secret_value

        with logfire.span(
            "task:{task_name}",
            task_name=task_name,
            task_type=self.task_type,
            command=self.command[:100],  # Truncate long commands
        ):
            try:
                with self.execution_context(
                    map_variable=map_variable, allow_complex=False
                ) as params:
                    logfire.info(
                        "Shell task started",
                        inputs=truncate_value(
                            {k: v.get_value() if hasattr(v, "get_value") else str(v) for k, v in params.items()}
                        ),
                    )

                    subprocess_env.update({k: v.get_value() for k, v in params.items()})

                    attempt_log.input_parameters = params.copy()
                    # Json dumps all runnable environment variables
                    for key, value in subprocess_env.items():
                        if isinstance(value, str):
                            continue
                        subprocess_env[key] = json.dumps(value)

                    collect_delimiter = "=== COLLECT ==="

                    command = (
                        self.command.strip() + f" && echo '{collect_delimiter}'  && env"
                    )
                    logger.info(f"Executing shell command: {command}")

                    capture = False
                    return_keys = {x.name: x for x in self.returns}

                    proc = subprocess.Popen(
                        command,
                        shell=True,
                        env=subprocess_env,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                    result = proc.communicate()
                    logger.debug(result)
                    logger.info(proc.returncode)

                    if proc.returncode != 0:
                        msg = ",".join(result[1].split("\n"))
                        task_console.print(msg, style=defaults.error_style)
                        raise exceptions.CommandCallError(msg)

                    # for stderr
                    for line in result[1].split("\n"):
                        if line.strip() == "":
                            continue
                        task_console.print(line, style=defaults.warning_style)

                    output_parameters: Dict[str, Parameter] = {}
                    metrics: Dict[str, Parameter] = {}

                    # only from stdout
                    for line in result[0].split("\n"):
                        if line.strip() == "":
                            continue

                        logger.info(line)
                        task_console.print(line)

                        if line.strip() == collect_delimiter:
                            # The lines from now on should be captured
                            capture = True
                            continue

                        if capture:
                            key, value = line.strip().split("=", 1)
                            if key in return_keys:
                                task_return = return_keys[key]

                                try:
                                    value = json.loads(value)
                                except json.JSONDecodeError:
                                    value = value

                                output_parameter = task_return_to_parameter(
                                    task_return=task_return,
                                    value=value,
                                )

                                if task_return.kind == "metric":
                                    metrics[task_return.name] = output_parameter

                                param_name = task_return.name
                                if map_variable:
                                    for _, v in map_variable.items():
                                        param_name = f"{v}_{param_name}"

                                output_parameters[param_name] = output_parameter

                        attempt_log.output_parameters = output_parameters
                        attempt_log.user_defined_metrics = metrics
                        params.update(output_parameters)

                    if output_parameters:
                        logfire.info(
                            "Shell task completed",
                            outputs=truncate_value(
                                {k: v.get_value() if hasattr(v, "get_value") else str(v) for k, v in output_parameters.items()}
                            ),
                            status="success",
                        )
                    else:
                        logfire.info("Shell task completed", status="success")

                    attempt_log.status = defaults.SUCCESS
            except exceptions.CommandCallError as e:
                msg = f"Call to the command {self.command} did not succeed"
                logger.exception(msg)
                logger.exception(e)

                task_console.log(msg, style=defaults.error_style)
                task_console.log(e, style=defaults.error_style)
                logfire.error("Shell task failed", error=str(e)[:256])

                attempt_log.status = defaults.FAIL

        attempt_log.end_time = str(datetime.now())
        return attempt_log
```

**Step 2: Run all task tests**

Run: `uv run pytest tests/runnable/test_tasks.py -v`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add runnable/tasks.py
git commit -m "feat: add telemetry spans to ShellTaskType"
```

---

## Task 9: Export Telemetry in Package __init__

**Files:**
- Modify: `runnable/__init__.py`

**Step 1: Check current __init__.py exports**

Run: `cat runnable/__init__.py | head -50`
Review what's currently exported.

**Step 2: Add telemetry exports**

Add to `runnable/__init__.py`:

```python
from runnable.telemetry import (
    set_stream_queue,
    get_stream_queue,
    truncate_value,
)

# Conditionally export StreamingSpanProcessor
try:
    from runnable.telemetry import StreamingSpanProcessor, OTEL_AVAILABLE
except ImportError:
    StreamingSpanProcessor = None
    OTEL_AVAILABLE = False
```

**Step 3: Verify import works**

Run: `uv run python -c "from runnable import set_stream_queue, get_stream_queue; print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add runnable/__init__.py
git commit -m "feat: export telemetry helpers from runnable package"
```

---

## Task 10: Run Full Test Suite

**Files:** None (verification only)

**Step 1: Run all tests**

Run: `uv run pytest tests/ -v --tb=short`
Expected: All tests PASS

**Step 2: Run a simple pipeline example to verify telemetry doesn't break anything**

Run: `uv run python examples/01-tasks/python_tasks.py`
Expected: Pipeline executes successfully (telemetry no-ops since logfire not configured)

**Step 3: Install telemetry extras and verify**

Run: `uv sync --extra telemetry && uv run python -c "from runnable.telemetry import StreamingSpanProcessor; print('StreamingSpanProcessor available')"`
Expected: `StreamingSpanProcessor available`

---

## Task 11: Local Telemetry Test

**Files:**
- Create: `examples/telemetry-local/simple_telemetry_test.py`

**Goal:** Validate telemetry works end-to-end without FastAPI complexity.

**Step 1: Create the local test script**

Create `examples/telemetry-local/simple_telemetry_test.py`:

```python
"""
Simple local test to verify telemetry is working.

Run with:
    uv run python examples/telemetry-local/simple_telemetry_test.py
"""

import logfire

from runnable import Pipeline, PythonTask, pickled


# Configure logfire to output to console
logfire.configure(
    send_to_logfire=False,  # Don't send to cloud
    console=logfire.ConsoleOptions(
        colors="auto",
        span_style="indented",
        include_timestamps=True,
        verbose=True,
    ),
)


def step_one(x: int) -> int:
    """First step - doubles the input."""
    print(f"Step one: received x={x}")
    result = x * 2
    print(f"Step one: returning {result}")
    return result


def step_two(doubled: int) -> str:
    """Second step - formats the result."""
    print(f"Step two: received doubled={doubled}")
    result = f"Final result: {doubled}"
    print(f"Step two: returning '{result}'")
    return result


def main():
    """Run a simple pipeline and observe telemetry output."""
    print("=" * 60)
    print("Running pipeline with telemetry enabled")
    print("You should see spans for pipeline and each task")
    print("=" * 60)
    print()

    pipeline = Pipeline(
        steps=[
            PythonTask(
                function=step_one,
                name="step_one",
                returns=[pickled("doubled")],
            ),
            PythonTask(
                function=step_two,
                name="step_two",
                returns=[pickled("final_result")],
            ),
        ]
    )

    # Execute with initial parameter
    result = pipeline.execute(parameters_file=None)

    print()
    print("=" * 60)
    print("Pipeline completed!")
    print(f"Result: {result}")
    print("=" * 60)

    return result


if __name__ == "__main__":
    main()
```

**Step 2: Create a parameters file for testing**

Create `examples/telemetry-local/params.yaml`:

```yaml
x: 5
```

**Step 3: Run the local test**

Run: `uv run python examples/telemetry-local/simple_telemetry_test.py`

Expected output should show:
- Console output with colored/indented spans
- `pipeline:*` span wrapping the execution
- `task:step_one` span with inputs/outputs
- `task:step_two` span with inputs/outputs
- Timestamps and durations

**Step 4: Verify spans are visible**

If telemetry is working correctly, you should see output like:
```
pipeline:unnamed
├── Pipeline execution started
├── task:step_one
│   ├── Task started inputs={"x": 5}
│   └── Task completed outputs={"doubled": 10} status=success
├── task:step_two
│   ├── Task started inputs={"doubled": 10}
│   └── Task completed outputs={"final_result": "..."} status=success
└── Pipeline completed status=success
```

**Step 5: Commit**

```bash
git add examples/telemetry-local/
git commit -m "feat: add local telemetry test example"
```

---

## Task 12: Create FastAPI Example (Placeholder)

> **Note:** This task will be updated to leverage logfire's built-in FastAPI integration.
> For now, create a basic example that can be enhanced later.

**Files:**
- Create: `examples/fastapi-telemetry/README.md`
- Create: `examples/fastapi-telemetry/main.py`
- Create: `examples/fastapi-telemetry/pipelines.py`

**Step 1: Create example directory and README**

Create `examples/fastapi-telemetry/README.md`:

```markdown
# FastAPI Telemetry Streaming Example

This example demonstrates how to integrate runnable with FastAPI for real-time
telemetry streaming via Server-Sent Events (SSE).

## Prerequisites

```bash
# Install runnable with telemetry support
uv add runnable[telemetry]

# Install FastAPI
uv add fastapi uvicorn
```

## Running the Example

```bash
# Start the FastAPI server
uv run uvicorn examples.fastapi-telemetry.main:app --reload

# In another terminal, trigger a workflow
curl -X POST http://localhost:8000/run-workflow \
  -H "Content-Type: application/json" \
  -d '{"pipeline_name": "example", "user_parameters": {"x": 10}}'
```

## How It Works

1. FastAPI receives a workflow request
2. A `Queue` is set up for streaming spans via `set_stream_queue()`
3. The pipeline runs in a thread pool (runnable is synchronous)
4. `StreamingSpanProcessor` pushes span events to the queue
5. FastAPI streams the events back to the client via SSE
```

**Step 2: Create pipelines.py**

Create `examples/fastapi-telemetry/pipelines.py`:

```python
"""Example pipelines for FastAPI integration."""

from runnable import Pipeline, PythonTask, pickled


def compute(x: int) -> int:
    """Simple compute function."""
    import time
    time.sleep(1)  # Simulate work
    return x * 2


def finalize(result: int) -> str:
    """Finalize the result."""
    import time
    time.sleep(0.5)
    return f"Result: {result}"


def build_example_pipeline() -> Pipeline:
    """Build an example pipeline."""
    return Pipeline(
        steps=[
            PythonTask(
                function=compute,
                name="compute",
                returns=[pickled("result")],
            ),
            PythonTask(
                function=finalize,
                name="finalize",
                returns=[pickled("final")],
            ),
        ]
    )


# Pipeline registry
PIPELINE_REGISTRY = {
    "example": {
        "builder": build_example_pipeline,
        "parameters_file": None,
    },
}
```

**Step 3: Create main.py**

Create `examples/fastapi-telemetry/main.py`:

```python
"""FastAPI integration example with telemetry streaming."""

import asyncio
import json
import os
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from runnable import Pipeline, set_stream_queue

# Import pipeline registry
from .pipelines import PIPELINE_REGISTRY

app = FastAPI(title="Runnable Telemetry Demo")
executor = ThreadPoolExecutor(max_workers=4)


class WorkflowRequest(BaseModel):
    pipeline_name: str
    user_parameters: dict = {}


def run_pipeline(pipeline: Pipeline, parameters_file: str | None, user_params: dict):
    """Run pipeline in thread pool with user parameters as env vars."""
    # Set user parameters as environment variables
    for key, value in user_params.items():
        env_key = f"RUNNABLE_PRM_{key}"
        if isinstance(value, (dict, list)):
            os.environ[env_key] = json.dumps(value)
        else:
            os.environ[env_key] = str(value)

    try:
        return pipeline.execute(parameters_file=parameters_file)
    finally:
        # Clean up env vars
        for key in user_params:
            env_key = f"RUNNABLE_PRM_{key}"
            if env_key in os.environ:
                del os.environ[env_key]


@app.post("/run-workflow")
async def run_workflow(request: WorkflowRequest):
    """Run a workflow with SSE streaming of telemetry."""
    if request.pipeline_name not in PIPELINE_REGISTRY:
        return {"error": f"Unknown pipeline: {request.pipeline_name}"}

    span_queue = Queue()
    set_stream_queue(span_queue)

    pipeline_def = PIPELINE_REGISTRY[request.pipeline_name]
    pipeline = pipeline_def["builder"]()
    parameters_file = pipeline_def["parameters_file"]

    async def event_stream():
        loop = asyncio.get_event_loop()

        future = loop.run_in_executor(
            executor,
            run_pipeline,
            pipeline,
            parameters_file,
            request.user_parameters,
        )

        while not future.done():
            try:
                span_data = span_queue.get_nowait()
                yield f"data: {json.dumps(span_data)}\n\n"
            except Empty:
                await asyncio.sleep(0.05)

        # Drain remaining spans
        while True:
            try:
                span_data = span_queue.get_nowait()
                yield f"data: {json.dumps(span_data)}\n\n"
            except Empty:
                break

        try:
            result = future.result()
            yield f"data: {json.dumps({'type': 'complete', 'status': 'success'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'complete', 'status': 'error', 'error': str(e)})}\n\n"
        finally:
            set_stream_queue(None)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/pipelines")
async def list_pipelines():
    """List available pipelines."""
    return {"pipelines": list(PIPELINE_REGISTRY.keys())}
```

**Step 4: Commit**

```bash
git add examples/fastapi-telemetry/
git commit -m "feat: add FastAPI telemetry streaming example"
```

---

## Task 13: Final Integration Test and Commit

**Step 1: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: All tests PASS

**Step 2: Run pre-commit checks**

Run: `uv run pre-commit run --all-files`
Expected: All checks PASS

**Step 3: Create final commit with all changes**

If any files weren't committed individually:

```bash
git add -A
git status
# Review changes, then:
git commit -m "feat: complete telemetry implementation phase 1"
```

---

## Summary

After completing all tasks, the following will be implemented:

1. **`logfire-api`** dependency added to core
2. **`runnable/telemetry.py`** module with:
   - `truncate_value()` helper
   - `set_stream_queue()` / `get_stream_queue()` for SSE streaming
   - `StreamingSpanProcessor` for dual output (collector + SSE)
3. **`runnable/context.py`** instrumented:
   - `PipelineContext.execute()` emits pipeline span
   - `JobContext.execute()` emits job span
4. **`runnable/tasks.py`** instrumented:
   - `PythonTaskType.execute_command()` emits task span
   - `NotebookTaskType.execute_command()` emits task span
   - `ShellTaskType.execute_command()` emits task span
5. **Local telemetry test** in `examples/telemetry-local/` - validates telemetry works with console output
6. **FastAPI example** in `examples/fastapi-telemetry/` (placeholder - will leverage logfire's FastAPI integration)
7. **Tests** for all telemetry functionality

## Task Order

| Task | Description | Status |
|------|-------------|--------|
| 1 | Add logfire-api dependency | Pending |
| 2 | Create telemetry module with helpers | Pending |
| 3 | Add StreamingSpanProcessor | Pending |
| 4 | Add telemetry to PipelineContext.execute() | Pending |
| 5 | Add telemetry to JobContext.execute() | Pending |
| 6 | Add telemetry to PythonTaskType | Pending |
| 7 | Add telemetry to NotebookTaskType | Pending |
| 8 | Add telemetry to ShellTaskType | Pending |
| 9 | Export telemetry in package __init__ | Pending |
| 10 | Run full test suite | Pending |
| 11 | **Local telemetry test** | Pending |
| 12 | FastAPI example (placeholder) | Pending |
| 13 | Final integration test | Pending |
