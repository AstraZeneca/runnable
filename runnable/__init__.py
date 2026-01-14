# ruff: noqa

import os

from rich.console import Console

console = Console(record=True)
console.print(":runner: Lets go!!")

task_console = Console(record=True)


def _configure_telemetry():
    """
    Auto-configure logfire/OpenTelemetry from environment variables.

    This runs at import time to ensure telemetry is configured before
    any task execution, especially for containerized execution where
    the entrypoint is `runnable execute_single_node ...`.

    Environment variables:
        RUNNABLE_TELEMETRY_CONSOLE: Set to "true" for console output
        OTEL_EXPORTER_OTLP_ENDPOINT: OTLP collector endpoint (e.g., http://localhost:4317)
        LOGFIRE_TOKEN: Logfire cloud token (enables send_to_logfire)
    """
    import logfire_api as logfire

    console_enabled = os.environ.get("RUNNABLE_TELEMETRY_CONSOLE", "").lower() == "true"
    otlp_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "")
    logfire_token = os.environ.get("LOGFIRE_TOKEN", "")

    # Skip if no telemetry config is set
    if not (console_enabled or otlp_endpoint or logfire_token):
        return

    try:
        import logfire

        config_kwargs = {
            "send_to_logfire": bool(logfire_token),
        }

        if console_enabled:
            config_kwargs["console"] = logfire.ConsoleOptions(
                colors="auto",
                span_style="indented",
                verbose=True,
            )

        if otlp_endpoint:
            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                    OTLPSpanExporter,
                )
                from opentelemetry.sdk.trace.export import BatchSpanProcessor

                config_kwargs["additional_span_processors"] = [
                    BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint))
                ]
            except ImportError:
                pass  # OTLP exporter not installed

        logfire.configure(**config_kwargs)  # ty: ignore

    except ImportError:
        pass  # logfire not installed, telemetry is no-op


_configure_telemetry()


from runnable.sdk import (  # noqa;
    AsyncPipeline,
    AsyncPythonTask,
    Catalog,
    Conditional,
    Fail,
    Loop,
    Map,
    NotebookJob,
    NotebookTask,
    Parallel,
    Pipeline,
    PythonJob,
    PythonTask,
    ShellJob,
    ShellTask,
    Stub,
    Success,
    json,
    metric,
    pickled,
)
from runnable.telemetry import (  # noqa;
    OTEL_AVAILABLE,
    get_stream_queue,
    set_stream_queue,
    truncate_value,
)

# Conditionally export StreamingSpanProcessor
if OTEL_AVAILABLE:
    from runnable.telemetry import StreamingSpanProcessor  # noqa;
else:
    StreamingSpanProcessor = None  # type: ignore

# Needed to disable ploomber telemetry
os.environ["PLOOMBER_STATS_ENABLED"] = "false"
