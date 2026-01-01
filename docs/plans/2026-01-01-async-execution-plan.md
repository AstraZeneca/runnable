# Async Execution Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add native async execution support to runnable for agentic workflows with async Python functions.

**Architecture:** Create parallel async execution stack (AsyncPythonTaskType, AsyncLocalExecutor, AsyncPipelineContext, AsyncPipeline) following existing plugin architecture. Async boundary at task execution and graph traversal; run log store and catalog remain sync. Streaming via AsyncGenerator pattern - events flow OUT to client, data flows INTERNALLY between tasks.

**Tech Stack:** Python asyncio, inspect (isasyncgen), pydantic, stevedore plugins, pytest-asyncio

---

## Task 1: Add pytest-asyncio dependency

**Files:**
- Modify: `pyproject.toml`

**Step 1: Add pytest-asyncio to dev dependencies**

In `pyproject.toml`, find the `[tool.uv]` section with dev-dependencies and add pytest-asyncio:

```toml
[tool.uv]
dev-dependencies = [
    # ... existing deps ...
    "pytest-asyncio>=0.23.0",
]
```

**Step 2: Sync dependencies**

Run: `uv sync --all-extras --dev`
Expected: Successfully installed pytest-asyncio

**Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: add pytest-asyncio for async tests"
```

---

## Task 2: Create AsyncPythonTaskType - Test

**Files:**
- Create: `tests/runnable/test_async_tasks.py`

**Step 1: Write the failing test for AsyncPythonTaskType initialization**

```python
import pytest

from runnable.tasks import AsyncPythonTaskType


def test_async_python_task_type_initialization():
    """Test AsyncPythonTaskType can be instantiated with command."""
    task = AsyncPythonTaskType(command="examples.common.functions.hello")
    assert task.task_type == "async-python"
    assert task.command == "examples.common.functions.hello"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/runnable/test_async_tasks.py::test_async_python_task_type_initialization -v`
Expected: FAIL with "cannot import name 'AsyncPythonTaskType'"

**Step 3: Commit test**

```bash
git add tests/runnable/test_async_tasks.py
git commit -m "test: add failing test for AsyncPythonTaskType"
```

---

## Task 3: Create AsyncPythonTaskType - Implementation

**Files:**
- Modify: `runnable/tasks.py`

**Step 1: Add inspect import at top of file**

```python
import inspect
```

**Step 2: Implement AsyncPythonTaskType class**

Add after the `ShellTaskType` class (around line 966):

```python
class AsyncPythonTaskType(BaseTaskType):
    """
    Task type for executing async Python functions.

    Similar to PythonTaskType but the command must point to an async function.
    The execute_command method is async and awaits the user's function.

    Supports AsyncGenerator functions for streaming - chunks are yielded as events
    while the accumulated result is stored for the next task.
    """

    task_type: str = Field(default="async-python", serialization_alias="command_type")
    command: str

    async def execute_command(
        self,
        map_variable: MapVariableType = None,
        event_callback: Optional[Callable[[dict], None]] = None,
    ) -> StepAttempt:
        """
        Execute an async Python function.

        Args:
            map_variable: If the node is of a map state, the iterable value.
            event_callback: Optional callback for streaming events (task_chunk, etc.)

        If the function returns an AsyncGenerator, chunks are:
        1. Yielded via event_callback as task_chunk events (for client streaming)
        2. Accumulated internally and stored as the return value (for next task)
        """
        attempt_log = StepAttempt(
            status=defaults.FAIL,
            start_time=str(datetime.now()),
            retry_indicator=self._context.retry_indicator,
        )

        def emit_event(event: dict):
            """Emit event via callback if provided."""
            if event_callback:
                event_callback(event)

        with logfire.span(
            "task:{task_name}",
            task_name=self.command,
            task_type=self.task_type,
        ):
            with (
                self.execution_context(map_variable=map_variable) as params,
                self.expose_secrets() as _,
            ):
                logfire.info(
                    "Task started",
                    inputs=self._safe_serialize_params(params),
                )
                emit_event({
                    "type": "task_started",
                    "name": self.command,
                    "inputs": self._safe_serialize_params(params),
                })

                module, func = utils.get_module_and_attr_names(self.command)
                sys.path.insert(0, os.getcwd())
                imported_module = importlib.import_module(module)
                f = getattr(imported_module, func)

                try:
                    try:
                        filtered_parameters = parameters.filter_arguments_for_func(
                            f, params.copy(), map_variable
                        )
                        logger.info(
                            f"Calling async {func} from {module} with {filtered_parameters}"
                        )
                        with redirect_output(console=task_console) as (
                            buffer,
                            stderr_buffer,
                        ):
                            # Call the async function
                            result = await f(**filtered_parameters)

                            # Handle AsyncGenerator (streaming) vs regular return
                            if inspect.isasyncgen(result):
                                # Stream chunks and accumulate
                                accumulated_chunks: List[str] = []
                                async for chunk in result:
                                    accumulated_chunks.append(str(chunk))
                                    emit_event({
                                        "type": "task_chunk",
                                        "name": self.command,
                                        "chunk": str(chunk),
                                    })
                                # Join accumulated chunks as the final value
                                user_set_parameters = "".join(accumulated_chunks)
                            else:
                                user_set_parameters = result

                    except Exception as e:
                        raise exceptions.CommandCallError(
                            f"Async function call: {self.command} did not succeed.\n"
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
                        if not isinstance(user_set_parameters, tuple):
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
                            outputs=self._safe_serialize_params(output_parameters),
                            status="success",
                        )
                        emit_event({
                            "type": "task_completed",
                            "name": self.command,
                            "outputs": self._safe_serialize_params(output_parameters),
                        })
                    else:
                        logfire.info("Task completed", status="success")
                        emit_event({
                            "type": "task_completed",
                            "name": self.command,
                        })

                    attempt_log.status = defaults.SUCCESS
                except Exception as _e:
                    msg = f"Call to the async function {self.command} did not succeed.\n"
                    attempt_log.message = msg
                    task_console.print_exception(show_locals=False)
                    task_console.log(_e, style=defaults.error_style)
                    logfire.error("Task failed", error=str(_e)[:256])
                    emit_event({
                        "type": "task_error",
                        "name": self.command,
                        "error": str(_e)[:256]
                    })

        attempt_log.end_time = str(datetime.now())
        return attempt_log
```

**Step 2: Run test to verify it passes**

Run: `uv run pytest tests/runnable/test_async_tasks.py::test_async_python_task_type_initialization -v`
Expected: PASS

**Step 3: Commit**

```bash
git add runnable/tasks.py
git commit -m "feat: add AsyncPythonTaskType for async function execution"
```

---

## Task 4: Register AsyncPythonTaskType plugin

**Files:**
- Modify: `pyproject.toml`

**Step 1: Add entry point for async-python task**

Find `[project.entry-points.'tasks']` section and add:

```toml
[project.entry-points.'tasks']
"python" = "runnable.tasks:PythonTaskType"
"shell" = "runnable.tasks:ShellTaskType"
"notebook" = "runnable.tasks:NotebookTaskType"
"async-python" = "runnable.tasks:AsyncPythonTaskType"
```

**Step 2: Reinstall to register plugin**

Run: `uv sync --all-extras --dev`
Expected: Package reinstalled with new entry point

**Step 3: Verify plugin registration**

Run: `uv run python -c "from stevedore import driver; mgr = driver.DriverManager(namespace='tasks', name='async-python', invoke_on_load=True, invoke_kwds={'command': 'test'}); print(mgr.driver)"`
Expected: `<runnable.tasks.AsyncPythonTaskType object ...>`

**Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "feat: register async-python task plugin"
```

---

## Task 5: Test AsyncPythonTaskType.execute_command

**Files:**
- Modify: `tests/runnable/test_async_tasks.py`

**Step 1: Add async execution test**

```python
import asyncio
from typing import AsyncGenerator
from unittest.mock import Mock, patch
from datetime import datetime

import pytest

from runnable import defaults
from runnable.datastore import JsonParameter, StepAttempt
from runnable.tasks import AsyncPythonTaskType


# Test async function to be called
async def sample_async_function(x: int = 10) -> int:
    await asyncio.sleep(0.01)  # Simulate async work
    return x * 2


# Test async generator function for streaming
async def sample_streaming_function(prompt: str = "test") -> AsyncGenerator[str, None]:
    """Simulate LLM-style streaming."""
    chunks = ["Hello", " ", "world", "!"]
    for chunk in chunks:
        await asyncio.sleep(0.01)
        yield chunk


@pytest.fixture
def mock_context(mocker):
    """Mock runnable.context at module level"""
    mock_ctx = Mock()
    mock_ctx.run_id = "test_run"
    mock_ctx.secrets = Mock()
    mock_ctx.run_log_store = Mock()
    mock_ctx.catalog = Mock()

    parameters_dict = {
        "x": JsonParameter(kind="json", value=5),
    }
    mock_ctx.run_log_store.get_parameters.return_value = parameters_dict.copy()
    mock_ctx.retry_indicator = ""

    mocker.patch("runnable.context.run_context", mock_ctx)
    return mock_ctx


@pytest.mark.asyncio
async def test_async_python_task_execute_command(mock_context, mocker):
    """Test AsyncPythonTaskType can execute an async function."""
    # Mock the module import to return our test function
    mocker.patch(
        "runnable.tasks.importlib.import_module",
        return_value=Mock(sample_async_function=sample_async_function)
    )
    mocker.patch("runnable.tasks.utils.get_module_and_attr_names", return_value=("test_module", "sample_async_function"))

    task = AsyncPythonTaskType(
        command="test_module.sample_async_function",
        returns=[{"name": "result", "kind": "json"}]
    )

    attempt_log = await task.execute_command()

    assert attempt_log.status == defaults.SUCCESS
    assert "result" in attempt_log.output_parameters
    assert attempt_log.output_parameters["result"].get_value() == 10  # 5 * 2


@pytest.mark.asyncio
async def test_async_python_task_execute_streaming(mock_context, mocker):
    """Test AsyncPythonTaskType handles AsyncGenerator (streaming) functions."""
    # Mock the module import to return our streaming function
    mocker.patch(
        "runnable.tasks.importlib.import_module",
        return_value=Mock(sample_streaming_function=sample_streaming_function)
    )
    mocker.patch("runnable.tasks.utils.get_module_and_attr_names", return_value=("test_module", "sample_streaming_function"))

    # Collect emitted events
    emitted_events = []
    def event_callback(event):
        emitted_events.append(event)

    task = AsyncPythonTaskType(
        command="test_module.sample_streaming_function",
        returns=[{"name": "text", "kind": "json"}]
    )

    attempt_log = await task.execute_command(event_callback=event_callback)

    # Should succeed
    assert attempt_log.status == defaults.SUCCESS

    # Should have accumulated the chunks
    assert "text" in attempt_log.output_parameters
    assert attempt_log.output_parameters["text"].get_value() == "Hello world!"

    # Should have emitted task_chunk events
    chunk_events = [e for e in emitted_events if e.get("type") == "task_chunk"]
    assert len(chunk_events) == 4
    assert chunk_events[0]["chunk"] == "Hello"
    assert chunk_events[1]["chunk"] == " "
    assert chunk_events[2]["chunk"] == "world"
    assert chunk_events[3]["chunk"] == "!"

    # Should have task_started and task_completed events
    assert any(e.get("type") == "task_started" for e in emitted_events)
    assert any(e.get("type") == "task_completed" for e in emitted_events)
```

**Step 2: Run tests**

Run: `uv run pytest tests/runnable/test_async_tasks.py -v`
Expected: Both tests PASS

**Step 3: Commit**

```bash
git add tests/runnable/test_async_tasks.py
git commit -m "test: add async execution and streaming tests for AsyncPythonTaskType"
```

---

## Task 6: Create AsyncLocalExecutor - Test

**Files:**
- Create: `tests/extensions/pipeline_executor/test_async_local_executor.py`

**Step 1: Write failing test for AsyncLocalExecutor**

```python
import pytest

from extensions.pipeline_executor.async_local import AsyncLocalExecutor


def test_async_local_executor_initialization():
    """Test AsyncLocalExecutor can be instantiated."""
    executor = AsyncLocalExecutor()
    assert executor.service_name == "async-local"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/extensions/pipeline_executor/test_async_local_executor.py::test_async_local_executor_initialization -v`
Expected: FAIL with "No module named 'extensions.pipeline_executor.async_local'"

**Step 3: Commit test**

```bash
git add tests/extensions/pipeline_executor/test_async_local_executor.py
git commit -m "test: add failing test for AsyncLocalExecutor"
```

---

## Task 7: Create AsyncLocalExecutor - Basic Implementation

**Files:**
- Create: `extensions/pipeline_executor/async_local.py`

**Step 1: Create the file with basic structure**

```python
import logging
import os
from typing import Optional

from pydantic import Field, PrivateAttr

from extensions.pipeline_executor import GenericPipelineExecutor
from runnable import console, defaults, exceptions
from runnable.datastore import DataCatalog
from runnable.defaults import MapVariableType
from runnable.graph import Graph
from runnable.nodes import BaseNode

logger = logging.getLogger(defaults.LOGGER_NAME)


class AsyncLocalExecutor(GenericPipelineExecutor):
    """
    Async local executor for running pipelines with async Python tasks.

    This executor provides async graph traversal and task execution,
    enabling native async/await patterns in pipeline code.

    Example config:

    ```yaml
    pipeline-executor:
      type: async-local
    ```
    """

    service_name: str = "async-local"

    _is_local: bool = PrivateAttr(default=True)

    async def execute_graph(self, dag: Graph, map_variable: MapVariableType = None):
        """
        Async graph traversal - follows same pattern as sync version.

        Args:
            dag: The directed acyclic graph to traverse and execute.
            map_variable: If the node is of a map state, the iterable value.
        """
        current_node = dag.start_at
        previous_node = None
        logger.info(f"Running async execution with {current_node}")

        branch_task_name: str = ""
        if dag.internal_branch_name:
            branch_task_name = BaseNode._resolve_map_placeholders(
                dag.internal_branch_name or "Graph",
                map_variable,
            )
            console.print(
                f":runner: Executing the branch {branch_task_name} ... ",
                style="bold color(208)",
            )

        while True:
            working_on = dag.get_node_by_name(current_node)
            task_name = working_on._resolve_map_placeholders(
                working_on.internal_name, map_variable
            )

            if previous_node == current_node:
                raise Exception("Potentially running in a infinite loop")

            previous_node = current_node

            try:
                await self.execute_from_graph(working_on, map_variable=map_variable)
                status, next_node_name = self._get_status_and_next_node_name(
                    current_node=working_on, dag=dag, map_variable=map_variable
                )

                if status == defaults.SUCCESS:
                    console.print(f":white_check_mark: Node {task_name} succeeded")
                else:
                    console.print(f":x: Node {task_name} failed")
            except Exception as e:
                console.print(":x: Error during execution", style="bold red")
                console.print(e, style=defaults.error_style)
                logger.exception(e)
                raise

            console.rule(style="[dark orange]")

            if working_on.node_type in ["success", "fail"]:
                break

            current_node = next_node_name

        run_log = self._context.run_log_store.get_branch_log(
            working_on._get_branch_log_name(map_variable), self._context.run_id
        )

        branch = "graph"
        if working_on.internal_branch_name:
            branch = working_on.internal_branch_name

        logger.info(f"Finished async execution of {branch} with status {run_log.status}")

    async def execute_from_graph(self, node: BaseNode, map_variable: MapVariableType = None):
        """
        Async node execution entry point from graph traversal.

        Follows same pattern as sync version but awaits async operations.
        """
        if self._should_skip_step_in_retry(node, map_variable):
            logger.info(f"Skipping execution of '{node.internal_name}' due to retry logic")
            console.print(
                f":fast_forward: Skipping node {node.internal_name} - already successful",
                style="bold yellow",
            )
            return

        # Handle step log creation for retry vs normal runs
        if self._context.is_retry:
            try:
                step_log = self._context.run_log_store.get_step_log(
                    node._get_step_log_name(map_variable), self._context.run_id
                )
                logger.info(f"Reusing existing step log for retry: {node.internal_name}")
            except exceptions.StepLogNotFoundError:
                step_log = self._context.run_log_store.create_step_log(
                    node.name, node._get_step_log_name(map_variable)
                )
                logger.info(f"Creating new step log for retry: {node.internal_name}")
        else:
            step_log = self._context.run_log_store.create_step_log(
                node.name, node._get_step_log_name(map_variable)
            )

        step_log.step_type = node.node_type
        step_log.status = defaults.PROCESSING
        self._context.run_log_store.add_step_log(step_log, self._context.run_id)

        logger.info(f"Executing node: {node.get_summary()}")

        # Terminal nodes
        if node.node_type in ["success", "fail"]:
            await self._execute_node(node, map_variable=map_variable)
            return

        # Composite nodes (parallel, map, dag)
        if node.is_composite:
            node.fan_out(map_variable=map_variable)  # sync
            await node.execute_as_graph(map_variable=map_variable)  # async
            node.fan_in(map_variable=map_variable)  # sync
            return

        # Task nodes
        task_name = node._resolve_map_placeholders(node.internal_name, map_variable)
        console.print(
            f":runner: Executing the node {task_name} ... ", style="bold color(208)"
        )
        await self.trigger_node_execution(node=node, map_variable=map_variable)

    async def trigger_node_execution(
        self, node: BaseNode, map_variable: MapVariableType = None
    ):
        """Trigger async node execution."""
        await self._execute_node(node=node, map_variable=map_variable)

    async def _execute_node(
        self,
        node: BaseNode,
        map_variable: MapVariableType = None,
        mock: bool = False,
    ):
        """
        Async node execution wrapper.

        Handles catalog sync (sync), task execution (async), and step log updates (sync).
        """
        current_attempt_number = self._calculate_attempt_number(node, map_variable)
        os.environ[defaults.ATTEMPT_NUMBER] = str(current_attempt_number)

        logger.info(
            f"Trying to execute node: {node.internal_name}, attempt: {current_attempt_number}"
        )

        self._context_node = node

        # Catalog get (sync)
        data_catalogs_get: Optional[list[DataCatalog]] = self._sync_catalog(stage="get")
        logger.debug(f"data_catalogs_get: {data_catalogs_get}")

        # ASYNC: Execute the node
        step_log = await node.execute(
            map_variable=map_variable,
            attempt_number=current_attempt_number,
            mock=mock,
        )

        allow_file_not_found_exc = True
        if step_log.status == defaults.SUCCESS:
            allow_file_not_found_exc = False

        # Catalog put (sync)
        data_catalogs_put: Optional[list[DataCatalog]] = self._sync_catalog(
            stage="put", allow_file_no_found_exc=allow_file_not_found_exc
        )
        logger.debug(f"data_catalogs_put: {data_catalogs_put}")
        step_log.add_data_catalogs(data_catalogs_put or [])
        step_log.add_data_catalogs(data_catalogs_get or [])

        console.print(f"Summary of the step: {step_log.internal_name}")
        console.print(step_log.get_summary(), style=defaults.info_style)

        self.add_task_log_to_catalog(
            name=self._context_node.internal_name, map_variable=map_variable
        )

        self._context_node = None

        # Add step log (sync)
        self._context.run_log_store.add_step_log(step_log, self._context.run_id)

    def execute_node(self, node: BaseNode, map_variable: MapVariableType = None):
        """
        Sync entry point - raises error as this executor requires async.

        Use await _execute_node() instead.
        """
        raise RuntimeError(
            "AsyncLocalExecutor requires async execution. "
            "Use 'await executor._execute_node()' or AsyncPipeline.execute()"
        )
```

**Step 2: Run test to verify it passes**

Run: `uv run pytest tests/extensions/pipeline_executor/test_async_local_executor.py::test_async_local_executor_initialization -v`
Expected: PASS

**Step 3: Commit**

```bash
git add extensions/pipeline_executor/async_local.py
git commit -m "feat: add AsyncLocalExecutor for async graph traversal"
```

---

## Task 8: Register AsyncLocalExecutor plugin

**Files:**
- Modify: `pyproject.toml`

**Step 1: Add entry point for async-local executor**

Find `[project.entry-points.'pipeline_executor']` section and add:

```toml
[project.entry-points.'pipeline_executor']
"local" = "extensions.pipeline_executor.local:LocalExecutor"
"local-container" = "extensions.pipeline_executor.local_container:LocalContainerExecutor"
"emulator" = "extensions.pipeline_executor.emulate:Emulator"
"argo" = "extensions.pipeline_executor.argo:ArgoExecutor"
"mocked" = "extensions.pipeline_executor.mocked:MockedExecutor"
"async-local" = "extensions.pipeline_executor.async_local:AsyncLocalExecutor"
```

**Step 2: Reinstall to register plugin**

Run: `uv sync --all-extras --dev`
Expected: Package reinstalled with new entry point

**Step 3: Verify plugin registration**

Run: `uv run python -c "from stevedore import driver; mgr = driver.DriverManager(namespace='pipeline_executor', name='async-local', invoke_on_load=True); print(mgr.driver)"`
Expected: `<extensions.pipeline_executor.async_local.AsyncLocalExecutor object ...>`

**Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "feat: register async-local executor plugin"
```

---

## Task 9: Create AsyncPipelineContext - Test

**Files:**
- Modify: `tests/runnable/test_context.py` or create `tests/runnable/test_async_context.py`

**Step 1: Write failing test**

```python
import pytest

from runnable.context import AsyncPipelineContext


def test_async_pipeline_context_exists():
    """Test AsyncPipelineContext class exists."""
    assert AsyncPipelineContext is not None
    assert hasattr(AsyncPipelineContext, 'execute')
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/runnable/test_async_context.py::test_async_pipeline_context_exists -v`
Expected: FAIL with "cannot import name 'AsyncPipelineContext'"

**Step 3: Commit test**

```bash
git add tests/runnable/test_async_context.py
git commit -m "test: add failing test for AsyncPipelineContext"
```

---

## Task 10: Create AsyncPipelineContext - Implementation

**Files:**
- Modify: `runnable/context.py`

**Step 1: Add AsyncPipelineContext class**

Add after `PipelineContext` class (around line 470):

```python
class AsyncPipelineContext(PipelineContext):
    """
    Pipeline context with async execution support.

    Uses AsyncLocalExecutor for graph traversal and supports
    AsyncPythonTask nodes with async functions.
    """

    async def execute(self):
        """Execute the pipeline asynchronously."""
        assert self.dag is not None

        pipeline_name = getattr(self.dag, "name", "unnamed")

        with logfire.span(
            "pipeline:{pipeline_name}",
            pipeline_name=pipeline_name,
            run_id=self.run_id,
            executor=self.pipeline_executor.__class__.__name__,
        ):
            logfire.info("Async pipeline execution started")

            console.print("Working with async context:")
            console.print(run_context)
            console.rule(style="[dark orange]")

            if self.pipeline_executor._should_setup_run_log_at_traversal:
                self.pipeline_executor._set_up_run_log(exists_ok=False)

            try:
                await self.pipeline_executor.execute_graph(dag=self.dag)

                run_log = run_context.run_log_store.get_run_log_by_id(
                    run_id=run_context.run_id, full=False
                )

                if run_log.status == defaults.SUCCESS:
                    console.print(
                        "Pipeline executed successfully!", style=defaults.success_style
                    )
                    logfire.info("Pipeline completed", status="success")
                else:
                    console.print(
                        "Pipeline execution failed.", style=defaults.error_style
                    )
                    logfire.error("Pipeline failed", status="failed")
                    raise exceptions.ExecutionFailedError(run_context.run_id)

            except Exception as e:
                console.print(e, style=defaults.error_style)
                logfire.error("Pipeline failed with exception", error=str(e)[:256])
                raise

            if self.pipeline_executor._should_setup_run_log_at_traversal:
                return run_context.run_log_store.get_run_log_by_id(
                    run_id=run_context.run_id
                )
```

**Step 2: Run test to verify it passes**

Run: `uv run pytest tests/runnable/test_async_context.py::test_async_pipeline_context_exists -v`
Expected: PASS

**Step 3: Commit**

```bash
git add runnable/context.py
git commit -m "feat: add AsyncPipelineContext with async execute()"
```

---

## Task 11: Create AsyncPythonTask SDK class - Test

**Files:**
- Create: `tests/runnable/test_async_sdk.py`

**Step 1: Write failing test**

```python
import pytest
import asyncio

from runnable.sdk import AsyncPythonTask


async def sample_func(x: int) -> int:
    return x * 2


def test_async_python_task_initialization():
    """Test AsyncPythonTask can be created with async function."""
    task = AsyncPythonTask(function=sample_func, name="test_task")
    assert task.name == "test_task"
    assert task.function == sample_func


def test_async_python_task_rejects_sync_function():
    """Test AsyncPythonTask rejects non-async functions."""
    def sync_func(x: int) -> int:
        return x * 2

    with pytest.raises(ValueError, match="must be an async function"):
        AsyncPythonTask(function=sync_func, name="test_task")
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/runnable/test_async_sdk.py -v`
Expected: FAIL with "cannot import name 'AsyncPythonTask'"

**Step 3: Commit test**

```bash
git add tests/runnable/test_async_sdk.py
git commit -m "test: add failing tests for AsyncPythonTask SDK class"
```

---

## Task 12: Create AsyncPythonTask SDK class - Implementation

**Files:**
- Modify: `runnable/sdk.py`

**Step 1: Add imports at top of file**

```python
import asyncio
```

**Step 2: Add AsyncPythonTask class**

Add after the `ShellTask` class:

```python
class AsyncPythonTask(BaseTraversal):
    """
    An async Python task that executes an async function.

    Use this for tasks that need to perform async I/O operations
    like HTTP requests, database queries, etc.

    Example:
        >>> async def fetch_data(url: str) -> dict:
        ...     async with aiohttp.ClientSession() as session:
        ...         async with session.get(url) as response:
        ...             return await response.json()
        ...
        >>> task = AsyncPythonTask(
        ...     function=fetch_data,
        ...     name="fetch",
        ...     returns=["data"]
        ... )
    """

    function: Callable = Field(exclude=True)
    returns: List[Union[str, TaskReturns]] = Field(default_factory=list)
    catalog: Optional[Catalog] = Field(default=None)
    overrides: Dict[str, Any] = Field(default_factory=dict)
    secrets: List[str] = Field(default_factory=list)

    _task_type: str = PrivateAttr(default="async-python")

    model_config = ConfigDict(extra="forbid")

    @field_validator("function")
    @classmethod
    def validate_async_function(cls, func: Callable) -> Callable:
        """Validate that the function is an async function."""
        if not asyncio.iscoroutinefunction(func):
            raise ValueError(
                f"function must be an async function (defined with 'async def'), "
                f"got {type(func).__name__}"
            )
        return func

    @field_validator("returns", mode="before")
    @classmethod
    def serialize_returns(
        cls, returns: List[Union[str, TaskReturns]]
    ) -> List[TaskReturns]:
        """Convert string returns to TaskReturns objects."""
        task_returns = []
        for ret in returns:
            if isinstance(ret, str):
                task_returns.append(TaskReturns(name=ret, kind="json"))
            else:
                task_returns.append(ret)
        return task_returns

    def create_node(self) -> TaskNode:
        """Create a TaskNode for this async task."""
        if self.function:
            module = self.function.__module__
            name = self.function.__qualname__

            command = f"{module}.{name}"
        else:
            raise ValueError("AsyncPythonTask requires a function")

        node_dict = {
            "command": command,
            "command_type": self._task_type,
            "returns": [r.model_dump() for r in self.returns],
            "catalog": self.catalog.model_dump() if self.catalog else None,
            "overrides": self.overrides,
            "secrets": self.secrets,
        }

        return TaskNode.parse_from_config(self.model_dump() | node_dict)
```

**Step 3: Update StepType union**

Find `StepType` definition and add `AsyncPythonTask`:

```python
StepType = Union[
    "Stub",
    "PythonTask",
    "NotebookTask",
    "ShellTask",
    "Parallel",
    "Map",
    "Conditional",
    "AsyncPythonTask",
]
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/runnable/test_async_sdk.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add runnable/sdk.py
git commit -m "feat: add AsyncPythonTask SDK class"
```

---

## Task 13: Create AsyncPipeline SDK class - Test

**Files:**
- Modify: `tests/runnable/test_async_sdk.py`

**Step 1: Add test for AsyncPipeline**

```python
from typing import AsyncGenerator
from runnable.sdk import AsyncPipeline, AsyncPythonTask, PipelineEvent


async def step1(x: int = 5) -> int:
    return x * 2


async def step2(x: int) -> str:
    return f"Result: {x}"


async def streaming_step(prompt: str = "test") -> AsyncGenerator[str, None]:
    """Streaming function that yields chunks."""
    for chunk in ["Hello", " ", "world", "!"]:
        yield chunk


def test_async_pipeline_initialization():
    """Test AsyncPipeline can be created with async tasks."""
    pipeline = AsyncPipeline(
        steps=[
            AsyncPythonTask(function=step1, name="step1", returns=["x"]),
            AsyncPythonTask(function=step2, name="step2", returns=["result"]),
        ]
    )
    assert len(pipeline.steps) == 2


def test_async_pipeline_has_async_execute():
    """Test AsyncPipeline.execute is a coroutine function."""
    pipeline = AsyncPipeline(
        steps=[AsyncPythonTask(function=step1, name="step1")]
    )
    assert asyncio.iscoroutinefunction(pipeline.execute)


def test_async_pipeline_has_execute_stream():
    """Test AsyncPipeline.execute_stream returns AsyncGenerator."""
    pipeline = AsyncPipeline(
        steps=[AsyncPythonTask(function=step1, name="step1")]
    )
    assert hasattr(pipeline, 'execute_stream')
    # execute_stream is an async generator function
    assert asyncio.iscoroutinefunction(pipeline.execute_stream)


def test_async_pipeline_accepts_streaming_tasks():
    """Test AsyncPipeline accepts tasks with AsyncGenerator functions."""
    pipeline = AsyncPipeline(
        steps=[
            AsyncPythonTask(function=streaming_step, name="stream", returns=["text"]),
            AsyncPythonTask(function=step2, name="process", returns=["result"]),
        ]
    )
    assert len(pipeline.steps) == 2
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/runnable/test_async_sdk.py::test_async_pipeline_initialization -v`
Expected: FAIL with "cannot import name 'AsyncPipeline'"

**Step 3: Commit test**

```bash
git add tests/runnable/test_async_sdk.py
git commit -m "test: add failing tests for AsyncPipeline SDK class"
```

---

## Task 14: Create AsyncPipeline SDK class - Implementation

**Files:**
- Modify: `runnable/sdk.py`

**Step 1: Add typing imports at top of file**

```python
from typing import AsyncGenerator
```

**Step 2: Add TaskChunkEvent model after other event models**

```python
class TaskChunkEvent(BaseEvent):
    """Event emitted when a streaming task yields a chunk."""

    type: Literal["task_chunk"] = "task_chunk"
    task_name: str
    chunk: str
```

**Step 3: Update PipelineEvent union to include TaskChunkEvent**

```python
PipelineEvent = Annotated[
    Union[
        PipelineStartedEvent,
        PipelineCompletedEvent,
        PipelineErrorEvent,
        TaskStartedEvent,
        TaskCompletedEvent,
        TaskErrorEvent,
        TaskChunkEvent,  # Add this
    ],
    Field(discriminator="type"),
]
```

**Step 4: Add AsyncPipeline class**

Add after the `Pipeline` class:

```python
class AsyncPipeline(BaseModel):
    """
    A pipeline with async execution support.

    Use this when your pipeline contains AsyncPythonTask steps
    that need to execute async functions.

    Supports two execution modes:
    - execute(): Simple async execution, blocks until complete
    - execute_stream(): Returns AsyncGenerator of events for SSE streaming

    Example (simple):
        >>> pipeline = AsyncPipeline(steps=[...])
        >>> await pipeline.execute()

    Example (streaming):
        >>> async for event in pipeline.execute_stream():
        ...     print(event)  # PipelineEvent objects
    """

    steps: List[StepType] = Field(default_factory=list)
    name: str = Field(default="")
    description: str = Field(default="")

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_steps(self) -> Self:
        """Validate and link steps."""
        if not self.steps:
            raise ValueError("AsyncPipeline must have at least one step")

        # Auto-link steps if not already linked
        for i, step in enumerate(self.steps[:-1]):
            if not step.next_node:
                step.next_node = self.steps[i + 1].name

        # Mark last step as terminal if not already
        last_step = self.steps[-1]
        if not last_step.terminate_with_success and not last_step.terminate_with_failure:
            last_step.terminate_with_success = True

        return self

    def return_dag(self) -> graph.Graph:
        """Build and return the DAG for this pipeline."""
        dag = graph.Graph(
            start_at=self.steps[0].name,
            name=self.name,
            description=self.description,
        )

        for step in self.steps:
            dag.add_node(step.create_node())

        # Add terminal nodes
        dag.add_terminal_nodes()
        dag.check_graph()

        return dag

    async def execute(
        self,
        configuration_file: str = "",
        run_id: str = "",
        tag: str = "",
        parameters_file: str = "",
    ):
        """
        Execute the pipeline asynchronously (simple mode, no streaming).

        Args:
            configuration_file: Path to configuration YAML (optional)
            run_id: Custom run ID (auto-generated if not provided)
            tag: Tag for this execution
            parameters_file: Path to parameters YAML
        """
        from runnable.context import AsyncPipelineContext

        dag = self.return_dag()

        # Build services config - force async-local executor
        services = defaults.DEFAULT_SERVICES.copy()
        services["pipeline_executor"] = {"type": "async-local"}

        context = AsyncPipelineContext(
            pipeline_executor=services["pipeline_executor"],
            catalog=services["catalog"],
            secrets=services["secrets"],
            pickler=services["pickler"],
            run_log_store=services["run_log_store"],
            run_id=run_id,
            tag=tag,
            parameters_file=parameters_file,
            configuration_file=configuration_file,
            pipeline_definition_file=f"{self.__class__.__module__}.{self.__class__.__qualname__}",
        )

        # Store dag on context for access
        context.__dict__["_dag"] = dag

        await context.execute()

    async def execute_stream(
        self,
        configuration_file: str = "",
        run_id: str = "",
        tag: str = "",
        parameters_file: str = "",
    ) -> AsyncGenerator[PipelineEvent, None]:
        """
        Execute the pipeline and yield events as they occur.

        This method returns an AsyncGenerator that yields PipelineEvent objects,
        suitable for Server-Sent Events (SSE) streaming.

        Args:
            configuration_file: Path to configuration YAML (optional)
            run_id: Custom run ID (auto-generated if not provided)
            tag: Tag for this execution
            parameters_file: Path to parameters YAML

        Yields:
            PipelineEvent: Events as the pipeline executes (task_started,
                task_chunk, task_completed, pipeline_completed, etc.)

        Example:
            @app.post("/run")
            async def run_pipeline(request: Request):
                async def event_stream():
                    async for event in pipeline.execute_stream():
                        yield f"data: {event.model_dump_json()}\\n\\n"
                return StreamingResponse(event_stream(), media_type="text/event-stream")
        """
        import asyncio
        from datetime import datetime
        from runnable.context import AsyncPipelineContext

        dag = self.return_dag()

        # Generate run_id if not provided
        if not run_id:
            run_id = f"stream-{datetime.now().strftime('%Y%m%d%H%M%S')}"

        # Build services config - force async-local executor
        services = defaults.DEFAULT_SERVICES.copy()
        services["pipeline_executor"] = {"type": "async-local"}

        # Event queue for streaming
        event_queue: asyncio.Queue[PipelineEvent] = asyncio.Queue()

        def event_callback(event_dict: dict):
            """Convert dict events to typed PipelineEvent and queue them."""
            event_type = event_dict.get("type")
            event_dict["run_id"] = run_id

            if event_type == "task_started":
                event_queue.put_nowait(TaskStartedEvent(**event_dict))
            elif event_type == "task_chunk":
                event_queue.put_nowait(TaskChunkEvent(**event_dict))
            elif event_type == "task_completed":
                event_queue.put_nowait(TaskCompletedEvent(**event_dict))
            elif event_type == "task_error":
                event_queue.put_nowait(TaskErrorEvent(**event_dict))

        context = AsyncPipelineContext(
            pipeline_executor=services["pipeline_executor"],
            catalog=services["catalog"],
            secrets=services["secrets"],
            pickler=services["pickler"],
            run_log_store=services["run_log_store"],
            run_id=run_id,
            tag=tag,
            parameters_file=parameters_file,
            configuration_file=configuration_file,
            pipeline_definition_file=f"{self.__class__.__module__}.{self.__class__.__qualname__}",
        )

        # Store dag and event callback on context
        context.__dict__["_dag"] = dag
        context.__dict__["_event_callback"] = event_callback

        # Emit pipeline started
        yield PipelineStartedEvent(
            run_id=run_id,
            tag=tag,
            total_tasks=len([s for s in self.steps if hasattr(s, 'function')]),
        )

        # Run pipeline in background task, yielding events as they arrive
        async def run_pipeline():
            try:
                await context.execute()
                event_queue.put_nowait(PipelineCompletedEvent(
                    run_id=run_id,
                    status="success",
                ))
            except Exception as e:
                event_queue.put_nowait(PipelineErrorEvent(
                    run_id=run_id,
                    error=str(e)[:256],
                    error_type=type(e).__name__,
                ))

        # Start pipeline execution
        pipeline_task = asyncio.create_task(run_pipeline())

        # Yield events until pipeline completes
        while True:
            try:
                # Wait for event with timeout to check if pipeline is done
                event = await asyncio.wait_for(event_queue.get(), timeout=0.1)
                yield event

                # Stop if pipeline completed or errored
                if isinstance(event, (PipelineCompletedEvent, PipelineErrorEvent)):
                    break
            except asyncio.TimeoutError:
                # Check if pipeline task is done
                if pipeline_task.done():
                    # Drain any remaining events
                    while not event_queue.empty():
                        yield event_queue.get_nowait()
                    break

        # Ensure pipeline task is awaited
        await pipeline_task
```

**Step 2: Override dag property for AsyncPipelineContext**

In `runnable/context.py`, update `AsyncPipelineContext` to handle the pre-built dag:

```python
class AsyncPipelineContext(PipelineContext):
    # ... existing code ...

    @computed_field  # type: ignore
    @cached_property
    def dag(self) -> Graph | None:
        """Get the dag - use pre-built dag if available."""
        if hasattr(self, '_dag') and self._dag is not None:
            return self._dag
        return super().dag
```

**Step 3: Run tests to verify they pass**

Run: `uv run pytest tests/runnable/test_async_sdk.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add runnable/sdk.py runnable/context.py
git commit -m "feat: add AsyncPipeline SDK class with async execute()"
```

---

## Task 15: Export async classes from runnable package

**Files:**
- Modify: `runnable/__init__.py`

**Step 1: Add exports**

Add to the imports from `runnable.sdk`:

```python
from runnable.sdk import (
    # ... existing imports ...
    AsyncPythonTask,
    AsyncPipeline,
)
```

Add to `runnable.context` imports (if not already exported):

```python
from runnable.context import (
    # ... existing imports ...
    AsyncPipelineContext,
)
```

**Step 2: Verify imports work**

Run: `uv run python -c "from runnable import AsyncPipeline, AsyncPythonTask; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add runnable/__init__.py
git commit -m "feat: export async classes from runnable package"
```

---

## Task 16: Integration test - Simple async pipeline

**Files:**
- Create: `tests/integration/test_async_pipeline.py`

**Step 1: Write integration test**

```python
import pytest
import asyncio

from runnable import AsyncPipeline, AsyncPythonTask


async def compute(x: int = 10) -> int:
    """Simple async compute."""
    await asyncio.sleep(0.01)
    return x * 2


async def finalize(result: int) -> str:
    """Finalize result."""
    await asyncio.sleep(0.01)
    return f"Final: {result}"


@pytest.mark.asyncio
async def test_simple_async_pipeline_execution():
    """Test a simple async pipeline executes successfully."""
    pipeline = AsyncPipeline(
        steps=[
            AsyncPythonTask(function=compute, name="compute", returns=["result"]),
            AsyncPythonTask(function=finalize, name="finalize", returns=["final"]),
        ]
    )

    # Should complete without error
    await pipeline.execute()
```

**Step 2: Run integration test**

Run: `uv run pytest tests/integration/test_async_pipeline.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/integration/test_async_pipeline.py
git commit -m "test: add integration test for async pipeline"
```

---

## Task 17: Add example async pipelines

**Files:**
- Create: `examples/async-execution/simple_async.py`
- Create: `examples/async-execution/streaming_llm.py`

**Step 1: Create simple example file**

```python
"""
Simple async pipeline example.

Run with:
    uv run python examples/async-execution/simple_async.py
"""

import asyncio

from runnable import AsyncPipeline, AsyncPythonTask


async def fetch_data(url: str = "https://api.example.com") -> dict:
    """Simulate async data fetching."""
    print(f"Fetching data from {url}...")
    await asyncio.sleep(1)  # Simulate network delay
    data = {"items": [1, 2, 3, 4, 5], "source": url}
    print(f"Fetched {len(data['items'])} items")
    return data


async def process_data(data: dict) -> int:
    """Process the fetched data."""
    print(f"Processing {len(data['items'])} items...")
    await asyncio.sleep(0.5)  # Simulate processing
    total = sum(data["items"])
    print(f"Processed total: {total}")
    return total


async def save_result(total: int) -> str:
    """Save the result."""
    print(f"Saving result: {total}")
    await asyncio.sleep(0.3)  # Simulate save
    result = f"Saved total: {total}"
    print(result)
    return result


async def main():
    pipeline = AsyncPipeline(
        steps=[
            AsyncPythonTask(
                function=fetch_data,
                name="fetch",
                returns=["data"],
            ),
            AsyncPythonTask(
                function=process_data,
                name="process",
                returns=["total"],
            ),
            AsyncPythonTask(
                function=save_result,
                name="save",
                returns=["result"],
            ),
        ]
    )

    print("Starting async pipeline...")
    await pipeline.execute()
    print("Pipeline completed!")


if __name__ == "__main__":
    asyncio.run(main())
```

**Step 2: Create streaming LLM example file**

```python
"""
Streaming LLM pipeline example with FastAPI SSE.

This demonstrates the AsyncGenerator streaming pattern:
- Events flow OUT to client (task_chunk, task_completed, etc.)
- Data flows INTERNALLY between tasks via parameter store

Run with:
    uv run uvicorn examples.async-execution.streaming_llm:app --reload
"""

import asyncio
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from runnable import AsyncPipeline, AsyncPythonTask


app = FastAPI()


# Simulate LLM streaming response
async def generate_text(prompt: str) -> AsyncGenerator[str, None]:
    """
    Simulate LLM streaming generation.

    Each chunk is:
    1. Yielded as task_chunk event -> streamed to client
    2. Accumulated internally -> stored for next task
    """
    words = ["The", " quick", " brown", " fox", " jumps", " over", " the", " lazy", " dog."]
    for word in words:
        await asyncio.sleep(0.1)  # Simulate token generation delay
        yield word


async def summarize(text: str) -> str:
    """
    Summarize the accumulated text.

    Receives the full accumulated text from generate_text: "The quick brown fox..."
    """
    await asyncio.sleep(0.2)
    return f"Summary ({len(text)} chars): {text[:50]}..."


class GenerateRequest(BaseModel):
    prompt: str = "Tell me a story"


@app.post("/generate")
async def generate(request: GenerateRequest):
    """Stream LLM-style generation with SSE."""

    pipeline = AsyncPipeline(
        steps=[
            AsyncPythonTask(
                function=generate_text,
                name="generate",
                returns=["text"],
            ),
            AsyncPythonTask(
                function=summarize,
                name="summarize",
                returns=["summary"],
            ),
        ]
    )

    async def event_stream():
        async for event in pipeline.execute_stream(
            parameters={"prompt": request.prompt}
        ):
            # Format as SSE
            yield f"data: {event.model_dump_json()}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
    )


@app.get("/")
async def root():
    return {"message": "Streaming LLM Pipeline - POST to /generate"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**Step 3: Test the simple example runs**

Run: `uv run python examples/async-execution/simple_async.py`
Expected: Pipeline executes with async task output

**Step 4: Test the streaming example (optional, requires fastapi)**

Run: `uv run uvicorn examples.async-execution.streaming_llm:app --port 8001`

Then in another terminal:
```bash
curl -X POST http://localhost:8001/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "test"}'
```
Expected: SSE stream with task_chunk events for each word, then task_completed

**Step 5: Commit**

```bash
git add examples/async-execution/
git commit -m "docs: add async pipeline examples (simple and streaming)"
```

---

## Task 18: Final verification and cleanup

**Step 1: Run all async-related tests**

Run: `uv run pytest tests/runnable/test_async_tasks.py tests/runnable/test_async_sdk.py tests/extensions/pipeline_executor/test_async_local_executor.py tests/integration/test_async_pipeline.py -v`
Expected: All tests PASS

**Step 2: Run full test suite to check no regressions**

Run: `uv run pytest --ignore=tests/integration -v`
Expected: All existing tests still PASS

**Step 3: Final commit**

```bash
git add -A
git commit -m "chore: async execution feature complete"
```

---

## Summary

This plan creates:

1. **AsyncPythonTaskType** (`runnable/tasks.py`) - Task plugin for async functions
   - Supports regular async functions (`async def func() -> T`)
   - Supports AsyncGenerator functions (`async def func() -> AsyncGenerator[str, None]`)
   - Accumulates streamed chunks internally, stores result for next task
   - Emits `task_chunk` events via callback for client streaming

2. **AsyncLocalExecutor** (`extensions/pipeline_executor/async_local.py`) - Async graph traversal
   - Follows same pattern as sync LocalExecutor
   - Async methods: `execute_graph`, `execute_from_graph`, `trigger_node_execution`, `_execute_node`
   - Sync operations: run_log_store, catalog, fan_out/fan_in

3. **AsyncPipelineContext** (`runnable/context.py`) - Async execution context
   - Async `execute()` method
   - Passes event callback to executor for streaming

4. **AsyncPythonTask** (`runnable/sdk.py`) - SDK class for async tasks
   - Validates function is async (coroutine or async generator)
   - Creates TaskNode with `command_type="async-python"`

5. **AsyncPipeline** (`runnable/sdk.py`) - SDK class for async pipelines
   - `execute()` - Simple async execution, blocks until complete
   - `execute_stream()` - Returns `AsyncGenerator[PipelineEvent, None]` for SSE streaming

6. **TaskChunkEvent** (`runnable/sdk.py`) - Event model for streaming chunks
   - Type: `"task_chunk"`
   - Fields: `run_id`, `timestamp`, `task_name`, `chunk`

Plugin registrations in `pyproject.toml`:

- `tasks.async-python`
- `pipeline_executor.async-local`

Tests in:

- `tests/runnable/test_async_tasks.py` - AsyncPythonTaskType + streaming tests
- `tests/runnable/test_async_sdk.py` - AsyncPythonTask + AsyncPipeline tests
- `tests/runnable/test_async_context.py` - AsyncPipelineContext tests
- `tests/extensions/pipeline_executor/test_async_local_executor.py` - Executor tests
- `tests/integration/test_async_pipeline.py` - End-to-end tests

Examples in:

- `examples/async-execution/simple_async.py` - Basic async pipeline
- `examples/async-execution/streaming_llm.py` - FastAPI SSE streaming with LLM-style generation

## Streaming Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        AsyncLocalExecutor                        │
│                                                                 │
│  Task 1 (generate_text)          Task 2 (summarize)             │
│  ┌─────────────────────┐         ┌─────────────────────┐        │
│  │ yield "The"         │         │                     │        │
│  │ yield " quick"      │         │ text = "The quick   │        │
│  │ yield " brown"      │         │   brown fox..."     │        │
│  │ ...                 │         │ return "Summary..." │        │
│  └─────────────────────┘         └─────────────────────┘        │
│           │                               ▲                      │
│           │ accumulate internally         │                      │
│           └──────────────────────────────►│                      │
│                    "The quick..."    (via params/run_log_store)  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
           │ yield PipelineEvent
           ▼
    ┌─────────────────┐
    │   FastAPI SSE   │
    │  execute_stream │
    └─────────────────┘
           │
           ▼
       Client/UI
```

**Key insight:** Events flow OUT to client, data flows INTERNALLY between tasks.
