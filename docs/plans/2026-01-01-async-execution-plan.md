# Async Execution Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add native async execution support to runnable for agentic workflows with async Python functions.

**Architecture:** Create parallel async execution stack (AsyncPythonTaskType, AsyncLocalExecutor, AsyncPipelineContext, AsyncPipeline) following existing plugin architecture. Async boundary at task execution and graph traversal; run log store and catalog remain sync.

**Tech Stack:** Python asyncio, pydantic, stevedore plugins, pytest-asyncio

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

**Step 1: Implement AsyncPythonTaskType class**

Add after the `ShellTaskType` class (around line 966):

```python
class AsyncPythonTaskType(BaseTaskType):
    """
    Task type for executing async Python functions.

    Similar to PythonTaskType but the command must point to an async function.
    The execute_command method is async and awaits the user's function.
    """

    task_type: str = Field(default="async-python", serialization_alias="command_type")
    command: str

    async def execute_command(
        self,
        map_variable: MapVariableType = None,
    ) -> StepAttempt:
        """Execute an async Python function."""
        attempt_log = StepAttempt(
            status=defaults.FAIL,
            start_time=str(datetime.now()),
            retry_indicator=self._context.retry_indicator,
        )

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
                self._emit_event({
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
                            # ASYNC: await the user's function
                            user_set_parameters = await f(**filtered_parameters)
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
                        self._emit_event({
                            "type": "task_completed",
                            "name": self.command,
                            "outputs": self._safe_serialize_params(output_parameters),
                        })
                    else:
                        logfire.info("Task completed", status="success")
                        self._emit_event({
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
                    self._emit_event({
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
```

**Step 2: Run test**

Run: `uv run pytest tests/runnable/test_async_tasks.py::test_async_python_task_execute_command -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/runnable/test_async_tasks.py
git commit -m "test: add async execution test for AsyncPythonTaskType"
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
from runnable.sdk import AsyncPipeline, AsyncPythonTask


async def step1(x: int = 5) -> int:
    return x * 2


async def step2(x: int) -> str:
    return f"Result: {x}"


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

**Step 1: Add AsyncPipeline class**

Add after the `Pipeline` class:

```python
class AsyncPipeline(BaseModel):
    """
    A pipeline with async execution support.

    Use this when your pipeline contains AsyncPythonTask steps
    that need to execute async functions.

    Example:
        >>> async def fetch(url: str) -> dict:
        ...     # async HTTP request
        ...     return data
        ...
        >>> pipeline = AsyncPipeline(
        ...     steps=[AsyncPythonTask(function=fetch, name="fetch", returns=["data"])]
        ... )
        >>> await pipeline.execute()
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
        Execute the pipeline asynchronously.

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

## Task 17: Add example async pipeline

**Files:**
- Create: `examples/async-execution/simple_async.py`

**Step 1: Create example file**

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

**Step 2: Test the example runs**

Run: `uv run python examples/async-execution/simple_async.py`
Expected: Pipeline executes with async task output

**Step 3: Commit**

```bash
git add examples/async-execution/simple_async.py
git commit -m "docs: add simple async pipeline example"
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
2. **AsyncLocalExecutor** (`extensions/pipeline_executor/async_local.py`) - Async graph traversal
3. **AsyncPipelineContext** (`runnable/context.py`) - Async execution context
4. **AsyncPythonTask** (`runnable/sdk.py`) - SDK class for async tasks
5. **AsyncPipeline** (`runnable/sdk.py`) - SDK class for async pipelines

Plugin registrations in `pyproject.toml`:
- `tasks.async-python`
- `pipeline_executor.async-local`

Tests in:
- `tests/runnable/test_async_tasks.py`
- `tests/runnable/test_async_sdk.py`
- `tests/runnable/test_async_context.py`
- `tests/extensions/pipeline_executor/test_async_local_executor.py`
- `tests/integration/test_async_pipeline.py`
