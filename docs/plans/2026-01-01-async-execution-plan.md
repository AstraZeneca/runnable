# Async Execution Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add native async execution support to runnable for agentic workflows with async Python functions.

**Architecture:** Hybrid sync/async pattern with explicit method separation (`execute_graph` vs `execute_graph_async`) and shared helpers to minimize code duplication. Base classes provide `NotImplementedError` defaults for async methods - only interactive executors implement them.

**Key Design Decisions:**
- Async methods in base classes raise `NotImplementedError` by default
- Only `LocalExecutor` implements async methods (no separate `AsyncLocalExecutor`)
- Shared helpers extract common logic called by both sync and async paths
- Composite nodes (`ParallelNode`, `MapNode`) get `execute_as_graph_async()` methods
- `PipelineContext` gets `execute_async()` method (no separate `AsyncPipelineContext`)

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

## Task 2: Add async methods to BasePipelineExecutor

**Files:**
- Modify: `runnable/executor.py`

**Step 1: Add async method stubs that raise NotImplementedError**

Add after the existing abstract methods in `BasePipelineExecutor`:

```python
    # ═══════════════════════════════════════════════════════════════
    # Async Path - Default implementations that raise NotImplementedError
    # Only interactive executors (local) override these
    # ═══════════════════════════════════════════════════════════════

    async def execute_graph_async(
        self, dag: "Graph", map_variable: MapVariableType = None
    ):
        """
        Async graph traversal.

        Only implemented by interactive executors that support async execution.
        Transpilers (Argo, etc.) do not implement this.

        Raises:
            NotImplementedError: If executor does not support async execution.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support async execution. "
            f"Use a local executor for async pipelines."
        )

    async def execute_from_graph_async(
        self, node: "BaseNode", map_variable: MapVariableType = None
    ):
        """
        Async node execution entry point.

        Raises:
            NotImplementedError: If executor does not support async execution.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support async execution."
        )

    async def trigger_node_execution_async(
        self, node: "BaseNode", map_variable: MapVariableType = None
    ):
        """
        Async trigger for node execution.

        Raises:
            NotImplementedError: If executor does not support async execution.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support async execution."
        )

    async def _execute_node_async(
        self,
        node: "BaseNode",
        map_variable: MapVariableType = None,
        mock: bool = False,
    ):
        """
        Async node execution wrapper.

        Raises:
            NotImplementedError: If executor does not support async execution.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support async execution."
        )
```

**Step 2: Run tests to verify no regression**

Run: `uv run pytest tests/runnable/test_executor.py -v`
Expected: All existing tests PASS

**Step 3: Commit**

```bash
git add runnable/executor.py
git commit -m "feat: add async method stubs to BasePipelineExecutor"
```

---

## Task 3: Add async method stub to BaseTaskType

**Files:**
- Modify: `runnable/tasks.py`

**Step 1: Add imports at top of file**

```python
import inspect
from typing import Callable, Optional
```

**Step 2: Add async method stub to BaseTaskType**

Add after `execute_command` method:

```python
    async def execute_command_async(
        self,
        map_variable: MapVariableType = None,
        event_callback: Optional[Callable[[dict], None]] = None,
    ) -> StepAttempt:
        """
        Async command execution.

        Only implemented by task types that support async execution
        (AsyncPythonTaskType). Sync task types (PythonTaskType,
        NotebookTaskType, ShellTaskType) raise NotImplementedError.

        Args:
            map_variable: If the command is part of map node.
            event_callback: Optional callback for streaming events.

        Raises:
            NotImplementedError: If task type does not support async execution.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support async execution. "
            f"Use AsyncPythonTask for async functions."
        )
```

**Step 3: Run tests to verify no regression**

Run: `uv run pytest tests/runnable/test_tasks.py -v`
Expected: All existing tests PASS

**Step 4: Commit**

```bash
git add runnable/tasks.py
git commit -m "feat: add execute_command_async stub to BaseTaskType"
```

---

## Task 4: Add shared helpers to GenericPipelineExecutor

**Files:**
- Modify: `extensions/pipeline_executor/__init__.py`

**Step 1: Extract shared helper for node preparation**

Add new method `_prepare_node_for_execution` before `execute_from_graph`:

```python
    def _prepare_node_for_execution(
        self, node: BaseNode, map_variable: MapVariableType = None
    ) -> Optional[StepLog]:
        """
        Setup before node execution - shared by sync/async paths.

        Returns None if node should be skipped (retry logic).
        """
        if self._should_skip_step_in_retry(node, map_variable):
            logger.info(f"Skipping execution of '{node.internal_name}' due to retry logic")
            console.print(
                f":fast_forward: Skipping node {node.internal_name} - already successful",
                style="bold yellow",
            )
            return None

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

        return step_log

    def _finalize_graph_execution(
        self, node: BaseNode, dag: Graph, map_variable: MapVariableType = None
    ):
        """Finalize after graph traversal - shared by sync/async paths."""
        run_log = self._context.run_log_store.get_branch_log(
            node._get_branch_log_name(map_variable), self._context.run_id
        )

        branch = "graph"
        if node.internal_branch_name:
            branch = node.internal_branch_name

        logger.info(f"Finished execution of {branch} with status {run_log.status}")

        if dag == self._context.dag:
            run_log = cast(RunLog, run_log)
            console.print("Completed Execution, Summary:", style="bold color(208)")
            console.print(run_log.get_summary(), style=defaults.info_style)
```

**Step 2: Refactor execute_from_graph to use shared helper**

Update `execute_from_graph` to use `_prepare_node_for_execution`:

```python
    def execute_from_graph(self, node: BaseNode, map_variable: MapVariableType = None):
        """Sync node execution entry point."""
        step_log = self._prepare_node_for_execution(node, map_variable)
        if step_log is None:
            return  # Skipped

        logger.info(f"Executing node: {node.get_summary()}")

        if node.node_type in ["success", "fail"]:
            self._execute_node(node, map_variable=map_variable)
            return

        if node.is_composite:
            node.execute_as_graph(map_variable=map_variable)
            return

        task_name = node._resolve_map_placeholders(node.internal_name, map_variable)
        console.print(
            f":runner: Executing the node {task_name} ... ", style="bold color(208)"
        )
        self.trigger_node_execution(node=node, map_variable=map_variable)
```

**Step 3: Run tests to verify no regression**

Run: `uv run pytest tests/extensions/pipeline_executor/ -v`
Expected: All existing tests PASS

**Step 4: Commit**

```bash
git add extensions/pipeline_executor/__init__.py
git commit -m "refactor: extract shared helpers in GenericPipelineExecutor"
```

---

## Task 5: Add async methods to LocalExecutor

**Files:**
- Modify: `extensions/pipeline_executor/local.py`

**Step 1: Add async method implementations**

Add after existing sync methods:

```python
    # ═══════════════════════════════════════════════════════════════
    # Async Path - implement async methods
    # ═══════════════════════════════════════════════════════════════

    async def execute_graph_async(self, dag: Graph, map_variable: MapVariableType = None):
        """Async graph traversal."""
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
                raise Exception("Potentially running in an infinite loop")
            previous_node = current_node

            try:
                await self.execute_from_graph_async(working_on, map_variable=map_variable)
                # Sync helper - no await needed
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

        # Sync helper - no await needed
        self._finalize_graph_execution(working_on, dag, map_variable)

    async def execute_from_graph_async(
        self, node: BaseNode, map_variable: MapVariableType = None
    ):
        """Async node execution entry point."""
        # Sync helper - no await needed
        step_log = self._prepare_node_for_execution(node, map_variable)
        if step_log is None:
            return  # Skipped

        logger.info(f"Executing node: {node.get_summary()}")

        if node.node_type in ["success", "fail"]:
            await self._execute_node_async(node, map_variable=map_variable)
            return

        if node.is_composite:
            await node.execute_as_graph_async(map_variable=map_variable)
            return

        task_name = node._resolve_map_placeholders(node.internal_name, map_variable)
        console.print(
            f":runner: Executing the node {task_name} ... ", style="bold color(208)"
        )
        await self.trigger_node_execution_async(node=node, map_variable=map_variable)

    async def trigger_node_execution_async(
        self, node: BaseNode, map_variable: MapVariableType = None
    ):
        """Async trigger for node execution."""
        await self._execute_node_async(node=node, map_variable=map_variable)

    async def _execute_node_async(
        self, node: BaseNode, map_variable: MapVariableType = None, mock: bool = False
    ):
        """Async node execution wrapper."""
        current_attempt_number = self._calculate_attempt_number(node, map_variable)
        os.environ[defaults.ATTEMPT_NUMBER] = str(current_attempt_number)

        logger.info(
            f"Trying to execute node: {node.internal_name}, attempt: {current_attempt_number}"
        )

        self._context_node = node

        # Sync - catalog get
        data_catalogs_get: Optional[List[DataCatalog]] = self._sync_catalog(stage="get")
        logger.debug(f"data_catalogs_get: {data_catalogs_get}")

        # ASYNC - execute the node
        step_log = await node.execute_async(
            map_variable=map_variable,
            attempt_number=current_attempt_number,
            mock=mock,
        )

        # Sync - catalog put and finalization
        allow_file_not_found_exc = step_log.status != defaults.SUCCESS
        data_catalogs_put: Optional[List[DataCatalog]] = self._sync_catalog(
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

        self._context.run_log_store.add_step_log(step_log, self._context.run_id)
```

**Step 2: Add necessary imports**

Add at top of file:
```python
from typing import List, Optional
from runnable.datastore import DataCatalog
```

**Step 3: Run tests to verify no regression**

Run: `uv run pytest tests/extensions/pipeline_executor/test_local_executor.py -v`
Expected: All existing tests PASS

**Step 4: Commit**

```bash
git add extensions/pipeline_executor/local.py
git commit -m "feat: add async execution methods to LocalExecutor"
```

---

## Task 6: Add async methods to BaseNode

**Files:**
- Modify: `runnable/nodes.py`

**Step 1: Add execute_async method to BaseNode**

Add after the `execute` method:

```python
    async def execute_async(
        self,
        map_variable: MapVariableType = None,
        attempt_number: int = 1,
        mock: bool = False,
    ) -> StepLog:
        """
        Async execution - default delegates to sync execute().

        Override in subclasses that support true async execution (TaskNode).
        Terminal nodes (SuccessNode, FailNode) use this default.
        """
        return self.execute(
            map_variable=map_variable,
            attempt_number=attempt_number,
            mock=mock,
        )
```

**Step 2: Run tests to verify no regression**

Run: `uv run pytest tests/runnable/test_nodes.py -v`
Expected: All existing tests PASS

**Step 3: Commit**

```bash
git add runnable/nodes.py
git commit -m "feat: add execute_async method to BaseNode"
```

---

## Task 7: Add execute_as_graph_async to CompositeNode

**Files:**
- Modify: `runnable/nodes.py`

**Step 1: Add execute_as_graph_async to CompositeNode**

Find `CompositeNode` class and add:

```python
    async def execute_as_graph_async(self, map_variable: MapVariableType = None):
        """
        Async execution of sub-graph.

        Default raises NotImplementedError - override in subclasses
        that support async execution (ParallelNode, MapNode, DagNode).
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement execute_as_graph_async() "
            f"for async execution support."
        )
```

**Step 2: Commit**

```bash
git add runnable/nodes.py
git commit -m "feat: add execute_as_graph_async stub to CompositeNode"
```

---

## Task 8: Add async methods to TaskNode

**Files:**
- Modify: `runnable/nodes.py`

**Step 1: Add execute_async to TaskNode**

Find `TaskNode` class and override `execute_async`:

```python
    async def execute_async(
        self,
        map_variable: MapVariableType = None,
        attempt_number: int = 1,
        mock: bool = False,
    ) -> StepLog:
        """Async task execution."""
        step_log = self._context.run_log_store.get_step_log(
            self._get_step_log_name(map_variable), self._context.run_id
        )

        task = self._get_task()

        # Try async first, fall back to sync
        try:
            attempt_log = await task.execute_command_async(map_variable=map_variable)
        except NotImplementedError:
            # Task doesn't support async, fall back to sync
            attempt_log = task.execute_command(map_variable=map_variable)

        step_log.attempts.append(attempt_log)
        step_log.status = attempt_log.status
        return step_log
```

**Step 2: Commit**

```bash
git add runnable/nodes.py
git commit -m "feat: add execute_async to TaskNode with fallback to sync"
```

---

## Task 9: Add async methods to ParallelNode

**Files:**
- Modify: `extensions/nodes/parallel.py`

**Step 1: Add execute_as_graph_async method**

Add after `execute_as_graph`:

```python
    async def execute_as_graph_async(self, map_variable: MapVariableType = None):
        """Async parallel execution."""
        self.fan_out(map_variable=map_variable)  # sync - just creates branch logs

        for _, branch in self.branches.items():
            await self._context.pipeline_executor.execute_graph_async(
                branch, map_variable=map_variable
            )

        self.fan_in(map_variable=map_variable)  # sync - just collates status
```

**Step 2: Run tests to verify no regression**

Run: `uv run pytest tests/extensions/nodes/test_parallel.py -v`
Expected: All existing tests PASS

**Step 3: Commit**

```bash
git add extensions/nodes/parallel.py
git commit -m "feat: add execute_as_graph_async to ParallelNode"
```

---

## Task 10: Add async methods to MapNode

**Files:**
- Modify: `extensions/nodes/map.py`

**Step 1: Add execute_as_graph_async method**

Add after `execute_as_graph`:

```python
    async def execute_as_graph_async(self, map_variable: MapVariableType = None):
        """Async map execution."""
        self.fan_out(map_variable=map_variable)  # sync

        iterate_on = self._context.run_log_store.get_parameters(self._context.run_id)[
            self.iterate_on
        ].get_value()

        for iter_variable in iterate_on:
            effective_map_variable = map_variable or {}
            effective_map_variable[self.iterate_as] = iter_variable

            await self._context.pipeline_executor.execute_graph_async(
                self.branch, map_variable=effective_map_variable
            )

        self.fan_in(map_variable=map_variable)  # sync
```

**Step 2: Commit**

```bash
git add extensions/nodes/map.py
git commit -m "feat: add execute_as_graph_async to MapNode"
```

---

## Task 11: Add execute_async to PipelineContext

**Files:**
- Modify: `runnable/context.py`

**Step 1: Add shared helper for completion handling**

Add to `PipelineContext`:

```python
    def _handle_completion(self):
        """Handle post-execution - shared by sync/async."""
        run_log = self.run_log_store.get_run_log_by_id(
            run_id=self.run_id, full=False
        )

        if run_log.status == defaults.SUCCESS:
            console.print("Pipeline executed successfully!", style=defaults.success_style)
            logfire.info("Pipeline completed", status="success")
        else:
            console.print("Pipeline execution failed.", style=defaults.error_style)
            logfire.error("Pipeline failed", status="failed")
            raise exceptions.ExecutionFailedError(self.run_id)
```

**Step 2: Add execute_async method**

Add after `execute` method:

```python
    async def execute_async(self):
        """Async pipeline execution."""
        assert self.dag is not None

        pipeline_name = getattr(self.dag, "name", "unnamed")

        with logfire.span(
            "pipeline:{pipeline_name}",
            pipeline_name=pipeline_name,
            run_id=self.run_id,
            executor=self.pipeline_executor.__class__.__name__,
        ):
            logfire.info("Async pipeline execution started")

            console.print("Working with context:")
            console.print(run_context)
            console.rule(style="[dark orange]")

            if self.pipeline_executor._should_setup_run_log_at_traversal:
                self.pipeline_executor._set_up_run_log(exists_ok=False)

            try:
                await self.pipeline_executor.execute_graph_async(dag=self.dag)
                self._handle_completion()

            except Exception as e:
                console.print(e, style=defaults.error_style)
                logfire.error("Pipeline failed with exception", error=str(e)[:256])
                raise

            if self.pipeline_executor._should_setup_run_log_at_traversal:
                return run_context.run_log_store.get_run_log_by_id(
                    run_id=run_context.run_id
                )
```

**Step 3: Refactor execute to use shared helper**

Update `execute` method to use `_handle_completion()`.

**Step 4: Commit**

```bash
git add runnable/context.py
git commit -m "feat: add execute_async to PipelineContext"
```

---

## Task 12: Create AsyncPythonTaskType - Test

**Files:**
- Create: `tests/runnable/test_async_tasks.py`

**Step 1: Write the failing test**

```python
import pytest

from runnable.tasks import AsyncPythonTaskType


def test_async_python_task_type_initialization():
    """Test AsyncPythonTaskType can be instantiated with command."""
    task = AsyncPythonTaskType(command="examples.common.functions.hello")
    assert task.task_type == "async-python"
    assert task.command == "examples.common.functions.hello"


def test_async_python_task_type_sync_raises():
    """Test AsyncPythonTaskType.execute_command raises RuntimeError."""
    task = AsyncPythonTaskType(command="test.func")
    with pytest.raises(RuntimeError, match="requires async execution"):
        task.execute_command()
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/runnable/test_async_tasks.py -v`
Expected: FAIL with "cannot import name 'AsyncPythonTaskType'"

**Step 3: Commit test**

```bash
git add tests/runnable/test_async_tasks.py
git commit -m "test: add failing tests for AsyncPythonTaskType"
```

---

## Task 13: Create AsyncPythonTaskType - Implementation

**Files:**
- Modify: `runnable/tasks.py`

**Step 1: Add AsyncPythonTaskType class**

Add after `ShellTaskType`:

```python
class AsyncPythonTaskType(BaseTaskType):
    """
    Task type for executing async Python functions.

    Supports both regular async functions and AsyncGenerator functions
    for streaming use cases. Sync execution raises RuntimeError.
    """

    task_type: str = Field(default="async-python", serialization_alias="command_type")
    command: str

    def execute_command(self, map_variable: MapVariableType = None) -> StepAttempt:
        """Sync execution - not supported for async tasks."""
        raise RuntimeError(
            "AsyncPythonTaskType requires async execution. "
            "Use AsyncPipeline or ensure executor calls execute_command_async()."
        )

    async def execute_command_async(
        self,
        map_variable: MapVariableType = None,
        event_callback: Optional[Callable[[dict], None]] = None,
    ) -> StepAttempt:
        """Execute an async Python function."""
        attempt_log = StepAttempt(
            status=defaults.FAIL,
            start_time=str(datetime.now()),
        )

        def emit_event(event: dict):
            if event_callback:
                event_callback(event)

        with logfire.span(
            "task:{task_name}",
            task_name=self.command,
            task_type=self.task_type,
        ):
            with self.execution_context(map_variable=map_variable) as params:
                with self.expose_secrets():
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
                        filtered_parameters = parameters.filter_arguments_for_func(
                            f, params.copy(), map_variable
                        )
                        logger.info(
                            f"Calling async {func} from {module} with {filtered_parameters}"
                        )

                        with redirect_output(console=task_console) as (buffer, stderr_buffer):
                            result = await f(**filtered_parameters)

                            # Handle AsyncGenerator (streaming)
                            if inspect.isasyncgen(result):
                                accumulated_chunks: List[str] = []
                                async for chunk in result:
                                    accumulated_chunks.append(str(chunk))
                                    emit_event({
                                        "type": "task_chunk",
                                        "name": self.command,
                                        "chunk": str(chunk),
                                    })
                                user_set_parameters = "".join(accumulated_chunks)
                            else:
                                user_set_parameters = result

                        # Process returns (reuse existing logic)
                        if self.returns:
                            if not isinstance(user_set_parameters, tuple):
                                user_set_parameters = (user_set_parameters,)

                            if len(user_set_parameters) != len(self.returns):
                                raise ValueError(
                                    "Returns task signature does not match function returns"
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
                            emit_event({"type": "task_completed", "name": self.command})

                        attempt_log.status = defaults.SUCCESS

                    except Exception as e:
                        msg = f"Call to async function {self.command} did not succeed.\n"
                        attempt_log.message = msg
                        task_console.print_exception(show_locals=False)
                        task_console.log(e, style=defaults.error_style)
                        logfire.error("Task failed", error=str(e)[:256])
                        emit_event({
                            "type": "task_error",
                            "name": self.command,
                            "error": str(e)[:256],
                        })

        attempt_log.end_time = str(datetime.now())
        return attempt_log
```

**Step 2: Run tests**

Run: `uv run pytest tests/runnable/test_async_tasks.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add runnable/tasks.py
git commit -m "feat: add AsyncPythonTaskType for async function execution"
```

---

## Task 14: Register AsyncPythonTaskType plugin

**Files:**
- Modify: `pyproject.toml`

**Step 1: Add entry point**

```toml
[project.entry-points.'tasks']
"async-python" = "runnable.tasks:AsyncPythonTaskType"
```

**Step 2: Sync and verify**

Run: `uv sync --all-extras --dev`

Verify:
```bash
uv run python -c "from stevedore import driver; mgr = driver.DriverManager(namespace='tasks', name='async-python', invoke_on_load=True, invoke_kwds={'command': 'test'}); print(mgr.driver)"
```

**Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "feat: register async-python task plugin"
```

---

## Task 15: Create AsyncPythonTask SDK class

**Files:**
- Modify: `runnable/sdk.py`

**Step 1: Add imports**

```python
import asyncio
```

**Step 2: Add AsyncPythonTask class**

```python
class AsyncPythonTask(BaseTraversal):
    """SDK class for async Python tasks."""

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
    def serialize_returns(cls, returns: List[Union[str, TaskReturns]]) -> List[TaskReturns]:
        task_returns = []
        for ret in returns:
            if isinstance(ret, str):
                task_returns.append(TaskReturns(name=ret, kind="json"))
            else:
                task_returns.append(ret)
        return task_returns

    def create_node(self) -> TaskNode:
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

Add `AsyncPythonTask` to `StepType`.

**Step 4: Commit**

```bash
git add runnable/sdk.py
git commit -m "feat: add AsyncPythonTask SDK class"
```

---

## Task 16: Create AsyncPipeline SDK class

**Files:**
- Modify: `runnable/sdk.py`

**Step 1: Add AsyncPipeline class**

```python
class AsyncPipeline(BaseModel):
    """Pipeline with async execution."""

    steps: List[StepType] = Field(default_factory=list)
    name: str = Field(default="")
    description: str = Field(default="")

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_steps(self) -> Self:
        if not self.steps:
            raise ValueError("AsyncPipeline must have at least one step")

        for i, step in enumerate(self.steps[:-1]):
            if not step.next_node:
                step.next_node = self.steps[i + 1].name

        last_step = self.steps[-1]
        if not last_step.terminate_with_success and not last_step.terminate_with_failure:
            last_step.terminate_with_success = True

        return self

    def return_dag(self) -> graph.Graph:
        dag = graph.Graph(
            start_at=self.steps[0].name,
            name=self.name,
            description=self.description,
        )

        for step in self.steps:
            dag.add_node(step.create_node())

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
        """Execute the pipeline asynchronously."""
        from runnable.context import PipelineContext

        dag = self.return_dag()

        services = defaults.DEFAULT_SERVICES.copy()
        services["pipeline_executor"] = {"type": "local"}

        context = PipelineContext(
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

        context.__dict__["_dag"] = dag

        await context.execute_async()
```

**Step 2: Commit**

```bash
git add runnable/sdk.py
git commit -m "feat: add AsyncPipeline SDK class"
```

---

## Task 17: Export async classes

**Files:**
- Modify: `runnable/__init__.py`

**Step 1: Add exports**

```python
from runnable.sdk import (
    # ... existing ...
    AsyncPythonTask,
    AsyncPipeline,
)
```

**Step 2: Verify**

```bash
uv run python -c "from runnable import AsyncPipeline, AsyncPythonTask; print('OK')"
```

**Step 3: Commit**

```bash
git add runnable/__init__.py
git commit -m "feat: export async classes from runnable package"
```

---

## Task 18: Integration test

**Files:**
- Create: `tests/integration/test_async_pipeline.py`

**Step 1: Write integration test**

```python
import pytest
import asyncio

from runnable import AsyncPipeline, AsyncPythonTask


async def compute(x: int = 10) -> int:
    await asyncio.sleep(0.01)
    return x * 2


async def finalize(result: int) -> str:
    await asyncio.sleep(0.01)
    return f"Final: {result}"


@pytest.mark.asyncio
async def test_simple_async_pipeline_execution():
    pipeline = AsyncPipeline(
        steps=[
            AsyncPythonTask(function=compute, name="compute", returns=["result"]),
            AsyncPythonTask(function=finalize, name="finalize", returns=["final"]),
        ]
    )

    await pipeline.execute()
```

**Step 2: Run test**

Run: `uv run pytest tests/integration/test_async_pipeline.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/integration/test_async_pipeline.py
git commit -m "test: add integration test for async pipeline"
```

---

## Task 19: Final verification

**Step 1: Run all async tests**

```bash
uv run pytest tests/runnable/test_async_tasks.py tests/integration/test_async_pipeline.py -v
```

**Step 2: Run full test suite**

```bash
uv run pytest --ignore=tests/integration -v
```

**Step 3: Final commit**

```bash
git add -A
git commit -m "chore: async execution feature complete"
```

---

## Summary

This plan implements:

1. **Base class async stubs** - `BasePipelineExecutor` and `BaseTaskType` get async methods that raise `NotImplementedError`

2. **Shared helpers** - `GenericPipelineExecutor` extracts common logic into helpers called by both sync/async paths

3. **LocalExecutor async methods** - `execute_graph_async`, `execute_from_graph_async`, `trigger_node_execution_async`, `_execute_node_async`

4. **Node async methods** - `BaseNode.execute_async`, `CompositeNode.execute_as_graph_async`, `TaskNode.execute_async` (with fallback)

5. **Composite node async** - `ParallelNode.execute_as_graph_async`, `MapNode.execute_as_graph_async`

6. **PipelineContext.execute_async** - Async pipeline execution entry point

7. **AsyncPythonTaskType** - Task type for async functions with streaming support

8. **SDK classes** - `AsyncPythonTask`, `AsyncPipeline`

**Key architectural decisions:**
- No separate `AsyncLocalExecutor` - `LocalExecutor` implements both sync and async
- No separate `AsyncPipelineContext` - `PipelineContext` has both `execute()` and `execute_async()`
- Shared helpers minimize code duplication between sync/async paths
- Clear `NotImplementedError` messages when async called on unsupported components
