# Async Execution Design

## Overview

Add native async support to runnable for agentic workflows that need to execute async Python functions and stream events in real-time.

## Problem

The current runnable execution is entirely synchronous. For agentic workflows:
- User functions may need to perform async IO (HTTP calls, database queries)
- Real-time streaming requires async execution to emit events as tasks complete
- FastAPI integration currently uses thread pool executor as a workaround

## Solution

Create a parallel async execution stack following the existing plugin architecture:
- `AsyncPythonTaskType` - task plugin for async functions
- `AsyncLocalExecutor` - pipeline executor with async graph traversal
- `AsyncPipelineContext` - context with async execute()
- `AsyncPythonTask` / `AsyncPipeline` - SDK classes

## Async Boundary

**Async (awaited):**
- `AsyncPipeline.execute()` - entry point
- `AsyncPipelineContext.execute()` - orchestration
- `AsyncLocalExecutor` methods: `execute_graph`, `execute_from_graph`, `trigger_node_execution`, `_execute_node`
- `AsyncPythonTaskType.execute_command()` - calls user's async function

**Sync (called directly from async):**
- `run_log_store` operations (create, get, add step log)
- `catalog` operations (get, put)
- `fan_out()` / `fan_in()` for composite nodes
- `_get_status_and_next_node_name()`
- Parameter handling
- All existing infrastructure

## Components

### 1. AsyncPythonTaskType

Location: `runnable/tasks.py`

```python
class AsyncPythonTaskType(BaseTaskType):
    task_type: str = Field(default="async-python", serialization_alias="command_type")
    command: str

    async def execute_command(
        self,
        map_variable: MapVariableType = None,
    ) -> StepAttempt:
        """Execute an async Python function."""
        attempt_log = StepAttempt(...)

        with logfire.span("task:{task_name}", ...):
            with self.execution_context(map_variable=map_variable) as params:
                self._emit_event({"type": "task_started", ...})

                # Get and call the async function
                module, func = utils.get_module_and_attr_names(self.command)
                imported_module = importlib.import_module(module)
                f = getattr(imported_module, func)

                filtered_parameters = parameters.filter_arguments_for_func(f, params, map_variable)

                # ASYNC: await the user's function
                user_set_parameters = await f(**filtered_parameters)

                # Handle returns, emit completion event
                self._emit_event({"type": "task_completed", ...})

        return attempt_log
```

Entry point: `tasks.async-python`

### 2. AsyncLocalExecutor

Location: `extensions/pipeline_executor/async_local.py`

Inherits from `GenericPipelineExecutor`, overrides methods to be async:

```python
class AsyncLocalExecutor(GenericPipelineExecutor):
    service_name: str = "async-local"

    async def execute_graph(self, dag: Graph, map_variable: MapVariableType = None):
        """Async graph traversal - follows same pattern as sync version."""
        current_node = dag.start_at
        previous_node = None

        while True:
            working_on = dag.get_node_by_name(current_node)

            if previous_node == current_node:
                raise Exception("Potentially running in a infinite loop")
            previous_node = current_node

            await self.execute_from_graph(working_on, map_variable=map_variable)

            # Sync call - fine
            status, next_node_name = self._get_status_and_next_node_name(
                current_node=working_on, dag=dag, map_variable=map_variable
            )

            if working_on.node_type in ["success", "fail"]:
                break

            current_node = next_node_name

    async def execute_from_graph(self, node: BaseNode, map_variable: MapVariableType = None):
        """Async node execution - follows same pattern as sync version."""
        # Skip logic for retry (sync)
        if self._should_skip_step_in_retry(node, map_variable):
            return

        # Step log creation (sync)
        step_log = self._context.run_log_store.create_step_log(...)
        step_log.status = defaults.PROCESSING
        self._context.run_log_store.add_step_log(step_log, self._context.run_id)

        # Terminal nodes
        if node.node_type in ["success", "fail"]:
            await self._execute_node(node, map_variable=map_variable)
            return

        # Composite nodes
        if node.is_composite:
            node.fan_out(map_variable=map_variable)  # sync
            await node.execute_as_graph(map_variable=map_variable)  # async
            node.fan_in(map_variable=map_variable)  # sync
            return

        # Task nodes
        await self.trigger_node_execution(node=node, map_variable=map_variable)

    async def trigger_node_execution(self, node: BaseNode, map_variable: MapVariableType = None):
        await self._execute_node(node=node, map_variable=map_variable)

    async def _execute_node(self, node: BaseNode, map_variable: MapVariableType = None, mock: bool = False):
        """Async node execution wrapper."""
        current_attempt_number = self._calculate_attempt_number(node, map_variable)
        os.environ[defaults.ATTEMPT_NUMBER] = str(current_attempt_number)

        self._context_node = node

        # Catalog get (sync)
        data_catalogs_get = self._sync_catalog(stage="get")

        # ASYNC: Execute the node
        step_log = await node.execute(
            map_variable=map_variable,
            attempt_number=current_attempt_number,
            mock=mock,
        )

        # Catalog put (sync)
        data_catalogs_put = self._sync_catalog(stage="put", ...)
        step_log.add_data_catalogs(data_catalogs_put or [])
        step_log.add_data_catalogs(data_catalogs_get or [])

        self.add_task_log_to_catalog(name=self._context_node.internal_name, map_variable=map_variable)
        self._context_node = None

        # Add step log (sync)
        self._context.run_log_store.add_step_log(step_log, self._context.run_id)
```

Entry point: `pipeline_executor.async-local`

### 3. AsyncPipelineContext

Location: `runnable/context.py`

```python
class AsyncPipelineContext(PipelineContext):
    """Pipeline context with async execution."""

    async def execute(self):
        """Async pipeline execution."""
        assert self.dag is not None

        pipeline_name = getattr(self.dag, "name", "unnamed")

        with logfire.span("pipeline:{pipeline_name}", ...):
            logfire.info("Pipeline execution started")

            if self.pipeline_executor._should_setup_run_log_at_traversal:
                self.pipeline_executor._set_up_run_log(exists_ok=False)

            try:
                await self.pipeline_executor.execute_graph(dag=self.dag)

                run_log = run_context.run_log_store.get_run_log_by_id(
                    run_id=run_context.run_id, full=False
                )

                if run_log.status == defaults.SUCCESS:
                    logfire.info("Pipeline completed", status="success")
                else:
                    logfire.error("Pipeline failed", status="failed")
                    raise exceptions.ExecutionFailedError(run_context.run_id)

            except Exception as e:
                logfire.error("Pipeline failed with exception", error=str(e)[:256])
                raise
```

### 4. SDK Classes

Location: `runnable/sdk.py`

```python
class AsyncPythonTask(BaseTraversal):
    """SDK class for async Python tasks."""

    function: Callable  # Must be async function
    returns: List[Union[str, TaskReturns]] = Field(default_factory=list)

    @field_validator("function")
    @classmethod
    def validate_async(cls, func):
        if not asyncio.iscoroutinefunction(func):
            raise ValueError("function must be an async function")
        return func

    def create_node(self) -> TaskNode:
        # Creates node with command_type="async-python"
        ...


class AsyncPipeline(BaseModel):
    """Pipeline with async execution."""

    steps: List[StepType]

    async def execute(
        self,
        configuration_file: str = "",
        run_id: str = "",
        tag: str = "",
        parameters_file: str = "",
    ):
        """Execute the pipeline asynchronously."""
        dag = self.return_dag()

        # Force async-local executor
        context = AsyncPipelineContext(
            pipeline_executor={"type": "async-local"},
            run_id=run_id,
            tag=tag,
            parameters_file=parameters_file,
            # ... other config
        )

        await context.execute()
```

## Plugin Registration

In `pyproject.toml`:

```toml
[project.entry-points.'tasks']
async-python = "runnable.tasks:AsyncPythonTaskType"

[project.entry-points.'pipeline_executor']
async-local = "extensions.pipeline_executor.async_local:AsyncLocalExecutor"
```

## Streaming Events

Uses existing `_emit_event()` / `set_stream_queue()` mechanism:
- Tasks emit events to queue during execution
- FastAPI/caller polls queue for real-time updates
- No changes needed to telemetry infrastructure

## Usage Example

```python
from runnable import AsyncPipeline, AsyncPythonTask

async def fetch_data(url: str) -> dict:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()

async def process(data: dict) -> str:
    await asyncio.sleep(1)  # async processing
    return f"Processed {len(data)} items"

pipeline = AsyncPipeline(
    steps=[
        AsyncPythonTask(function=fetch_data, name="fetch", returns=["data"]),
        AsyncPythonTask(function=process, name="process", returns=["result"]),
    ]
)

# In async context
await pipeline.execute(parameters_file="params.yaml")
```

## Out of Scope

- Async notebook/shell tasks (inherently sync)
- Async container/argo executors (don't make sense)
- Changes to run log store or catalog (stay sync)
- Async parameter handling

## Testing Strategy

1. Unit tests for `AsyncPythonTaskType.execute_command()`
2. Unit tests for `AsyncLocalExecutor` graph traversal
3. Integration tests with simple async pipelines
4. Integration tests with composite nodes (parallel, map)
5. Test error handling and retry logic
