# Async Execution Design

## Overview

Add native async support to runnable for agentic workflows that need to execute async Python functions and stream events in real-time.

## Problem

The current runnable execution is entirely synchronous. For agentic workflows:
- User functions may need to perform async IO (HTTP calls, database queries)
- Real-time streaming requires async execution to emit events as tasks complete
- FastAPI integration currently uses thread pool executor as a workaround

## Solution

**Hybrid sync/async architecture** with explicit method separation and shared helpers to minimize code duplication.

### Design Principles

1. **Explicit sync/async methods** - Clear naming convention (`execute_graph` vs `execute_graph_async`)
2. **Shared helpers** - Extract common logic into sync helper methods called by both paths
3. **No breaking changes** - Existing sync code continues to work unchanged
4. **Async boundary at execution** - Run log store, catalog, and parameter handling remain sync
5. **Base class defaults** - Async methods in base classes raise `NotImplementedError` for non-local executors

### Architecture Pattern

```
┌─────────────────────────────────────────────────────────────────┐
│                    Shared Helpers (sync)                        │
│  _prepare_node_for_execution(), _finalize_node_execution()      │
│  _get_next_node(), fan_out(), fan_in(), _sync_catalog()         │
└─────────────────────────────────────────────────────────────────┘
                ▲                              ▲
                │                              │
┌───────────────┴───────────────┐  ┌──────────┴────────────────┐
│     Sync Path                 │  │     Async Path            │
│  execute_graph()              │  │  execute_graph_async()    │
│  execute_from_graph()         │  │  execute_from_graph_async()│
│  execute_as_graph()           │  │  execute_as_graph_async() │
│  _execute_node()              │  │  _execute_node_async()    │
│  execute_command()            │  │  execute_command_async()  │
└───────────────────────────────┘  └───────────────────────────┘
```

## Components

### 1. BasePipelineExecutor (Updated)

Location: `runnable/executor.py`

Add async method stubs that raise `NotImplementedError` by default. Only interactive executors (local, async-local) implement these.

```python
class BasePipelineExecutor(BaseExecutor):
    """
    Base executor with both sync and async method signatures.

    Async methods raise NotImplementedError by default - only implemented
    by interactive executors that support async execution.
    """

    service_type: str = "pipeline_executor"
    overrides: dict[str, Any] = {}

    _context_node: Optional[BaseNode] = PrivateAttr(default=None)

    # ═══════════════════════════════════════════════════════════════
    # Sync Path - Abstract methods (existing)
    # ═══════════════════════════════════════════════════════════════

    @abstractmethod
    def execute_graph(self, dag: Graph, map_variable: MapVariableType = None):
        """Sync graph traversal - implemented by all executors."""
        ...

    @abstractmethod
    def execute_from_graph(self, node: BaseNode, map_variable: MapVariableType = None):
        """Sync node execution entry point."""
        ...

    @abstractmethod
    def trigger_node_execution(self, node: BaseNode, map_variable: MapVariableType = None):
        """Sync trigger for node execution."""
        ...

    @abstractmethod
    def _execute_node(
        self, node: BaseNode, map_variable: MapVariableType = None, mock: bool = False
    ):
        """Sync node execution wrapper."""
        ...

    # ... other existing abstract methods ...

    # ═══════════════════════════════════════════════════════════════
    # Async Path - Default implementations that raise NotImplementedError
    # Only interactive executors (local, async-local) override these
    # ═══════════════════════════════════════════════════════════════

    async def execute_graph_async(self, dag: Graph, map_variable: MapVariableType = None):
        """
        Async graph traversal.

        Only implemented by interactive executors that support async execution.
        Transpilers (Argo, etc.) do not implement this.

        Raises:
            NotImplementedError: If executor does not support async execution.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support async execution. "
            f"Use a local executor or async-local executor for async pipelines."
        )

    async def execute_from_graph_async(
        self, node: BaseNode, map_variable: MapVariableType = None
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
        self, node: BaseNode, map_variable: MapVariableType = None
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
        self, node: BaseNode, map_variable: MapVariableType = None, mock: bool = False
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

### 2. BaseTaskType (Updated)

Location: `runnable/tasks.py`

Add async method stub that raises `NotImplementedError` by default. Only async task types implement this.

```python
class BaseTaskType(BaseModel):
    """
    Base task type with both sync and async execution methods.

    Async method raises NotImplementedError by default - only implemented
    by task types that support async execution (AsyncPythonTaskType).
    """

    task_type: str = Field(serialization_alias="command_type")
    secrets: List[str] = Field(default_factory=list)
    returns: List[TaskReturns] = Field(default_factory=list, alias="returns")

    model_config = ConfigDict(extra="forbid")

    # ═══════════════════════════════════════════════════════════════
    # Shared Helpers (sync) - used by both sync and async paths
    # ═══════════════════════════════════════════════════════════════

    @property
    def _context(self):
        current_context = context.get_run_context()
        if current_context is None:
            raise RuntimeError("No run context available")
        return current_context

    def _safe_serialize_params(self, params: Dict[str, Parameter]) -> Dict[str, Any]:
        """Safely serialize parameters for logging/events."""
        result = {}
        for k, v in params.items():
            try:
                result[k] = truncate_value(v.get_value())
            except Exception:
                result[k] = "<unserializable>"
        return result

    @contextlib.contextmanager
    def expose_secrets(self):
        """Context manager to expose secrets - shared by sync/async."""
        self.set_secrets_as_env_variables()
        try:
            yield
        finally:
            self.delete_secrets_from_env_variables()

    @contextlib.contextmanager
    def execution_context(self, map_variable: MapVariableType = None):
        """
        Context manager for task execution setup/teardown.

        Handles parameter loading and storing - shared by sync/async.
        """
        params = self._context.run_log_store.get_parameters(self._context.run_id)
        yield params
        # Store updated parameters after execution
        diff_parameters = self._diff_parameters(
            self._context.run_log_store.get_parameters(self._context.run_id),
            params
        )
        if diff_parameters:
            self._context.run_log_store.set_parameters(
                self._context.run_id, diff_parameters
            )

    def _process_returns(
        self,
        user_set_parameters: Any,
        attempt_log: StepAttempt,
        params: Dict[str, Parameter],
        map_variable: MapVariableType = None,
    ):
        """
        Process task returns and update attempt log - shared by sync/async.
        """
        if not self.returns:
            return

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

    # ═══════════════════════════════════════════════════════════════
    # Sync Path
    # ═══════════════════════════════════════════════════════════════

    def execute_command(
        self,
        map_variable: MapVariableType = None,
    ) -> StepAttempt:
        """
        Sync command execution - implemented by sync task types.

        Raises:
            NotImplementedError: Base class, must be overridden.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement execute_command()"
        )

    # ═══════════════════════════════════════════════════════════════
    # Async Path
    # ═══════════════════════════════════════════════════════════════

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

### 3. GenericPipelineExecutor (Updated)

Location: `extensions/pipeline_executor/__init__.py`

Implement shared helpers and both sync/async paths:

```python
class GenericPipelineExecutor(BasePipelineExecutor):
    """
    Base executor implementation with both sync and async execution paths.

    Shared helpers contain the actual logic, called by both sync and async methods.
    """

    service_name: str = ""
    service_type: str = "pipeline_executor"

    # ═══════════════════════════════════════════════════════════════
    # Shared Helpers (sync) - contain the actual logic
    # ═══════════════════════════════════════════════════════════════

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
            except exceptions.StepLogNotFoundError:
                step_log = self._context.run_log_store.create_step_log(
                    node.name, node._get_step_log_name(map_variable)
                )
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

    # Existing shared helpers: _get_status_and_next_node_name, _sync_catalog,
    # _calculate_attempt_number, _should_skip_step_in_retry, etc.

    # ═══════════════════════════════════════════════════════════════
    # Sync Path - existing methods (LocalExecutor, Argo transpiler)
    # ═══════════════════════════════════════════════════════════════

    def execute_graph(self, dag: Graph, map_variable: MapVariableType = None):
        """Sync graph traversal - existing implementation."""
        current_node = dag.start_at
        previous_node = None

        while True:
            working_on = dag.get_node_by_name(current_node)

            if previous_node == current_node:
                raise Exception("Potentially running in an infinite loop")
            previous_node = current_node

            self.execute_from_graph(working_on, map_variable=map_variable)
            status, next_node_name = self._get_status_and_next_node_name(
                current_node=working_on, dag=dag, map_variable=map_variable
            )

            if working_on.node_type in ["success", "fail"]:
                break
            current_node = next_node_name

        self._finalize_graph_execution(working_on, dag, map_variable)

    def execute_from_graph(self, node: BaseNode, map_variable: MapVariableType = None):
        """Sync node execution entry point - existing implementation."""
        step_log = self._prepare_node_for_execution(node, map_variable)
        if step_log is None:
            return  # Skipped

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

    # _execute_node - existing implementation
    # trigger_node_execution - existing implementation

    # ═══════════════════════════════════════════════════════════════
    # Async Path - new methods, override base class NotImplementedError
    # Only LocalExecutor and AsyncLocalExecutor implement these
    # ═══════════════════════════════════════════════════════════════

    async def execute_graph_async(self, dag: Graph, map_variable: MapVariableType = None):
        """
        Async graph traversal.

        Default implementation raises NotImplementedError.
        Override in LocalExecutor/AsyncLocalExecutor.
        """
        # Call parent which raises NotImplementedError
        await super().execute_graph_async(dag, map_variable)

    async def execute_from_graph_async(
        self, node: BaseNode, map_variable: MapVariableType = None
    ):
        """
        Async node execution entry point.

        Default implementation raises NotImplementedError.
        Override in LocalExecutor/AsyncLocalExecutor.
        """
        await super().execute_from_graph_async(node, map_variable)

    # ... other async methods delegate to parent NotImplementedError ...
```

### 4. LocalExecutor (Updated)

Location: `extensions/pipeline_executor/local.py`

Implement async methods for the local executor:

```python
class LocalExecutor(GenericPipelineExecutor):
    """
    Local executor with both sync and async execution support.
    """

    service_name: str = "local"

    # Sync methods - existing implementation (unchanged)

    # ═══════════════════════════════════════════════════════════════
    # Async Path - implement async methods
    # ═══════════════════════════════════════════════════════════════

    async def execute_graph_async(self, dag: Graph, map_variable: MapVariableType = None):
        """Async graph traversal."""
        current_node = dag.start_at
        previous_node = None

        while True:
            working_on = dag.get_node_by_name(current_node)

            if previous_node == current_node:
                raise Exception("Potentially running in an infinite loop")
            previous_node = current_node

            await self.execute_from_graph_async(working_on, map_variable=map_variable)
            # Sync helper - no await needed
            status, next_node_name = self._get_status_and_next_node_name(
                current_node=working_on, dag=dag, map_variable=map_variable
            )

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

        self._context_node = node

        # Sync - catalog get
        data_catalogs_get = self._sync_catalog(stage="get")

        # ASYNC - execute the node
        step_log = await node.execute_async(
            map_variable=map_variable,
            attempt_number=current_attempt_number,
            mock=mock,
        )

        # Sync - catalog put and finalization
        allow_file_not_found_exc = step_log.status != defaults.SUCCESS
        data_catalogs_put = self._sync_catalog(
            stage="put", allow_file_no_found_exc=allow_file_not_found_exc
        )
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

### 5. ArgoExecutor (No async support)

Location: `extensions/pipeline_executor/argo.py`

Transpilers do NOT implement async methods - they inherit the `NotImplementedError` from base class:

```python
class ArgoExecutor(GenericPipelineExecutor):
    """
    Argo executor - transpiles DAG to Argo workflow spec.

    Does NOT support async execution - async methods raise NotImplementedError.
    This is intentional: transpilers don't execute, they generate specs.
    """

    service_name: str = "argo"

    # Only sync methods implemented
    # execute_graph - transpiles to Argo spec
    # execute_from_graph - not used by transpilers

    # Async methods: inherited NotImplementedError from BasePipelineExecutor
    # execute_graph_async - raises NotImplementedError
    # execute_from_graph_async - raises NotImplementedError
    # etc.
```

### 6. BaseNode (Updated)

Location: `runnable/nodes.py`

Add async execute method with default fallback to sync:

```python
class BaseNode(ABC, BaseModel):
    """Base node with sync and async execution methods."""

    # ═══════════════════════════════════════════════════════════════
    # Sync Path
    # ═══════════════════════════════════════════════════════════════

    @abstractmethod
    def execute(
        self,
        map_variable: MapVariableType = None,
        attempt_number: int = 1,
        mock: bool = False,
    ) -> StepLog:
        """Sync execution - implemented by subclasses."""
        ...

    # ═══════════════════════════════════════════════════════════════
    # Async Path
    # ═══════════════════════════════════════════════════════════════

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

### 7. CompositeNode (Updated)

Location: `runnable/nodes.py`

Add async variant for composite node execution:

```python
class CompositeNode(TraversalNode):
    """Base class for nodes that contain sub-graphs."""

    is_composite: bool = Field(default=True, exclude=True)

    # fan_out() and fan_in() stay sync - they only do run log operations

    @abstractmethod
    def execute_as_graph(self, map_variable: MapVariableType = None):
        """Sync execution of sub-graph - override in subclasses."""
        ...

    async def execute_as_graph_async(self, map_variable: MapVariableType = None):
        """
        Async execution of sub-graph.

        Default raises NotImplementedError - override in subclasses
        that support async execution.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement execute_as_graph_async() "
            f"for async execution support."
        )
```

### 8. ParallelNode (Updated)

Location: `extensions/nodes/parallel.py`

Add async variant:

```python
class ParallelNode(CompositeNode):
    """Parallel execution node with sync and async paths."""

    # fan_out() stays sync - just creates branch logs
    # fan_in() stays sync - just collates branch status

    def execute_as_graph(self, map_variable: MapVariableType = None):
        """Sync parallel execution - existing implementation."""
        self.fan_out(map_variable=map_variable)

        for _, branch in self.branches.items():
            self._context.pipeline_executor.execute_graph(
                branch, map_variable=map_variable
            )

        self.fan_in(map_variable=map_variable)

    async def execute_as_graph_async(self, map_variable: MapVariableType = None):
        """Async parallel execution."""
        self.fan_out(map_variable=map_variable)  # sync helper

        for _, branch in self.branches.items():
            await self._context.pipeline_executor.execute_graph_async(
                branch, map_variable=map_variable
            )

        self.fan_in(map_variable=map_variable)  # sync helper
```

### 9. TaskNode (Updated)

Location: `runnable/nodes.py`

Add async execution that checks task type support:

```python
class TaskNode(TraversalNode):
    """Task node with sync and async execution paths."""

    def execute(
        self,
        map_variable: MapVariableType = None,
        attempt_number: int = 1,
        mock: bool = False,
    ) -> StepLog:
        """Sync task execution - existing implementation."""
        step_log = self._create_step_log(map_variable)

        task = self._get_task()
        attempt_log = task.execute_command(map_variable=map_variable)

        step_log.attempts.append(attempt_log)
        step_log.status = attempt_log.status
        return step_log

    async def execute_async(
        self,
        map_variable: MapVariableType = None,
        attempt_number: int = 1,
        mock: bool = False,
    ) -> StepLog:
        """Async task execution."""
        step_log = self._create_step_log(map_variable)

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

### 10. AsyncPythonTaskType

Location: `runnable/tasks.py`

New task type that implements async execution:

```python
class AsyncPythonTaskType(BaseTaskType):
    """
    Task type for executing async Python functions.

    Supports both regular async functions and AsyncGenerator functions
    for streaming use cases.
    """

    task_type: str = Field(default="async-python", serialization_alias="command_type")
    command: str

    # ═══════════════════════════════════════════════════════════════
    # Sync Path - raises error (async functions can't run sync)
    # ═══════════════════════════════════════════════════════════════

    def execute_command(self, map_variable: MapVariableType = None) -> StepAttempt:
        """Sync execution - not supported for async tasks."""
        raise RuntimeError(
            "AsyncPythonTaskType requires async execution. "
            "Use AsyncPipeline or ensure executor calls execute_command_async()."
        )

    # ═══════════════════════════════════════════════════════════════
    # Async Path - the real implementation
    # ═══════════════════════════════════════════════════════════════

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

        with self.execution_context(map_variable=map_variable) as params:
            with self.expose_secrets():
                module, func = utils.get_module_and_attr_names(self.command)
                sys.path.insert(0, os.getcwd())
                imported_module = importlib.import_module(module)
                f = getattr(imported_module, func)

                filtered_parameters = parameters.filter_arguments_for_func(
                    f, params.copy(), map_variable
                )

                # Call the async function
                result = await f(**filtered_parameters)

                # Handle AsyncGenerator (streaming)
                if inspect.isasyncgen(result):
                    accumulated_chunks: List[str] = []
                    async for chunk in result:
                        accumulated_chunks.append(str(chunk))
                        if event_callback:
                            event_callback({
                                "type": "task_chunk",
                                "name": self.command,
                                "chunk": str(chunk),
                            })
                    user_set_parameters = "".join(accumulated_chunks)
                else:
                    user_set_parameters = result

                # Shared helper for return processing
                self._process_returns(user_set_parameters, attempt_log, params, map_variable)
                attempt_log.status = defaults.SUCCESS

        attempt_log.end_time = str(datetime.now())
        return attempt_log
```

### 11. PipelineContext (Updated)

Location: `runnable/context.py`

Add async execute method:

```python
class PipelineContext(BaseModel):
    """Pipeline context with sync and async execution."""

    def execute(self):
        """Sync pipeline execution - existing implementation."""
        assert self.dag is not None

        if self.pipeline_executor._should_setup_run_log_at_traversal:
            self.pipeline_executor._set_up_run_log(exists_ok=False)

        self.pipeline_executor.execute_graph(dag=self.dag)
        self._handle_completion()

    async def execute_async(self):
        """Async pipeline execution."""
        assert self.dag is not None

        if self.pipeline_executor._should_setup_run_log_at_traversal:
            self.pipeline_executor._set_up_run_log(exists_ok=False)

        await self.pipeline_executor.execute_graph_async(dag=self.dag)
        self._handle_completion()  # Sync helper

    def _handle_completion(self):
        """Handle post-execution - shared by sync/async."""
        run_log = self.run_log_store.get_run_log_by_id(
            run_id=self.run_id, full=False
        )

        if run_log.status == defaults.SUCCESS:
            console.print("Pipeline executed successfully!", style=defaults.success_style)
        else:
            console.print("Pipeline execution failed.", style=defaults.error_style)
            raise exceptions.ExecutionFailedError(self.run_id)
```

### 12. SDK Classes

Location: `runnable/sdk.py`

```python
class Pipeline(BaseModel):
    """Pipeline with sync execution (existing behavior unchanged)."""

    def execute(self, ...):
        """Sync execution - existing implementation."""
        context = PipelineContext(...)
        context.execute()


class AsyncPipeline(BaseModel):
    """Pipeline with async execution."""

    async def execute(self, ...):
        """Async execution."""
        context = PipelineContext(
            pipeline_executor={"type": "local"},  # or "async-local"
            ...
        )
        await context.execute_async()

    async def execute_stream(self, ...) -> AsyncGenerator[PipelineEvent, None]:
        """Streaming execution with events."""
        # Implementation for SSE streaming
        ...
```

## Method Naming Convention

| Sync Method | Async Method | Location |
|-------------|--------------|----------|
| `execute_graph()` | `execute_graph_async()` | BasePipelineExecutor |
| `execute_from_graph()` | `execute_from_graph_async()` | BasePipelineExecutor |
| `trigger_node_execution()` | `trigger_node_execution_async()` | BasePipelineExecutor |
| `_execute_node()` | `_execute_node_async()` | BasePipelineExecutor |
| `execute_as_graph()` | `execute_as_graph_async()` | CompositeNode subclasses |
| `execute()` | `execute_async()` | BaseNode, TaskNode |
| `execute_command()` | `execute_command_async()` | BaseTaskType subclasses |
| `execute()` | `execute_async()` | PipelineContext |

## Error Handling for Unsupported Async

| Component | Async Method Called On | Error |
|-----------|------------------------|-------|
| ArgoExecutor | `execute_graph_async()` | `NotImplementedError: ArgoExecutor does not support async execution` |
| NotebookTaskType | `execute_command_async()` | `NotImplementedError: NotebookTaskType does not support async execution` |
| ShellTaskType | `execute_command_async()` | `NotImplementedError: ShellTaskType does not support async execution` |

## Shared Helpers (Sync)

These methods contain the actual logic and are called by both sync and async paths:

| Helper Method | Purpose | Location |
|---------------|---------|----------|
| `_prepare_node_for_execution()` | Setup before node execution | GenericPipelineExecutor |
| `_finalize_graph_execution()` | Cleanup after graph traversal | GenericPipelineExecutor |
| `_get_status_and_next_node_name()` | Determine next node | GenericPipelineExecutor |
| `_sync_catalog()` | Catalog get/put operations | GenericPipelineExecutor |
| `_calculate_attempt_number()` | Calculate retry attempt | GenericPipelineExecutor |
| `fan_out()` | Create branch logs | CompositeNode |
| `fan_in()` | Collate branch status | CompositeNode |
| `_handle_completion()` | Post-execution handling | PipelineContext |
| `_process_returns()` | Handle task returns | BaseTaskType |
| `execution_context()` | Parameter load/store | BaseTaskType |
| `expose_secrets()` | Secret management | BaseTaskType |

## Plugin Registration

In `pyproject.toml`:

```toml
[project.entry-points.'tasks']
async-python = "runnable.tasks:AsyncPythonTaskType"

# No new executor needed - LocalExecutor now supports async
# Optionally register async-local as alias
[project.entry-points.'pipeline_executor']
async-local = "extensions.pipeline_executor.local:LocalExecutor"
```

## Streaming Events

**AsyncGenerator approach** - events flow out to client, data flows internally:

```
┌─────────────────────────────────────────────────────────────────┐
│                          LocalExecutor                          │
│                                                                 │
│  Task 1 (generate_text)          Task 2 (process_result)       │
│  ┌─────────────────────┐         ┌─────────────────────┐       │
│  │ yield "Hello"       │         │                     │       │
│  │ yield " world"      │         │ text = "Hello world"│       │
│  │ yield "!"           │         │ return {"len": 11}  │       │
│  └─────────────────────┘         └─────────────────────┘       │
│           │                               ▲                     │
│           │ accumulate internally         │                     │
│           └──────────────────────────────►│                     │
│                    "Hello world"    (via params/run_log_store) │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
           │ yield events
           ▼
    ┌─────────────────┐
    │   FastAPI SSE   │
    └─────────────────┘
```

**Event Types:**

| Type | Description |
|------|-------------|
| `pipeline_started` | Pipeline execution began |
| `task_started` | Task began execution |
| `task_chunk` | Chunk yielded from async generator task |
| `task_completed` | Task completed (includes outputs) |
| `task_error` | Task failed |
| `pipeline_completed` | Pipeline finished |

## Usage Examples

### Sync execution (unchanged)

```python
from runnable import Pipeline, PythonTask

def my_func(x: int) -> int:
    return x * 2

pipeline = Pipeline(
    steps=[PythonTask(function=my_func, name="step1", returns=["result"])]
)
pipeline.execute()  # Sync - works exactly as before
```

### Async execution

```python
from runnable import AsyncPipeline, AsyncPythonTask

async def fetch_data(url: str) -> dict:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()

pipeline = AsyncPipeline(
    steps=[AsyncPythonTask(function=fetch_data, name="fetch", returns=["data"])]
)
await pipeline.execute()  # Async
```

### Streaming execution

```python
@app.post("/generate")
async def generate(request: Request):
    pipeline = AsyncPipeline(steps=[...])

    async def event_stream():
        async for event in pipeline.execute_stream():
            yield f"data: {event.model_dump_json()}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
```

## Out of Scope

- Async notebook/shell tasks (inherently sync - raise NotImplementedError)
- Async container/argo executors (transpilers don't execute - raise NotImplementedError)
- Changes to run log store or catalog (stay sync)
- Async parameter handling

## Testing Strategy

1. Unit tests for shared helpers (called by both paths)
2. Unit tests for sync path (ensure no regression)
3. Unit tests for async path (new functionality)
4. Unit tests for NotImplementedError on unsupported executors/tasks
5. Integration tests with composite nodes (parallel, map)
6. Integration tests for streaming
7. Test error handling and retry logic in both paths
