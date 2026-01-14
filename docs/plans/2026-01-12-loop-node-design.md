# Loop Node Design

## Overview

Add a loop node to runnable's composite node capabilities. The loop node iterates over a branch until a break condition is met or max iterations is reached.

## Requirements

- Similar to composite nodes (parallel, map, conditional)
- Iterate until:
  - `max_iterations` is reached, OR
  - A boolean parameter (`break_on`) evaluates to `True`
- Do-while style: branch executes at least once, condition checked after each iteration
- Expose iteration index as a plain environment variable
- Parameters update in place (last iteration wins)
- Full lineage tracking via separate branch logs per iteration
- Compatible with Argo workflows via recursive template pattern

## Node Definition

```python
class LoopNode(CompositeNode):
    node_type: str = Field(default="loop", serialization_alias="type")

    # The sub-graph to execute repeatedly
    branch: Graph

    # Maximum iterations (safety limit)
    max_iterations: int

    # Boolean parameter name - when True, loop exits
    break_on: str

    # Environment variable name for iteration index (no prefix)
    index_as: str
```

## SDK Interface

```python
from runnable import Pipeline, PythonTask, Loop

process = PythonTask(
    name="process",
    function=process_func,
    returns=[pickled("result"), json("should_stop")]
)

# Branch must be a Pipeline (consistent with other composite nodes)
process_pipeline = Pipeline(steps=[process])

loop = Loop(
    name="retry_loop",
    branch=process_pipeline,
    max_iterations=5,
    break_on="should_stop",
    index_as="attempt_num"  # env var: attempt_num=0,1,2...
)

pipeline = Pipeline(steps=[loop, ...])
```

## Termination Logic

The loop terminates when EITHER condition is met:

1. `parameters[break_on] == True`
2. `current_iteration >= max_iterations`

## Naming Convention

Following the existing dot-path convention:

| Node Type | Pattern | Example |
|-----------|---------|---------|
| Parallel | `node.branch_name.step` | `parallel_node.branch_a.task1` |
| Map | `node.iter_value.step` | `map_node.chunk1.task1` |
| Loop | `node.iteration_index.step` | `loop_node.0.task1` |

For a loop node named `retry_loop` with a task `process_data`:

- Iteration 0: `retry_loop.0.process_data`
- Iteration 1: `retry_loop.1.process_data`
- Iteration 2: `retry_loop.2.process_data`

Branch log names:

- `retry_loop.0`
- `retry_loop.1`
- `retry_loop.2`

## Parameter Flow

1. **fan_out**: Copy parent parameters to branch scope for iteration 0

2. **Each iteration**:
   - Branch reads/writes parameters in branch scope
   - Parameters persist in run_log_store between iterations
   - Each iteration sees parameters left by previous iteration
   - On continue: copy params from `node.N` scope to `node.N+1` scope

3. **fan_in (final)**: Copy branch parameters back to parent scope

## Method Responsibilities

### fan_out()

- Create branch log for iteration 0: `loop_node.0`
- Copy parent parameters to iteration 0 branch scope
- Called once at the start

### execute_as_graph()

Local execution loop:

```python
def execute_as_graph(self, iter_variable=None):
    self.fan_out(iter_variable)

    iteration = 0
    while True:
        # Set iteration index env var
        os.environ[self.index_as] = str(iteration)

        # Execute branch
        self._context.pipeline_executor.execute_graph(
            self.branch,
            iter_variable=build_iter_variable(iter_variable, iteration)
        )

        # Check termination
        should_exit = self.fan_in(iter_variable, iteration)

        if should_exit:
            break

        iteration += 1
        # Create next branch log, copy params
        self._create_next_iteration(iteration, iter_variable)
```

### fan_in()

- Check `parameters[break_on] == True`
- Check `iteration >= max_iterations - 1` (0-indexed)
- Compute `should_exit = break_condition OR max_reached`
- If `should_exit`:
  - Set step_log status based on branch status
  - Roll back parameters to parent scope
- Return `should_exit` flag
- For Argo: write `should_exit` to `/tmp/output.txt`

## IterableParameterModel Integration

Use the existing `loop_variable` field in `IterableParameterModel`:

```python
class IterableParameterModel(BaseModel):
    map_variable: OrderedDict[str, MapVariableModel] | None
    loop_variable: list[LoopIndexModel] | None  # Use this for loop iterations
```

Build iter_variable for nested loops:

```python
def build_iter_variable(parent_iter_variable, iteration):
    iter_var = parent_iter_variable.model_copy(deep=True) if parent_iter_variable else IterableParameterModel()
    iter_var.loop_variable = iter_var.loop_variable or []
    iter_var.loop_variable.append(LoopIndexModel(value=iteration))
    return iter_var
```

## Argo Implementation

### Template Structure

```
loop-node-template:
  tasks:
    - fan-out (initialize)
    - loop-body (depends: fan-out, inputs: loop_index=0)

loop-body-template (inputs: loop_index):
  tasks:
    - branch (execute sub-graph, env: INDEX_AS=loop_index)
    - fan-in (depends: branch)
        → outputs: should_exit
    - recurse (depends: fan-in, when: should_exit != "true")
        → calls loop-body-template with loop_index + 1
```

### fan_out Container

- Initializes branch log for iteration 0
- Copies parent parameters to branch scope

### fan_in Container

- Reads `break_on` parameter from run_log_store
- Checks if `loop_index >= max_iterations - 1`
- Writes `should_exit` ("true" or "false") to `/tmp/output.txt`
- If exiting: rolls back parameters to parent scope

### Recurse Task

- Conditional on `should_exit != "true"`
- Calls `loop-body-template` with `loop_index + 1`
- Argo handles the recursion natively

## File Changes Required

### New Files

- `extensions/nodes/loop.py` - LoopNode implementation

### Modified Files

- `runnable/sdk.py` - Add `Loop` class for SDK interface
- `pyproject.toml` - Register loop node entry point
- `extensions/pipeline_executor/argo.py` - Add loop node handling in `_gather_tasks_for_dag_template`
- `runnable/context.py` - Add fan command support for loop node (if needed)

## Testing Strategy

1. **Unit tests** (`tests/extensions/nodes/test_loop.py`):
   - Test termination on break condition
   - Test termination on max_iterations
   - Test parameter flow between iterations
   - Test nested loops

2. **Integration tests**:
   - Local execution with break condition
   - Local execution hitting max_iterations
   - Loop with multi-step branch

3. **Argo tests**:
   - Verify generated YAML structure
   - Test recursive template generation

## Example Use Cases

### Retry Pattern

```python
def attempt_operation(attempt_num):
    # attempt_num available as env var
    result = call_external_api()
    return {"success": result.ok, "should_stop": result.ok}

task = PythonTask(
    name="call_api",
    function=attempt_operation,
    returns=[json("success"), json("should_stop")]
)

retry_pipeline = Pipeline(steps=[task])

retry_loop = Loop(
    name="retry",
    branch=retry_pipeline,
    max_iterations=3,
    break_on="should_stop",
    index_as="attempt_num"
)
```

### Convergence Loop

```python
def train_epoch(epoch):
    # epoch available as env var
    loss = train_model()
    return {"loss": loss, "converged": loss < 0.01}

training = PythonTask(
    name="train",
    function=train_epoch,
    returns=[json("loss"), json("converged")]
)

training_pipeline = Pipeline(steps=[training])

training_loop = Loop(
    name="training",
    branch=training_pipeline,
    max_iterations=100,
    break_on="converged",
    index_as="epoch"
)
```
