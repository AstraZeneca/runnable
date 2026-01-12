# Branch Parameter Rollback Implementation Plan

> **For Claude:** This work is complete in conditional.py and parallel.py. Only need end-to-end examples and assertions.

**Goal:** Create end-to-end testable pipeline examples demonstrating that parameters set in conditional and parallel branches correctly roll back to parent scope after successful execution.

**Context:** Parameter rollback has been implemented in both ConditionalNode and ParallelNode fan_in() methods. When branches execute successfully, their parameters are merged into the parent scope. This is analogous to the map node's reducer pattern, but without reduction - just a simple merge/overwrite.

**Architecture:** Follow the existing pattern in test_pipeline_examples.py where examples are Python files in examples/ directory with assertions that verify run log state after execution.

---

## Critical Files

### To Create:
- `examples/10-branch-parameters/conditional_rollback.py` - Conditional parameter rollback example
- `examples/10-branch-parameters/parallel_rollback.py` - Parallel parameter rollback example
- `examples/common/functions.py` - Add helper functions (modify existing file)
- `tests/assertions.py` - Add root parameter assertion (modify existing file)

### To Modify:
- `tests/test_pipeline_examples.py` - Add new examples to python_examples list

---

## Implementation Steps

### Step 1: Add Assertion for Root Parameters

**File:** `tests/assertions.py`

Add new assertion function to verify parameters exist at root level:

```python
def should_have_root_parameters(parameters: dict):
    """Verify parameters exist at run log root level with expected values."""
    from runnable import defaults

    run_id = os.environ[defaults.ENV_RUN_ID]
    run_log = load_run_log(run_id)

    for param_name, expected_value in parameters.items():
        assert param_name in run_log.parameters, f"Parameter {param_name} not found in root parameters"
        actual_value = run_log.parameters[param_name].get_value()
        assert actual_value == expected_value, f"Expected {param_name}={expected_value}, got {actual_value}"
```

### Step 2: Add Helper Functions to Common Functions

**File:** `examples/common/functions.py`

Add functions for conditional branches:

```python
def set_conditional_heads_param():
    """Set parameter when heads branch executes."""
    return "heads_value"


def set_conditional_tails_param():
    """Set parameter when tails branch executes."""
    return "tails_value"


def set_conditional_multiple():
    """Set multiple parameters in conditional branch."""
    return "param1_value", "param2_value", "param3_value"


def verify_conditional_rollback(branch_param: str):
    """Verify rolled back parameter from conditional branch."""
    assert branch_param in ["heads_value", "tails_value"]
    return branch_param
```

Add functions for parallel branches:

```python
def set_parallel_branch1():
    """Set parameter in parallel branch 1."""
    return "branch1_value"


def set_parallel_branch2():
    """Set parameter in parallel branch 2."""
    return "branch2_value"


def set_parallel_branch3():
    """Set parameter in parallel branch 3."""
    return "branch3_value"


def verify_parallel_rollback(result1: str, result2: str, result3: str):
    """Verify all parallel branch parameters rolled back."""
    assert result1 == "branch1_value"
    assert result2 == "branch2_value"
    assert result3 == "branch3_value"
    return "verified"


def set_shared_param_a():
    """Set shared parameter to value A."""
    return "value_a"


def set_shared_param_b():
    """Set shared parameter to value B."""
    return "value_b"
```

### Step 3: Create Conditional Parameter Rollback Example

**File:** `examples/10-branch-parameters/conditional_rollback.py`

```python
"""
Demonstrate parameter rollback from conditional branches.

When a conditional branch executes and succeeds, parameters set within
that branch roll back to the parent scope.

Execute with:
    python examples/10-branch-parameters/conditional_rollback.py
"""

from examples.common.functions import (
    set_conditional_heads_param,
    set_conditional_tails_param,
    verify_conditional_rollback,
)
from runnable import Conditional, Pipeline, PythonTask, Stub


def decide_heads():
    """Return 'heads' to select heads branch."""
    return "heads"


def main():
    # Create branch pipelines that set parameters
    heads_pipeline = PythonTask(
        name="heads_task",
        function=set_conditional_heads_param,
        returns=["branch_param"],
    ).as_pipeline()

    tails_pipeline = PythonTask(
        name="tails_task",
        function=set_conditional_tails_param,
        returns=["branch_param"],
    ).as_pipeline()

    # Conditional node selects branch based on 'choice' parameter
    conditional = Conditional(
        name="conditional",
        branches={"heads": heads_pipeline, "tails": tails_pipeline},
        parameter="choice",
    )

    # Task to set the choice parameter
    decide_task = PythonTask(
        name="decide",
        function=decide_heads,
        returns=["choice"],
    )

    # Task to verify the parameter rolled back from branch
    verify_task = PythonTask(
        name="verify",
        function=verify_conditional_rollback,
        terminate_with_success=True,
    )

    pipeline = Pipeline(steps=[decide_task, conditional, verify_task])
    pipeline.execute()

    return pipeline


if __name__ == "__main__":
    main()
```

### Step 4: Create Parallel Parameter Rollback Example

**File:** `examples/10-branch-parameters/parallel_rollback.py`

```python
"""
Demonstrate parameter rollback from parallel branches.

When parallel branches execute and succeed, parameters from all branches
roll back to the parent scope.

Execute with:
    python examples/10-branch-parameters/parallel_rollback.py
"""

from examples.common.functions import (
    set_parallel_branch1,
    set_parallel_branch2,
    set_parallel_branch3,
    verify_parallel_rollback,
)
from runnable import Parallel, Pipeline, PythonTask, Stub


def main():
    # Create branch pipelines that set different parameters
    branch1_pipeline = PythonTask(
        name="branch1_task",
        function=set_parallel_branch1,
        returns=["result1"],
    ).as_pipeline()

    branch2_pipeline = PythonTask(
        name="branch2_task",
        function=set_parallel_branch2,
        returns=["result2"],
    ).as_pipeline()

    branch3_pipeline = PythonTask(
        name="branch3_task",
        function=set_parallel_branch3,
        returns=["result3"],
    ).as_pipeline()

    # Parallel node executes all branches
    parallel = Parallel(
        name="parallel",
        branches={
            "branch1": branch1_pipeline,
            "branch2": branch2_pipeline,
            "branch3": branch3_pipeline,
        },
    )

    # Task to verify all parameters rolled back from branches
    verify_task = PythonTask(
        name="verify",
        function=verify_parallel_rollback,
        terminate_with_success=True,
    )

    pipeline = Pipeline(steps=[parallel, verify_task])
    pipeline.execute()

    return pipeline


if __name__ == "__main__":
    main()
```

### Step 5: Create Parameter Conflict Example (Parallel)

**File:** `examples/10-branch-parameters/parallel_conflict.py`

```python
"""
Demonstrate parameter conflict resolution in parallel branches.

When multiple branches set the same parameter, last write wins
based on dictionary iteration order.

Execute with:
    python examples/10-branch-parameters/parallel_conflict.py
"""

from examples.common.functions import set_shared_param_a, set_shared_param_b
from runnable import Parallel, Pipeline, PythonTask


def main():
    # Both branches set the same parameter name
    branch1_pipeline = PythonTask(
        name="branch1_task",
        function=set_shared_param_a,
        returns=["shared"],
    ).as_pipeline()

    branch2_pipeline = PythonTask(
        name="branch2_task",
        function=set_shared_param_b,
        returns=["shared"],
    ).as_pipeline()

    parallel = Parallel(
        name="parallel",
        branches={"branch1": branch1_pipeline, "branch2": branch2_pipeline},
    )

    pipeline = Pipeline(steps=[parallel])
    pipeline.execute()

    return pipeline


if __name__ == "__main__":
    main()
```

### Step 6: Add Examples to Test Suite

**File:** `tests/test_pipeline_examples.py`

Add to `python_examples` list (after the last entry, before contexts are defined):

```python
    (
        "10-branch-parameters/conditional_rollback",
        True,  # no_yaml
        False,  # fails
        [],  # ignore_contexts
        "",  # parameters_file
        [
            partial(conditions.should_have_num_steps, 4),
            partial(conditions.should_be_successful),
            partial(conditions.should_step_be_successful, "decide"),
            partial(conditions.should_step_be_successful, "conditional"),
            partial(conditions.should_step_be_successful, "verify"),
            partial(conditions.should_branch_have_steps, "conditional", "heads", 2),
            partial(conditions.should_branch_be_successful, "conditional", "heads"),
            # Verify parameter rolled back to root
            partial(conditions.should_have_root_parameters, {"branch_param": "heads_value"}),
        ],
    ),
    (
        "10-branch-parameters/parallel_rollback",
        True,  # no_yaml
        False,  # fails
        [],  # ignore_contexts
        "",  # parameters_file
        [
            partial(conditions.should_have_num_steps, 3),
            partial(conditions.should_be_successful),
            partial(conditions.should_step_be_successful, "parallel"),
            partial(conditions.should_step_be_successful, "verify"),
            partial(conditions.should_branch_have_steps, "parallel", "branch1", 2),
            partial(conditions.should_branch_have_steps, "parallel", "branch2", 2),
            partial(conditions.should_branch_have_steps, "parallel", "branch3", 2),
            partial(conditions.should_branch_be_successful, "parallel", "branch1"),
            partial(conditions.should_branch_be_successful, "parallel", "branch2"),
            partial(conditions.should_branch_be_successful, "parallel", "branch3"),
            # Verify all parameters rolled back to root
            partial(
                conditions.should_have_root_parameters,
                {"result1": "branch1_value", "result2": "branch2_value", "result3": "branch3_value"},
            ),
        ],
    ),
    (
        "10-branch-parameters/parallel_conflict",
        True,  # no_yaml
        False,  # fails
        [],  # ignore_contexts
        "",  # parameters_file
        [
            partial(conditions.should_have_num_steps, 2),
            partial(conditions.should_be_successful),
            partial(conditions.should_step_be_successful, "parallel"),
            partial(conditions.should_branch_have_steps, "parallel", "branch1", 2),
            partial(conditions.should_branch_have_steps, "parallel", "branch2", 2),
            # Verify one of the values is present (last write wins)
            # We can't assert which specific value because dict iteration order
            # Just verify the parameter exists at root
            partial(conditions.should_have_root_parameters, {"shared": "value_b"}),
        ],
    ),
```

---

## Verification

### Manual Testing

Run the examples directly:

```bash
# Test conditional rollback
uv run python examples/10-branch-parameters/conditional_rollback.py

# Test parallel rollback
uv run python examples/10-branch-parameters/parallel_rollback.py

# Test parallel conflict
uv run python examples/10-branch-parameters/parallel_conflict.py
```

### Automated Testing

Run the test suite:

```bash
# Run only the new parameter rollback tests
uv run pytest tests/test_pipeline_examples.py::test_python_examples -k "branch-parameters" -v

# Run all example tests
uv run pytest tests/test_pipeline_examples.py -v
```

### Expected Outcomes

1. **Conditional Rollback:**
   - `decide` task sets `choice="heads"`
   - Conditional executes `heads` branch which sets `branch_param="heads_value"`
   - After fan_in, `branch_param` exists in root parameters
   - `verify` task receives `branch_param` and succeeds

2. **Parallel Rollback:**
   - All 3 branches execute in parallel
   - Each sets its own unique parameter (`result1`, `result2`, `result3`)
   - After fan_in, all 3 parameters exist in root
   - `verify` task receives all 3 parameters and succeeds

3. **Parallel Conflict:**
   - Both branches set `shared` parameter
   - After fan_in, `shared` exists with one of the values (dict iteration determines which)
   - Parameter exists at root level

### What to Look For

- All tests pass without errors
- Run logs show parameters at root level after composite nodes
- Branch parameters are properly isolated during execution
- Verification tasks can access rolled-back parameters
- No test failures in other examples (no regressions)

---

## Notes

- The implementation in `conditional.py` and `parallel.py` is already complete
- Only failure handling differs: conditional has one executed branch, parallel has multiple
- Both use same pattern: get parent params → merge branch params → set back to parent
- Map node has similar logic but with reducer function - this is just direct merge
- Examples follow existing patterns from `07-map` and `02-sequential` directories
- Integration tests were removed - examples with assertions are the preferred testing approach
