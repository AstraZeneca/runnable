# 🔗 Connecting Functions

The magic happens when you chain functions together. Runnable makes this effortless.

## What you already know

You probably chain functions like this:

```python
def create_data():
    return pd.DataFrame({"x": [1, 2, 3]})

def analyze_data(df):
    return df.mean()

# Manual chaining
df = create_data()
result = analyze_data(df)
```

## Runnable does the chaining for you

Same functions, automatic parameter passing:

```python linenums="1"
--8<-- "examples/03-parameters/passing_parameters_python.py:26:43"
```

✨ **Magic**: The `df` returned by `write_parameter` automatically becomes the `df` parameter for `read_parameter`.

## How it works

1. **Step 1** returns values with names: `returns=["df", "score"]`
2. **Step 2** function signature: `def analyze(df, score):`
3. **Runnable matches** return names to parameter names automatically

## Multiple parameters flow together

```python linenums="1"
--8<-- "examples/common/functions.py:85:99"
```

When chained in a pipeline:

```python
step1 = PythonTask(
    function=write_parameter,
    returns=["df", "integer", "floater", "stringer", "pydantic_param", "score"]
)

step2 = PythonTask(function=read_parameter)  # Gets all 6 parameters automatically
```

## Mix different task types

Python functions, notebooks, and shell scripts all work together:

```python linenums="1"
--8<-- "examples/02-sequential/traversal.py:18:27"
```

!!! tip "Parameter matching"

    Return names must match parameter names. `returns=["data"]` → `def process(data):`

Next: Understand [when to use jobs vs pipelines](../concepts/building-blocks/jobs-vs-pipelines.md).
