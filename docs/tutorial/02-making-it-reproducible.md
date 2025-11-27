# Making It Reproducible

Now let's solve the first major problem: lack of execution tracking. We'll transform our basic function into a reproducible, tracked job without changing the ML logic at all.

## The Solution: PythonJob

Instead of calling our function directly, we'll wrap it with Runnable's `PythonJob`:

```python title="examples/tutorials/getting-started/02_making_it_reproducible.py"
from runnable import PythonJob
from functions import train_ml_model_basic

def main():
    # Same function, now wrapped as a Job
    job = PythonJob(function=train_ml_model_basic)
    job.execute()
    return job

if __name__ == "__main__":
    main()
```

**Try it:**

```bash
uv run examples/tutorials/getting-started/02_making_it_reproducible.py
```

## What Just Happened?

Your ML function ran exactly the same way, but Runnable automatically added powerful tracking capabilities:

### 📝 **Execution Logging**

Every run gets logged with complete details:

```bash
ls .run_log_store/
# Shows JSON files with run IDs: curious-jang-0442.json
```

Each run directory contains:

- **Execution metadata**: when it ran, how long it took
- **Environment info**: Python version, package versions
- **Results**: function return values
- **Status**: success/failure with any error details

### ♻️ **Result Preservation**

Unlike the basic version that overwrote `model.pkl` and `results.json`, each Runnable execution gets its own directory. Your results are never lost.

### 🔍 **Full Reproducibility**

Each run captures everything needed to reproduce it:

- Exact timestamp
- Code version (if using git)
- Environment details
- Input parameters (we'll add those next!)

### 🎯 **Zero Code Changes**

Notice that `train_ml_model_basic()` didn't change at all. Runnable works with your existing functions - no decorators, no API changes, no refactoring required.

## Run It Multiple Times

Try running the script several times:

```bash
uv run examples/tutorials/getting-started/02_making_it_reproducible.py
uv run examples/tutorials/getting-started/02_making_it_reproducible.py
uv run examples/tutorials/getting-started/02_making_it_reproducible.py
```

Each run creates a separate log entry in `.run_log_store/`. You now have a complete history of all your experiments!

## Compare: Before vs After

**Before (Chapter 1):**

- ❌ Results overwritten each time
- ❌ No execution history
- ❌ No timestamps or metadata
- ❌ Hard to track what worked

**After (Chapter 2):**

- ✅ Every run preserved with timestamp
- ✅ Complete execution history
- ✅ Full metadata captured automatically
- ✅ Easy to see what worked when

## What's Still Missing?

We solved execution tracking, but we still have:

- Parameters hardcoded in the function
- No easy way to run experiments with different settings

**Next chapter:** We'll make the function configurable without changing the ML logic.

---

**Next:** [Adding Flexibility](03-adding-flexibility.md) - Configure experiments without touching code
