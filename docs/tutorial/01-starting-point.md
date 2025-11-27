# The Starting Point

Let's start with a realistic scenario: you have a machine learning function that works great on your laptop, but suffers from common problems that prevent it from being production-ready.

## The Problem

Here's a typical ML training function that many data scientists write:

```python title="examples/tutorials/getting-started/functions.py (excerpt)"
def train_ml_model_basic():
    """
    Basic ML training function - works locally but has typical problems:
    - Hardcoded parameters
    - No tracking of runs
    - Results get overwritten
    - No reproducibility guarantees
    """
    print("Loading data...")
    df = load_data("data.csv")

    print("Preprocessing...")
    preprocessed = preprocess_data(df, test_size=0.2, random_state=42)

    print("Training model...")
    model_data = train_model(preprocessed, n_estimators=100, random_state=42)

    print("Evaluating...")
    results = evaluate_model(model_data, preprocessed)

    print(f"Accuracy: {results['accuracy']:.4f}")

    # Save everything (gets overwritten each run!)
    save_model(model_data, "model.pkl")
    save_results(results, "results.json")

    return results
```

**Try it yourself:**

```bash
uv run examples/tutorials/getting-started/01_starting_point.py
```

## What's Wrong Here?

This function works, but it has several problems that will bite you in production:

### 🚫 **No Execution Tracking**

- When did you run this?
- What were the exact parameters?
- Which version of the code produced these results?

### 🚫 **Results Get Overwritten**

- Run it twice → lose the first results
- No way to compare different experiments
- Can't track model performance over time

### 🚫 **Hardcoded Parameters**

- Want to try different `n_estimators`? Edit the code
- Want different train/test split? Edit the code
- Testing becomes cumbersome and error-prone

### 🚫 **No Reproducibility**

- Even with `random_state`, environment differences can cause variations
- No record of what Python packages were used
- Impossible to recreate exact results months later

### 🚫 **Hard to Share and Deploy**

- How do you run this in a container?
- What about on Kubernetes?
- Sharing with colleagues means sharing your entire environment

## The Real Impact

These aren't just theoretical problems. In real projects, this leads to:

- **"Which model was that?"** - Lost track of good results
- **"I can't reproduce the paper results"** - Different environments, different outcomes
- **"It worked yesterday"** - No history of what changed
- **"How do I run this in production?"** - Deployment becomes a separate project

## What We'll Build

Throughout this tutorial, we'll transform this exact function into a production-ready ML pipeline that solves all these problems:

✅ **Automatic execution tracking** - Every run logged with timestamps and parameters

✅ **Result preservation** - All experiments saved and easily comparable

✅ **Flexible configuration** - Change parameters without touching code

✅ **Full reproducibility** - Recreate exact results anytime, anywhere

✅ **Deploy anywhere** - Same code runs on laptop, containers, Kubernetes

**Your functions won't change** - we'll just wrap them with Runnable patterns.

---

**Next:** [Making It Reproducible](02-making-it-reproducible.md) - Add automatic tracking without changing your ML logic
