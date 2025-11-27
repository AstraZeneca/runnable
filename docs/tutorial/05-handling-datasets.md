# Handling Large Datasets

Our pipeline is working great with small datasets that fit in memory. But what happens when your dataset is 100GB? Or when preprocessing generates gigabytes of intermediate results? Let's solve this with efficient file-based storage.

## The Memory Problem

In Chapter 4, we passed data between steps using `pickled()`:

```python
PythonTask(
    function=preprocess_data,
    returns=[pickled("preprocessed_data")]  # All data kept in memory!
)
```

**Problems with this approach:**

- Large datasets won't fit in memory
- Pickling/unpickling is slow for big objects
- Can't easily inspect intermediate results
- Memory pressure on your system

## The Solution: Catalog for File Storage

Instead of passing data through memory, save it to files and let Runnable manage them:

```python title="examples/tutorials/getting-started/05_handling_datasets.py" hl_lines="11 19-20"
from runnable import Pipeline, PythonTask, Catalog, pickled

def load_data_to_file(data_path="data.csv"):
    """Load data and save to file."""
    df = load_data(data_path)
    df.to_csv("dataset.csv", index=False)
    return {"rows": len(df), "columns": len(df.columns)}

# Store the dataset file automatically
PythonTask(
    function=load_data_to_file,
    name="load_data",
    catalog=Catalog(put=["dataset.csv"]),  # Store this file
    returns=[pickled("dataset_info")]  # Only metadata in memory
)
```

**Try it:**

```bash
uv run examples/tutorials/getting-started/05_handling_datasets.py
```

## How Catalog Works

### Step 1: Create and Store Files

```python
def preprocess_from_file(test_size=0.2, random_state=42):
    # Load from file
    df = pd.read_csv("dataset.csv")

    # Do your preprocessing
    preprocessed = preprocess_data(df, test_size, random_state)

    # Save results to files
    preprocessed['X_train'].to_csv("X_train.csv", index=False)
    preprocessed['X_test'].to_csv("X_test.csv", index=False)
    preprocessed['y_train'].to_csv("y_train.csv", index=False)
    preprocessed['y_test'].to_csv("y_test.csv", index=False)

    return {"train_samples": len(preprocessed['X_train'])}

PythonTask(
    function=preprocess_from_file,
    name="preprocess",
    catalog=Catalog(
        get=["dataset.csv"],  # Get input file
        put=["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]  # Store outputs
    )
)
```

### Step 2: Retrieve and Use Files

```python
def train_from_files(n_estimators=100, random_state=42):
    # Files are automatically available!
    X_train = pd.read_csv("X_train.csv")
    y_train = pd.read_csv("y_train.csv")['target']

    # Train your model
    model = train_model(...)
    return model

PythonTask(
    function=train_from_files,
    name="train",
    catalog=Catalog(get=["X_train.csv", "y_train.csv"])  # Get only what you need
)
```

## Complete File-Based Pipeline

Here's the full pipeline using file storage for large data:

```python title="examples/tutorials/getting-started/05_handling_datasets.py"
--8<-- "examples/tutorials/getting-started/05_handling_datasets.py:62:119"
```

## What You Get with File-Based Storage

### 💾 **Handle Large Datasets**

Your dataset can be bigger than available RAM - only load what you need when you need it:

```python
# Only load training data for training step
X_train = pd.read_csv("X_train.csv")  # Maybe 50GB
# X_test isn't loaded - saves memory!
```

### 🔄 **Automatic File Management**

Runnable handles file locations transparently:

- **`put=["file.parquet"]`** - Stores file safely in `.runnable/` catalog
- **`get=["file.parquet"]`** - Makes file available in your working directory
- Files appear exactly where your code expects them

### 📦 **Inspect Intermediate Results**

All intermediate files are preserved:

```bash
# Check what preprocessing produced
ls .runnable/catalog/
# X_train.csv  X_test.csv  y_train.csv  y_test.csv
```

### 🚀 **Resume Without Reloading**

If training fails, you don't need to reload and preprocess your 100GB dataset - it's already there!

### 🤝 **Share Results**

Team members can reuse your preprocessed data without running expensive preprocessing steps.

## When to Use Files vs Memory

**Use `Catalog(put=[...])` for files when:**

- Dataset is large (>1GB)
- Preprocessing is expensive
- You want to inspect intermediate results
- Team members need to share data

**Use `pickled()` for memory when:**

- Data is small (<100MB)
- Objects are complex (models, configs)
- You need fast passing between steps

## Mixing Files and Memory

You can use both approaches in the same pipeline:

```python
pipeline = Pipeline(steps=[
    PythonTask(
        function=load_data_to_file,
        catalog=Catalog(put=["dataset.csv"]),  # Large data → file
        returns=[pickled("metadata")]  # Small metadata → memory
    ),
    PythonTask(
        function=train_from_files,
        catalog=Catalog(get=["dataset.csv"]),  # Get large data from file
        returns=[pickled("model")]  # Model usually fits in memory
    )
])
```

## Compare: Memory vs File Storage

**Memory Passing (Chapters 1-4):**

- ❌ Limited by available RAM
- ❌ Slow for large objects
- ❌ Hard to inspect intermediate data
- ✅ Simple for small objects
- ✅ Fast for small data

**File Storage (Chapter 5):**

- ✅ Handle datasets larger than RAM
- ✅ Efficient for large files
- ✅ Easy to inspect intermediate results
- ✅ Shareable across runs and team members
- ✅ Automatic file management

## What's Next?

We can now handle large datasets efficiently. But what about saving your trained models and results permanently? What if your teammate wants to use your model without rerunning everything?

**Next chapter:** We'll add persistent storage for models and results that can be shared across runs and team members.

---

**Next:** [Sharing Results](06-sharing-results.md) - Persistent model artifacts and metrics
