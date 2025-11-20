# File Storage 📁

Automatically store files created during Job execution using the Catalog system.

## Basic File Storage

Jobs can capture and store files your function creates:

```python
from examples.common.functions import write_files
from runnable import Catalog, PythonJob

def main():
    write_catalog = Catalog(put=["df.csv", "data_folder/data.txt"])
    job = PythonJob(
        function=write_files,
        catalog=write_catalog,
    )

    job.execute()
    return job

if __name__ == "__main__":
    main()
```

??? example "See complete runnable code"
    ```python title="examples/11-jobs/catalog.py"
    --8<-- "examples/11-jobs/catalog.py"
    ```

    **Try it now:**
    ```bash
    uv run examples/11-jobs/catalog.py
    ```

## What Happens

**Function Creates Files:**
- `df.csv` in working directory
- `data_folder/data.txt` in subdirectory

**Catalog Stores Copies:**
```
.catalog/unsolvable-ramanujan-0634/
├── df.csv                    # Copied CSV file
├── data_folder/
│   └── data.txt             # Copied text file
└── jobBGR.execution.log     # Execution log
```

**Summary Shows:**
```json
{
    "Output catalog content": ["df.csv", "data_folder/data.txt"],
    "status": "SUCCESS"
}
```

## Copy vs No-Copy Modes

### Copy Mode (Default)
```python
# Files are copied to catalog
Catalog(put=["results.csv", "model.pkl"])
# Same as: Catalog(put=["results.csv", "model.pkl"], store_copy=True)
```

- ✅ Files copied to `.catalog/{run-id}/`
- ✅ Original files remain in working directory
- ✅ Full file versioning and backup

### No-Copy Mode (Hash Only)
```python
# Files are tracked but not copied
Catalog(put=["large_dataset.csv", "model.pkl"], store_copy=False)
```

??? example "See complete runnable code"
    ```python title="examples/11-jobs/catalog_no_copy.py"
    --8<-- "examples/11-jobs/catalog_no_copy.py"
    ```

- ✅ **MD5 hash captured** for integrity verification
- ✅ Files remain in working directory only
- ✅ Prevents copying large or frequently unchanged data

**When to use `store_copy=False`:**

- **Large files** (datasets, models) where copying is expensive
- **Unchanging reference data** that doesn't need versioning
- **Network storage** where files are already backed up
- **Performance optimization** for frequently accessed files

## File Pattern Support

### Exact File Names
```python
Catalog(put=["results.csv", "model.pkl", "config.json"])
```

### Directory Support
```python
Catalog(put=["output_folder/", "logs/debug.log", "data/processed.csv"])
```

### Glob Pattern Support
```python
# Glob patterns are supported
Catalog(put=["plots/*.png", "reports/*.pdf", "logs/*.log"])

# Multiple patterns
Catalog(put=["output/**/*.csv", "results/*.json", "charts/*.png"])

# Complex patterns
Catalog(put=["data/**/processed_*.parquet", "models/best_model_*.pkl"])
```

## Common Use Cases

### Data Analysis Job
```python
def analyze_sales_data():
    # Analysis creates multiple outputs
    df.to_csv("sales_summary.csv")

    # Create multiple plots
    for region in ["north", "south", "east", "west"]:
        plot_regional_data(region)
        plt.savefig(f"plots/sales_trend_{region}.png")

    with open("insights.txt", "w") as f:
        f.write("Key findings...")

    return {"total_sales": 50000}

# Store all outputs using glob patterns
catalog = Catalog(put=["sales_summary.csv", "plots/*.png", "insights.txt"])
job = PythonJob(function=analyze_sales_data, catalog=catalog)
```

### Model Training Job
```python
def train_model():
    # Training creates artifacts
    model.save("trained_model.pkl")
    history.to_csv("training_history.csv")

    with open("model_metrics.json", "w") as f:
        json.dump({"accuracy": 0.95}, f)

    return model

# Store model artifacts
catalog = Catalog(put=["trained_model.pkl", "training_history.csv", "model_metrics.json"])
job = PythonJob(function=train_model, catalog=catalog)
```

### Report Generation Job
```python
def generate_monthly_report():
    # Report generation creates files
    create_pdf_report("monthly_report.pdf")

    # Generate multiple chart types
    save_charts_to("charts/")  # Creates charts/sales.png, charts/growth.png, etc.

    # Export data in multiple formats
    export_data_to("data/summary.csv")
    export_data_to("data/details.json")

    return "Report completed"

# Store report outputs using glob patterns
catalog = Catalog(
    put=["monthly_report.pdf", "charts/*.png", "data/*.csv", "data/*.json"],
    store_copy=True  # Reports should be archived
)
job = PythonJob(function=generate_monthly_report, catalog=catalog)
```

### Large Data Processing Job
```python
def process_large_dataset():
    # Processing creates large intermediate files
    processed_data.to_parquet("processed_data.parquet")  # Large file
    summary_stats.to_csv("summary.csv")                   # Small file

    return {"rows_processed": 1000000}

# Mixed storage strategy
catalog = Catalog(put=["processed_data.parquet", "summary.csv"], store_copy=False)
# Hash-only for the large file, but still tracks both
job = PythonJob(function=process_large_dataset, catalog=catalog)
```

## Catalog Structure

Jobs organize files by run ID:

```
.catalog/
├── run-id-001/
│   ├── function_name.execution.log
│   ├── output_file1.csv
│   └── data_folder/
│       └── nested_file.txt
├── run-id-002/
│   ├── function_name.execution.log
│   └── different_output.json
└── run-id-003/
    ├── function_name.execution.log
    └── large_file.parquet  # Only if store_copy=True
```

## Best Practices

### ✅ **Choose Appropriate Storage Mode**
```python
# Small, important files - copy them
Catalog(put=["config.json", "results.csv"], store_copy=True)

# Large, reference files - hash only
Catalog(put=["dataset.parquet", "model.pkl"], store_copy=False)
```

### ✅ **Organize Output Files**
```python
def my_analysis():
    # Create organized output structure
    os.makedirs("outputs", exist_ok=True)
    results.to_csv("outputs/results.csv")
    plots.savefig("outputs/visualization.png")

catalog = Catalog(put=["outputs/"])  # Store entire directory
```

### ✅ **Document File Purposes**
```python
# Clear naming for catalog files
catalog = Catalog(put=[
    "final_results.csv",      # Main output
    "diagnostic_plots.png",   # Quality checks
    "processing_log.txt",     # Execution details
])
```

## What's Next?

You can now store Job outputs automatically! Next topics:

- **[Job Types](job-types.md)** - Shell and Notebook Jobs
- **[Configuration](configuration.md)** - Advanced Job options and settings
- **[Examples](examples.md)** - Real-world Job patterns

Ready to explore different Job types? Continue to **[Job Types](job-types.md)**!
