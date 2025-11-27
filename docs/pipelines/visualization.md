# Pipeline Visualization 📊

Visualize your pipeline execution history with interactive timelines that show execution flow, timing, and hierarchical structure.

## Why Visualize Pipelines?

Pipeline visualization helps you:

- **Debug execution flows** - See exactly how your pipeline executed
- **Identify bottlenecks** - Find slow tasks and optimization opportunities
- **Understand parallel execution** - Visualize concurrent branches and timing
- **Monitor production runs** - Track pipeline performance over time
- **Document workflows** - Share visual pipeline reports with stakeholders

## Quick Start

```bash
# Run any pipeline to generate execution logs
uv run examples/02-sequential/traversal.py

# Visualize the execution (console + HTML + browser)
uv run runnable timeline ancient-pike-2335
```

## Timeline Command

### Basic Usage
```bash
runnable timeline [RUN_ID_OR_PATH]
```

### Input Options

**Using Run ID** (looks in `.run_log_store/`):
```bash
uv run runnable timeline forgiving-joliot-0645
```

**Using JSON file path**:
```bash
uv run runnable timeline .run_log_store/pipeline-run.json
uv run runnable timeline /path/to/my-run.json
```

### Output Control

| Option | Description | Default |
|--------|-------------|---------|
| `--output`, `-o` | Custom HTML file path | `{run_id}_timeline.html` |
| `--console` / `--no-console` | Show console output | `true` |
| `--open` / `--no-open` | Auto-open in browser | `true` |

## Console Timeline Output

### Sequential Pipeline
```
🔄 Pipeline Timeline - ancient-pike-2335
Status: SUCCESS
================================================================================
  📝 ✅ hello stub (0.0ms)
  ⚙️ ✅ hello python (2.2ms)
     📝 PYTHON: examples.common.functions.hello
  ⚙️ ✅ hello shell (16.5ms)
     📝 SHELL: echo 'Hello World!'
  ⚙️ ✅ hello notebook (4325.8ms)
     📝 NOTEBOOK: examples/common/simple_notebook.ipynb
  ✅ ✅ success (0.0ms)
================================================================================
```

### Parallel Pipeline
```bash
# Run parallel example first
uv run examples/06-parallel/parallel.py
uv run runnable timeline fried-pasteur-2336 --no-open
```

```
🔄 Pipeline Timeline - fried-pasteur-2336
Status: SUCCESS
================================================================================
🔀 parallel_step (parallel)
  ├─ Branch: branch1
      📝 ✅ hello stub (0.0ms)
      ⚙️ ✅ hello python (2.0ms)
      ⚙️ ✅ hello shell (19.6ms)
      ⚙️ ✅ hello notebook (1269.7ms)
      ✅ ✅ success (0.0ms)
  ├─ Branch: branch2
      📝 ✅ hello stub (0.0ms)
      ⚙️ ✅ hello python (2.4ms)
      ⚙️ ✅ hello shell (16.5ms)
      ⚙️ ✅ hello notebook (23.3ms)
      ✅ ✅ success (0.0ms)
  📝 ✅ continue to (0.0ms)
  ✅ ✅ success (0.0ms)
================================================================================
```

### Visual Elements

| Symbol | Meaning |
|--------|---------|
| 📝 | Stub task |
| ⚙️ | Executable task (Python, Shell, Notebook) |
| ✅ | Success node |
| ❌ | Failure node |
| 🔀 | Parallel execution block |
| ├─ | Branch indicator |

## Interactive HTML Timeline

The HTML output provides rich interactive features:

- **Hover tooltips** - Detailed task information
- **Expandable sections** - Collapse/expand parallel branches
- **Rich metadata** - Commands, parameters, execution details
- **Visual timeline** - Graphical execution flow
- **Responsive design** - Works on all devices

### Example Commands
```bash
# Default: Console + HTML + Browser
uv run runnable timeline my-pipeline-run

# Custom HTML file
uv run runnable timeline complex-pipeline --output report.html

# Console only (no browser)
uv run runnable timeline debug-run --no-open

# HTML only (no console)
uv run runnable timeline prod-run --no-console
```

## Practical Examples

### 🔍 **Development Debugging**
```bash
# Quick console feedback during development
uv run examples/02-sequential/traversal.py
uv run runnable timeline $(ls .run_log_store/ | tail -1 | cut -d. -f1) --no-open
```

### 📊 **Performance Analysis**
```bash
# Compare sequential vs parallel execution
uv run examples/02-sequential/traversal.py
uv run runnable timeline sequential-run --output sequential.html --no-open

uv run examples/06-parallel/parallel.py
uv run runnable timeline parallel-run --output parallel.html --no-open
```

### 🐛 **Failure Investigation**
```bash
# Run a pipeline that might fail
uv run examples/02-sequential/default_fail.py
uv run runnable timeline failed-run --output failure-analysis.html
```

### 📋 **Production Monitoring**
```bash
# Generate timeline reports for production runs
uv run runnable timeline $PROD_RUN_ID --output "prod-$(date +%Y%m%d).html" --no-open
```

## Supported Pipeline Types

Timeline visualization works with all pipeline patterns:

- ✅ **Sequential workflows** - Linear task execution
- ✅ **Parallel execution** - Multi-branch concurrent processing
- ✅ **Map operations** - Iterative data processing
- ✅ **Conditional workflows** - Branching logic based on parameters
- ✅ **Nested structures** - Complex hierarchical pipelines
- ✅ **Mixed task types** - Python, Shell, Notebook, Stub tasks
- ✅ **Error scenarios** - Failed executions with clear indicators

## Integration with Pipeline Development

### During Pipeline Design
```bash
# Test your pipeline structure
uv run your_pipeline.py
uv run runnable timeline latest-run --no-open  # Quick console check
```

### During Optimization
```bash
# Before optimization
uv run slow_pipeline.py
uv run runnable timeline baseline --output baseline.html --no-open

# After optimization
uv run optimized_pipeline.py
uv run runnable timeline optimized --output optimized.html --no-open
```

### For Documentation
```bash
# Generate visual documentation
uv run example_pipeline.py
uv run runnable timeline demo-run --output pipeline-demo.html --no-open
```

## Tips and Troubleshooting

### 🎯 **Best Practices**
- Use `--no-open` during development to avoid browser spam
- Use custom output names for comparison: `--output baseline.html`
- Check console output first before opening HTML for quick debugging

### 🔧 **Common Issues**
- **Empty timeline**: Pipeline might not have completed - check run logs
- **Run ID not found**: Check `.run_log_store/` directory or use full JSON path
- **No timing data**: Ensure pipeline actually executed (not just validated)

## What's Next?

- **[Parallel Execution](parallel-execution.md)** - Create parallel workflows to visualize
- **[Map Patterns](map-patterns.md)** - Build iterative pipelines with rich timelines
- **[Conditional Workflows](conditional-workflows.md)** - Visualize branching logic
- **[Failure Handling](failure-handling.md)** - Debug error scenarios with timelines

**Ready to visualize your pipelines?** Run any example and explore its timeline! 🚀
