# Standard PyTorch Examples with Runnable

This directory demonstrates how **runnable can execute standard PyTorch scripts with minimal modification**. The PyTorch code follows typical patterns that users are familiar with, and runnable provides orchestration capabilities on top.

## ⚡ Key Finding: Only Type Annotations Required

The scripts in this directory prove that runnable can work with standard PyTorch code with **only one change**: adding type annotations to function signatures. No other modifications are needed!

## Files Overview

### Standard PyTorch Scripts (Minimal Runnable Compatibility)

- **`train.py`** - Standard single-process PyTorch training script with argparse + type annotations
- **`train_distributed.py`** - Standard distributed PyTorch training using DDP with torchrun + type annotations

### Runnable Integration Examples

- **`train_runnable.py`** - Executes the standard training function via PythonJob
- **`train_distributed_runnable.py`** - Executes the distributed training function via PythonJob

### Configuration Files

- **`train_parameters.yaml`** - Configuration file with training parameters for runnable execution
- **`parameters.yaml`** - Alternative parameter configuration (legacy)

## Key Demonstration Points

### 1. **Standard PyTorch Code with Minimal Changes**

The PyTorch scripts (`train.py`, `train_distributed.py`) follow typical PyTorch patterns with **only type annotations added**:

```python
# Standard argparse pattern with type annotations
def train_model(args: argparse.Namespace) -> dict:
    """Main training function."""
    # Everything else remains exactly the same
    for epoch in range(args.epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            # Standard forward/backward pass

def main():
    parser = argparse.ArgumentParser(description='Standard PyTorch Training')
    parser.add_argument('--epochs', type=int, default=10)
    # ... standard argparse code unchanged
    args = parser.parse_args()
    results = train_model(args)  # Call typed function
```

### 2. **The Only Required Change: Type Annotations**

To make standard PyTorch scripts runnable-compatible, you only need to add:

```python
# Before (standard PyTorch)
def train_model(args):
    # training code...

# After (runnable-compatible)
def train_model(args: argparse.Namespace) -> dict:
    # same training code, no other changes!
```

This enables runnable to:
- Automatically handle parameter passing from YAML files
- Provide type safety and validation
- Enable direct function calls via PythonJob

### 3. **Runnable Integration Pattern**

The `*_runnable.py` files show the clean integration pattern:

```python
# train_runnable.py
from examples.standard_pytorch.train import train_model as train_main
from runnable import Catalog, PythonJob

def main():
    job = PythonJob(
        function=train_main,  # Direct function reference
        catalog=Catalog(put=["training_output/*"])  # Artifact management
    )
    job.execute()

# Set parameters file via environment variable
os.environ["RUNNABLE_PARAMETERS_FILE"] = "examples/standard_pytorch/train_parameters.yaml"
```

## Running the Examples

### 1. Direct Execution (Standard PyTorch Way)

```bash
# Single process training
uv run examples/standard_pytorch/train.py --epochs 5 --batch-size 32

# Distributed training (if torchrun available)
torchrun --nproc_per_node=2 examples/standard_pytorch/train_distributed.py --epochs 5
```

### 2. Through Runnable Integration

```bash
# Single training via runnable
uv run examples/standard_pytorch/train_runnable.py

# Distributed training via runnable
uv run examples/standard_pytorch/train_distributed_runnable.py
```

### 3. Parameter Configuration

The `train_parameters.yaml` file allows you to configure training without command-line arguments:

```yaml
# Training parameters
epochs: 10
batch_size: 64
learning_rate: 0.001

# Model architecture
input_size: 784
hidden_size: 128
num_classes: 10

# Dataset settings
num_samples: 10000
seed: 42

# System settings
output_dir: "training_output"
num_workers: 2
log_interval: 100
save_interval: 5
```

## What Runnable Adds (Without Changing PyTorch Code)

### 🔧 **Orchestration Layer**
- Clean job execution with PythonJob
- Parameter management through YAML files
- Environment variable configuration

### 📁 **Automatic Artifact Management**
- Captures model checkpoints, logs, and results via Catalog
- Organizes outputs systematically
- Tracks file dependencies between jobs

### 📊 **Enhanced Execution**
- Type-safe parameter passing
- Automatic error handling and logging
- Consistent execution environment

### 🔄 **Reproducibility**
- Parameter versioning through YAML configs
- Automatic logging of execution metadata
- Consistent environment management

## Standard vs. Runnable Comparison

| Aspect | Standard PyTorch | With Runnable |
|--------|------------------|---------------|
| **Code Changes** | N/A | ✅ **Only type annotations** |
| **Execution** | `python train.py --args` | `job.execute()` with YAML params |
| **Parameters** | Command line args | YAML config files |
| **Outputs** | Manual file management | Automatic catalog |
| **Integration** | Script-based | Function-based with jobs |
| **Reproducibility** | Manual setup | Automatic versioning |

## The Key Insight

**Runnable preserves how you write PyTorch code with minimal changes.** It provides an orchestration layer that:

1. **Executes your existing functions** with only type annotations added
2. **Adds production capabilities** (artifact management, reproducibility)
3. **Enables job-based execution** for better integration
4. **Maintains the familiar PyTorch development experience**

### Migration Path for Existing PyTorch Code

For teams with existing PyTorch scripts, the migration is incredibly simple:

```python
# Your existing function
def train_model(args):
    # ... all your existing training code

# Add type annotations (that's it!)
def train_model(args: argparse.Namespace) -> dict:
    # ... exact same training code, zero changes
```

Then create a simple runnable wrapper:

```python
# train_runnable.py
from your_module import train_model
from runnable import PythonJob

def main():
    job = PythonJob(function=train_model)
    job.execute()
```

This minimal change enables:
- ✅ **Direct function execution** via PythonJob
- ✅ **Automatic parameter handling** from YAML configs
- ✅ **Type safety** and validation
- ✅ **Artifact management** capabilities
- ✅ **Backward compatibility** - scripts still work with direct execution

### Benefits Without Rewriting

Teams can:
- **Keep 99% of existing PyTorch code unchanged**
- **Add only type annotations** for runnable compatibility
- **Create simple wrapper jobs** for orchestration benefits
- **Scale from development to production** without architectural changes
- **Maintain full compatibility** with standard PyTorch tooling and workflows

The key insight is that runnable doesn't replace your PyTorch code - it orchestrates it with minimal changes required.
