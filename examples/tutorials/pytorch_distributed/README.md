# PyTorch Distributed Training Tutorial

A comprehensive tutorial demonstrating single-node distributed training using PyTorch's DistributedDataParallel (DDP) on CPU cores, orchestrated with Runnable.

## What You'll Learn

This tutorial showcases advanced PyTorch and Runnable integration:
- **Multi-process distributed training** on CPU cores
- **PyTorch DistributedDataParallel (DDP)** for gradient synchronization
- **Process coordination** and communication
- **Model checkpointing** and persistence
- **Training metrics aggregation** across processes
- **Runnable orchestration** of distributed workloads

## Distributed Training Architecture

```
              Prepare Data
                    ↓
            Setup Distributed Environment
                    ↓
    ┌─────────────────────────────────────┐
    │        Multi-Process Training        │
    └─────────────────────────────────────┘
    ↓         ↓         ↓         ↓
[Process 0] [Process 1] [Process 2] [Process 3]
    \         |         |         /
     \        |         |        /
      Gradient Synchronization (DDP)
                    ↓
            Aggregate Results
                    ↓
             Evaluate Model
```

### Key Components

1. **DistributedDataParallel (DDP)** - Synchronizes gradients across processes
2. **DistributedSampler** - Ensures each process sees different data
3. **Process Group** - Coordinates communication between processes
4. **Checkpoint Management** - Saves model state from rank 0 process

## Running the Tutorial

### Prerequisites

Install the tutorial dependencies:

```bash
uv sync --group tutorial
```

### Execute the Pipeline

```bash
uv run --group tutorial examples/tutorials/pytorch_distributed/pipeline.py
```

### What Gets Generated

**Training Artifacts:**
- `distributed_model_checkpoint.pt` - Complete model checkpoint with optimizer state
- `distributed_training_results.json` - Comprehensive training metrics and performance data

**Performance Metrics:**
- **Training Speed**: ~2,840 samples/second across 4 processes
- **Scaling Efficiency**: Linear scaling with number of processes
- **Memory Usage**: Distributed across processes for large datasets

**Catalog Contents** (`.catalog/<run_id>/`):
- Model checkpoint and training results
- Process-specific metrics from each rank
- Dataset information and configuration
- Complete execution logs

## Configuration Options

Modify `parameters.yaml` to experiment with different distributed settings:

### Distributed Configuration
```yaml
world_size: 4           # Number of processes (≤ CPU cores)
epochs: 10              # Training epochs
batch_size: 64          # Batch size per process
learning_rate: 0.001    # Learning rate
```

### Model Architecture
```yaml
hidden_size: 256        # Neural network hidden layer size
input_size: 784         # Input features (28x28 for MNIST-like)
num_classes: 10         # Output classes
```

### Dataset Parameters
```yaml
num_samples: 20000      # Total synthetic samples
train_split: 0.8        # Training data fraction
```

## Expected Results

On a **4-core CPU system**:
- **Training Time**: ~70 seconds for 10 epochs
- **Throughput**: ~2,840 samples/second
- **Training Accuracy**: ~73% (synthetic data)
- **Test Accuracy**: Varies (synthetic evaluation)
- **Memory Distribution**: Efficient per-process usage

### Performance Characteristics:
- **Scaling**: Near-linear with process count
- **Communication**: Efficient gradient synchronization
- **Load Balancing**: Equal data distribution via DistributedSampler

## Key PyTorch DDP Features Demonstrated

### 1. Process Initialization
```python
def setup_distributed(rank: int, world_size: int, backend: str = "gloo"):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(backend, rank=rank, world_size=world_size)
```

### 2. Model Wrapping
```python
model = SimpleNet(**model_params)
ddp_model = DDP(model)  # Automatic gradient synchronization
```

### 3. Data Distribution
```python
sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
```

### 4. Process Spawning
```python
mp.spawn(train_process, args=(world_size, ...), nprocs=world_size, join=True)
```

## Runnable Integration Benefits

### Orchestration
- **Simplified Execution**: Single command launches complex distributed training
- **Parameter Management**: YAML configuration flows to all processes
- **Metrics Tracking**: Automatic collection of training and performance metrics

### Reproducibility
- **Checkpoint Storage**: Models and states saved in catalog
- **Execution History**: Complete audit trail of distributed runs
- **Configuration Versioning**: Parameters tracked with each execution

### Monitoring
- **Real-time Logs**: Process-specific logging and aggregation
- **Performance Metrics**: Training speed, accuracy, and resource usage
- **Error Handling**: Graceful failure handling across processes

## Advanced Configurations

### High-Performance Setup
```yaml
world_size: 8           # Use all CPU cores
batch_size: 32          # Smaller batch per process
hidden_size: 512        # Larger model
num_samples: 50000      # More training data
```

### Memory-Efficient Setup
```yaml
world_size: 2           # Fewer processes
batch_size: 128         # Larger batch per process
hidden_size: 128        # Smaller model
```

### Single-Process Comparison
```yaml
world_size: 1           # Disable distribution
batch_size: 256         # Single large batch
```

## Extending the Tutorial

### Real Datasets
Replace `SyntheticDataset` with real data loaders:
```python
from torchvision import datasets, transforms

dataset = datasets.MNIST(root='./data', train=True,
                        transform=transforms.ToTensor(), download=True)
```

### GPU Support
Modify for GPU distributed training:
```python
# Change backend and device management
setup_distributed(rank, world_size, backend="nccl")
device = torch.device(f"cuda:{rank}")
model = model.to(device)
```

### Advanced Models
Replace SimpleNet with more sophisticated architectures:
- Convolutional Neural Networks for vision tasks
- Transformer models for NLP
- Custom architectures for specific domains

## Performance Analysis

The tutorial demonstrates excellent distributed training characteristics:

### Scaling Analysis
- **Linear Scaling**: Performance scales proportionally with process count
- **Communication Overhead**: Minimal due to efficient DDP implementation
- **Load Balancing**: Equal work distribution via DistributedSampler

### Resource Utilization
- **CPU Usage**: Efficiently uses all available cores
- **Memory Distribution**: Each process handles subset of data
- **I/O Optimization**: Parallel data loading across processes

## Troubleshooting

### Common Issues
1. **Port Conflicts**: Change `MASTER_PORT` if 12355 is in use
2. **Process Hanging**: Ensure all processes call collective operations
3. **Memory Issues**: Reduce batch_size or model size
4. **Slow Training**: Check CPU core count vs world_size

### Performance Optimization
1. **Tune world_size** to match available CPU cores
2. **Adjust batch_size** per process for optimal memory usage
3. **Use proper data loading** with sufficient num_workers
4. **Monitor CPU utilization** to detect bottlenecks

This tutorial provides a solid foundation for building production-scale distributed training pipelines with PyTorch and Runnable!
