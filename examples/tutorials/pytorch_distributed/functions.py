"""
PyTorch Distributed Training Tutorial - Core Functions

This module demonstrates single-node distributed training using PyTorch's
DistributedDataParallel (DDP) on CPU cores.

Features demonstrated:
- Multi-process distributed training on CPU
- PyTorch DistributedDataParallel (DDP)
- Gradient synchronization across processes
- Model checkpoint saving and loading
- Training metrics aggregation
"""

import json
import logging
import os
import sys
import tempfile
import time
from io import StringIO

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

def setup_process_logging(rank: int, checkpoint_dir: str):
    """
    Setup logging and output capture for a specific process rank.

    Args:
        rank: Process rank
        checkpoint_dir: Directory to save log files

    Returns:
        Tuple of (logger, stdout_capture, stderr_capture)
    """
    # Create process-specific log file
    log_file = os.path.join(checkpoint_dir, f"process_{rank}_output.log")

    # Setup logger for this process
    logger = logging.getLogger(f"rank_{rank}")
    logger.setLevel(logging.INFO)

    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # File handler for process-specific logs
    file_handler = logging.FileHandler(log_file, mode='w')
    file_formatter = logging.Formatter(
        f'[RANK {rank}] %(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler (still show in console)
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(f'[RANK {rank}] %(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Capture stdout and stderr
    stdout_capture = StringIO()
    stderr_capture = StringIO()

    return logger, stdout_capture, stderr_capture, log_file


class ProcessOutputCapture:
    """Context manager to capture all output from a process."""

    def __init__(self, rank: int, checkpoint_dir: str):
        self.rank = rank
        self.checkpoint_dir = checkpoint_dir
        self.stdout_file = os.path.join(checkpoint_dir, f"process_{rank}_stdout.txt")
        self.stderr_file = os.path.join(checkpoint_dir, f"process_{rank}_stderr.txt")
        self.combined_file = os.path.join(checkpoint_dir, f"process_{rank}_combined.txt")

    def __enter__(self):
        # Open files for capturing output
        self.stdout_f = open(self.stdout_file, 'w')
        self.stderr_f = open(self.stderr_file, 'w')
        self.combined_f = open(self.combined_file, 'w')

        # Save original stdout/stderr
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr

        # Create tee-like objects that write to both original and file
        self.tee_stdout = TeeOutput(self.original_stdout, self.stdout_f, self.combined_f, f"[RANK {self.rank} STDOUT]")
        self.tee_stderr = TeeOutput(self.original_stderr, self.stderr_f, self.combined_f, f"[RANK {self.rank} STDERR]")

        # Replace stdout/stderr
        sys.stdout = self.tee_stdout
        sys.stderr = self.tee_stderr

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original stdout/stderr
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr

        # Close files
        self.stdout_f.close()
        self.stderr_f.close()
        self.combined_f.close()


class TeeOutput:
    """Write to multiple outputs simultaneously."""

    def __init__(self, original, file_handle, combined_handle, prefix=""):
        self.original = original
        self.file_handle = file_handle
        self.combined_handle = combined_handle
        self.prefix = prefix

    def write(self, text):
        # Write to original (console)
        self.original.write(text)

        # Write to specific file
        self.file_handle.write(text)
        self.file_handle.flush()

        # Write to combined file with prefix
        if text.strip():  # Only add prefix to non-empty lines
            prefixed_text = f"{self.prefix} {text}" if not text.startswith('\n') else text
            self.combined_handle.write(prefixed_text)
        else:
            self.combined_handle.write(text)
        self.combined_handle.flush()

    def flush(self):
        self.original.flush()
        self.file_handle.flush()
        self.combined_handle.flush()


# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleNet(nn.Module):
    """
    A simple neural network for demonstration.
    This will work well with MNIST or similar datasets.
    """

    def __init__(self, input_size: int = 784, hidden_size: int = 128, num_classes: int = 10):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class SyntheticDataset(Dataset):
    """
    Synthetic dataset for demonstration purposes.
    Creates random data that resembles MNIST structure.
    """

    def __init__(self, num_samples: int = 10000, input_size: int = 784, num_classes: int = 10):
        self.num_samples = num_samples
        self.input_size = input_size
        self.num_classes = num_classes

        # Generate synthetic data
        torch.manual_seed(42)  # For reproducibility
        self.data = torch.randn(num_samples, input_size)
        # Create labels with some pattern (not completely random)
        self.labels = torch.randint(0, num_classes, (num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def setup_distributed(rank: int, world_size: int, backend: str = "gloo"):
    """
    Initialize the distributed environment.

    Args:
        rank: Process rank
        world_size: Total number of processes
        backend: Communication backend ("gloo" for CPU, "nccl" for GPU)
    """
    # Set environment variables for distributed training
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # Initialize the process group
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    logger.info(f"Distributed setup complete for rank {rank}/{world_size}")


def cleanup_distributed():
    """Clean up the distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def create_data_loader(dataset: Dataset, world_size: int, rank: int, batch_size: int = 32):
    """
    Create a distributed data loader.

    Args:
        dataset: PyTorch dataset
        world_size: Total number of processes
        rank: Current process rank
        batch_size: Batch size per process

    Returns:
        DataLoader with DistributedSampler
    """
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=2,
        pin_memory=False  # CPU training
    )

    return dataloader


def train_process(
    rank: int,
    world_size: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    model_params: dict,
    data_params: dict,
    checkpoint_dir: str
):
    """
    Training function for each distributed process.

    Args:
        rank: Process rank
        world_size: Total number of processes
        epochs: Number of training epochs
        batch_size: Batch size per process
        learning_rate: Learning rate
        model_params: Model hyperparameters
        data_params: Dataset parameters
        checkpoint_dir: Directory to save checkpoints
    """
    # Setup output capture for this process
    with ProcessOutputCapture(rank, checkpoint_dir):
        try:
            # Setup process-specific logging
            process_logger, _, _, log_file = setup_process_logging(rank, checkpoint_dir)

            process_logger.info(f"Process {rank} starting with PID: {os.getpid()}")
            print(f"[PROCESS {rank}] Hello from distributed training process!")
            print(f"[PROCESS {rank}] Working directory: {os.getcwd()}")
            print(f"[PROCESS {rank}] Training parameters: epochs={epochs}, batch_size={batch_size}, lr={learning_rate}")

            # Setup distributed environment
            setup_distributed(rank, world_size, backend="gloo")
            process_logger.info(f"Distributed setup complete for process {rank}")

            # Create dataset and data loader
            dataset = SyntheticDataset(**data_params)
            train_loader = create_data_loader(dataset, world_size, rank, batch_size)

            print(f"[PROCESS {rank}] Dataset created with {len(dataset)} samples")
            print(f"[PROCESS {rank}] DataLoader created with {len(train_loader)} batches per epoch")

            # Create model and wrap with DDP
            model = SimpleNet(**model_params)
            ddp_model = DDP(model)

            print(f"[PROCESS {rank}] Model created and wrapped with DistributedDataParallel")
            print(f"[PROCESS {rank}] Model parameters: {sum(p.numel() for p in model.parameters())} total parameters")

            # Optimizer
            optimizer = optim.Adam(ddp_model.parameters(), lr=learning_rate)
            criterion = nn.CrossEntropyLoss()

            print(f"[PROCESS {rank}] Optimizer and loss function initialized")

            # Training metrics
            training_metrics = {
                "rank": rank,
                "epoch_losses": [],
                "epoch_accuracies": [],
                "epoch_times": [],
                "total_samples": 0
            }

            process_logger.info(f"Starting training for {epochs} epochs")
            print(f"[PROCESS {rank}] ===== TRAINING START =====")
            print(f"[PROCESS {rank}] Training configuration:")
            print(f"[PROCESS {rank}]   - Epochs: {epochs}")
            print(f"[PROCESS {rank}]   - Batch size: {batch_size}")
            print(f"[PROCESS {rank}]   - Learning rate: {learning_rate}")
            print(f"[PROCESS {rank}]   - World size: {world_size}")
            print(f"[PROCESS {rank}]   - Rank: {rank}")

            for epoch in range(epochs):
                print(f"[PROCESS {rank}] ----- EPOCH {epoch} START -----")
                epoch_start = time.time()
                ddp_model.train()

                # Set epoch for DistributedSampler
                train_loader.sampler.set_epoch(epoch)
                print(f"[PROCESS {rank}] Sampler epoch set to {epoch}")

                epoch_loss = 0.0
                correct = 0
                total = 0

                for batch_idx, (data, target) in enumerate(train_loader):
                    optimizer.zero_grad()
                    output = ddp_model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    total += target.size(0)

                    # More frequent logging per process
                    if batch_idx % 25 == 0:  # Every 25 batches
                        print(f"[PROCESS {rank}] Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}, Samples: {total}")

                    if batch_idx % 50 == 0 and rank == 0:  # Only rank 0 logs to main logger
                        logger.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

                epoch_time = time.time() - epoch_start
                epoch_loss /= len(train_loader)
                epoch_accuracy = correct / total

                training_metrics["epoch_losses"].append(epoch_loss)
                training_metrics["epoch_accuracies"].append(epoch_accuracy)
                training_metrics["epoch_times"].append(epoch_time)
                training_metrics["total_samples"] += total

                # Every process logs its own results
                print(f"[PROCESS {rank}] Epoch {epoch} COMPLETE:")
                print(f"[PROCESS {rank}]   - Loss: {epoch_loss:.4f}")
                print(f"[PROCESS {rank}]   - Accuracy: {epoch_accuracy:.4f}")
                print(f"[PROCESS {rank}]   - Time: {epoch_time:.2f}s")
                print(f"[PROCESS {rank}]   - Samples processed: {total}")
                process_logger.info(f"Epoch {epoch}: Loss={epoch_loss:.4f}, Accuracy={epoch_accuracy:.4f}, Time={epoch_time:.2f}s")

                if rank == 0:  # Only rank 0 logs to main logger
                    logger.info(
                        f"Epoch {epoch}: Loss={epoch_loss:.4f}, "
                        f"Accuracy={epoch_accuracy:.4f}, Time={epoch_time:.2f}s"
                    )

                    # Save epoch checkpoint (only rank 0)
                    print(f"[PROCESS {rank}] Saving checkpoint for epoch {epoch}")
                    epoch_checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
                    torch.save({
                    'epoch': epoch,
                    'model_state_dict': ddp_model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch_loss': epoch_loss,
                    'epoch_accuracy': epoch_accuracy,
                    'training_metrics_so_far': {
                        'epoch_losses': training_metrics["epoch_losses"].copy(),
                        'epoch_accuracies': training_metrics["epoch_accuracies"].copy(),
                        'epoch_times': training_metrics["epoch_times"].copy()
                    },
                    'model_params': model_params,
                    'total_epochs': epochs,
                    'rank': rank
                }, epoch_checkpoint_path)

                # Also save as "latest" checkpoint for easy resuming
                latest_checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_latest.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': ddp_model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch_loss': epoch_loss,
                    'epoch_accuracy': epoch_accuracy,
                    'training_metrics_so_far': {
                        'epoch_losses': training_metrics["epoch_losses"].copy(),
                        'epoch_accuracies': training_metrics["epoch_accuracies"].copy(),
                        'epoch_times': training_metrics["epoch_times"].copy()
                    },
                    'model_params': model_params,
                    'total_epochs': epochs,
                    'rank': rank
                }, latest_checkpoint_path)

            # Save final model checkpoint (only rank 0)
            if rank == 0:
                final_checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_final.pt")
                torch.save({
                    'epoch': epochs - 1,  # Final epoch
                    'model_state_dict': ddp_model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'training_metrics': training_metrics,
                    'model_params': model_params,
                    'total_epochs': epochs,
                    'training_complete': True,
                    'rank': rank
                }, final_checkpoint_path)
                logger.info(f"Final model checkpoint saved to {final_checkpoint_path}")

                # Save process-specific metrics
                metrics_path = os.path.join(checkpoint_dir, f"metrics_rank_{rank}.json")
                with open(metrics_path, 'w') as f:
                    json.dump(training_metrics, f, indent=2)

                print(f"[PROCESS {rank}] ===== TRAINING COMPLETE =====")
                print(f"[PROCESS {rank}] Final training metrics:")
                print(f"[PROCESS {rank}]   - Total epochs: {epochs}")
                print(f"[PROCESS {rank}]   - Final loss: {training_metrics['epoch_losses'][-1]:.4f}")
                print(f"[PROCESS {rank}]   - Final accuracy: {training_metrics['epoch_accuracies'][-1]:.4f}")
                print(f"[PROCESS {rank}]   - Total samples processed: {training_metrics['total_samples']}")
                print(f"[PROCESS {rank}]   - Average epoch time: {sum(training_metrics['epoch_times'])/len(training_metrics['epoch_times']):.2f}s")
                print(f"[PROCESS {rank}] Metrics saved to: {metrics_path}")

                process_logger.info("Training completed successfully")
                logger.info(f"Rank {rank}: Training completed")

        except Exception as e:
            print(f"[PROCESS {rank}] ERROR: Training failed with error: {e}")
            process_logger.error(f"Training failed with error: {e}")
            logger.error(f"Rank {rank}: Training failed with error: {e}")
            raise
        finally:
            print(f"[PROCESS {rank}] Cleaning up distributed environment")
            cleanup_distributed()
            print(f"[PROCESS {rank}] Process {rank} finished")


def prepare_training_data(
    num_samples: int = 10000,
    input_size: int = 784,
    num_classes: int = 10,
    train_split: float = 0.8
):
    """
    Prepare training and validation datasets.

    Args:
        num_samples: Total number of samples
        input_size: Input feature size
        num_classes: Number of classes
        train_split: Fraction of data for training

    Returns:
        Dictionary with dataset information
    """
    logger.info(f"Preparing synthetic dataset with {num_samples} samples...")

    # Create synthetic dataset
    full_dataset = SyntheticDataset(num_samples, input_size, num_classes)

    # Split into train and validation
    train_size = int(train_split * num_samples)
    val_size = num_samples - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    dataset_info = {
        "total_samples": num_samples,
        "train_samples": train_size,
        "val_samples": val_size,
        "input_size": input_size,
        "num_classes": num_classes,
        "train_split": train_split
    }

    logger.info(f"Dataset prepared: {train_size} train, {val_size} validation samples")

    return dataset_info


def run_distributed_training(
    world_size: int = 4,
    epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    hidden_size: int = 128,
    num_samples: int = 10000,
    input_size: int = 784,
    num_classes: int = 10
):
    """
    Launch distributed training across multiple processes.

    Args:
        world_size: Number of processes to spawn
        epochs: Number of training epochs
        batch_size: Batch size per process
        learning_rate: Learning rate
        hidden_size: Hidden layer size
        num_samples: Number of synthetic samples
        input_size: Input feature size
        num_classes: Number of classes

    Returns:
        Dictionary with training results
    """
    # Ensure we don't exceed available CPU cores
    available_cores = torch.get_num_threads()
    if world_size > available_cores:
        logger.warning(f"Requested {world_size} processes, but only {available_cores} CPU cores available")
        world_size = min(world_size, available_cores)

    logger.info(f"Starting distributed training with {world_size} processes")

    # Create temporary directory for checkpoints
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_dir = temp_dir

        # Model and data parameters
        model_params = {
            "input_size": input_size,
            "hidden_size": hidden_size,
            "num_classes": num_classes
        }

        data_params = {
            "num_samples": num_samples,
            "input_size": input_size,
            "num_classes": num_classes
        }

        # Launch distributed training processes
        start_time = time.time()

        try:
            mp.spawn(
                train_process,
                args=(
                    world_size,
                    epochs,
                    batch_size,
                    learning_rate,
                    model_params,
                    data_params,
                    checkpoint_dir
                ),
                nprocs=world_size,
                join=True
            )
        except Exception as e:
            logger.error(f"Distributed training failed: {e}")
            raise

        total_time = time.time() - start_time

        # Collect results from all processes
        all_metrics = []
        for rank in range(world_size):
            metrics_path = os.path.join(checkpoint_dir, f"metrics_rank_{rank}.json")
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                    all_metrics.append(metrics)

        # Load and save all checkpoints to permanent storage
        final_model_info = None
        saved_checkpoints = []

        # Save final checkpoint
        final_checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_final.pt")
        if os.path.exists(final_checkpoint_path):
            checkpoint = torch.load(final_checkpoint_path, map_location='cpu')
            final_model_info = {
                "model_params": checkpoint["model_params"],
                "final_epoch": checkpoint["epoch"],
                "training_complete": checkpoint.get("training_complete", True),
                "final_loss": checkpoint.get("training_metrics", {}).get("epoch_losses", [0])[-1],
                "final_accuracy": checkpoint.get("training_metrics", {}).get("epoch_accuracies", [0])[-1]
            }

            # Save final model to permanent location
            permanent_final_path = "distributed_model_checkpoint.pt"
            torch.save(checkpoint, permanent_final_path)
            logger.info(f"Final model saved to {permanent_final_path}")
            saved_checkpoints.append(permanent_final_path)

        # Save all epoch checkpoints to permanent storage
        for epoch in range(epochs):
            epoch_checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
            if os.path.exists(epoch_checkpoint_path):
                permanent_epoch_path = f"checkpoint_epoch_{epoch}.pt"
                checkpoint = torch.load(epoch_checkpoint_path, map_location='cpu')
                torch.save(checkpoint, permanent_epoch_path)
                saved_checkpoints.append(permanent_epoch_path)
                logger.info(f"Epoch {epoch} checkpoint saved to {permanent_epoch_path}")

        # Save latest checkpoint
        latest_checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_latest.pt")
        if os.path.exists(latest_checkpoint_path):
            permanent_latest_path = "checkpoint_latest.pt"
            checkpoint = torch.load(latest_checkpoint_path, map_location='cpu')
            torch.save(checkpoint, permanent_latest_path)
            saved_checkpoints.append(permanent_latest_path)
            logger.info(f"Latest checkpoint saved to {permanent_latest_path}")

        # Save all process output files to permanent storage
        process_output_files = []
        for rank in range(world_size):
            # Copy process-specific output files
            for file_type in ['stdout', 'stderr', 'combined', 'output']:
                if file_type == 'output':
                    temp_file = os.path.join(checkpoint_dir, f"process_{rank}_output.log")
                    permanent_file = f"process_{rank}_output.log"
                else:
                    temp_file = os.path.join(checkpoint_dir, f"process_{rank}_{file_type}.txt")
                    permanent_file = f"process_{rank}_{file_type}.txt"

                if os.path.exists(temp_file):
                    import shutil
                    shutil.copy2(temp_file, permanent_file)
                    process_output_files.append(permanent_file)
                    logger.info(f"Process {rank} {file_type} saved to {permanent_file}")

        if final_model_info:
            final_model_info["saved_checkpoints"] = saved_checkpoints
            final_model_info["process_output_files"] = process_output_files

        # Aggregate metrics across processes
        if all_metrics:
            # Calculate average metrics across processes
            num_processes = len(all_metrics)
            avg_final_loss = sum(m["epoch_losses"][-1] for m in all_metrics) / num_processes
            avg_final_accuracy = sum(m["epoch_accuracies"][-1] for m in all_metrics) / num_processes
            total_samples_processed = sum(m["total_samples"] for m in all_metrics)

            training_results = {
                "distributed_training": {
                    "world_size": world_size,
                    "total_time": total_time,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate
                },
                "performance": {
                    "avg_final_loss": avg_final_loss,
                    "avg_final_accuracy": avg_final_accuracy,
                    "total_samples_processed": total_samples_processed,
                    "samples_per_second": total_samples_processed / total_time,
                    "time_per_epoch": total_time / epochs
                },
                "process_metrics": all_metrics,
                "model_info": final_model_info,
                "success": True
            }
        else:
            training_results = {
                "distributed_training": {
                    "world_size": world_size,
                    "total_time": total_time,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate
                },
                "success": False,
                "error": "No metrics collected from training processes"
            }

    logger.info(f"Distributed training completed in {total_time:.2f} seconds")

    # Save detailed results
    with open("distributed_training_results.json", "w") as f:
        json.dump(training_results, f, indent=2)

    # Extract metrics for Runnable
    if training_results["success"]:
        avg_accuracy = training_results["performance"]["avg_final_accuracy"]
        total_time = training_results["distributed_training"]["total_time"]
    else:
        avg_accuracy = 0.0
        total_time = total_time

    return training_results, avg_accuracy, total_time


def evaluate_distributed_model(model_checkpoint_path: str = "distributed_model_checkpoint.pt"):
    """
    Evaluate the trained distributed model.

    Args:
        model_checkpoint_path: Path to the saved model checkpoint

    Returns:
        Evaluation results
    """
    logger.info("Evaluating distributed model...")

    if not os.path.exists(model_checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(model_checkpoint_path, map_location='cpu')
    model_params = checkpoint["model_params"]

    # Create and load model
    model = SimpleNet(**model_params)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Create test dataset
    test_dataset = SyntheticDataset(num_samples=1000, **{k: v for k, v in model_params.items() if k != "hidden_size"})
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Evaluate
    correct = 0
    total = 0
    test_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    test_loss /= len(test_loader)
    test_accuracy = correct / total

    evaluation_results = {
        "test_accuracy": test_accuracy,
        "test_loss": test_loss,
        "test_samples": total,
        "correct_predictions": correct,
        "model_params": model_params,
        "checkpoint_info": {
            "epochs_trained": checkpoint.get("total_epochs", checkpoint.get("epoch", 0) + 1),
            "final_epoch": checkpoint.get("epoch", 0),
            "training_complete": checkpoint.get("training_complete", True)
        }
    }

    logger.info(f"Model evaluation complete: Accuracy={test_accuracy:.4f}, Loss={test_loss:.4f}")

    return evaluation_results, test_accuracy


def load_checkpoint_for_resuming(
    checkpoint_path: str = "checkpoint_latest.pt",
    model_params: dict = None
):
    """
    Load a checkpoint for resuming training.

    Args:
        checkpoint_path: Path to the checkpoint file
        model_params: Model parameters (if None, will use from checkpoint)

    Returns:
        Dictionary with model, optimizer, epoch info for resuming
    """
    logger.info(f"Loading checkpoint for resuming: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Use model params from checkpoint if not provided
    if model_params is None:
        model_params = checkpoint["model_params"]

    # Create model and load state
    model = SimpleNet(**model_params)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Create optimizer and load state
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    resume_info = {
        "model": model,
        "optimizer": optimizer,
        "start_epoch": checkpoint["epoch"] + 1,  # Start from next epoch
        "model_params": model_params,
        "previous_metrics": checkpoint.get("training_metrics_so_far", {}),
        "checkpoint_info": {
            "loaded_from": checkpoint_path,
            "epoch": checkpoint["epoch"],
            "loss": checkpoint.get("epoch_loss", 0.0),
            "accuracy": checkpoint.get("epoch_accuracy", 0.0)
        }
    }

    logger.info(f"Checkpoint loaded: resuming from epoch {resume_info['start_epoch']}")
    logger.info(f"Previous training - Epoch {checkpoint['epoch']}: "
               f"Loss={checkpoint.get('epoch_loss', 0.0):.4f}, "
               f"Accuracy={checkpoint.get('epoch_accuracy', 0.0):.4f}")

    return resume_info


def list_available_checkpoints():
    """
    List all available checkpoint files.

    Returns:
        Dictionary with information about available checkpoints
    """
    import glob

    checkpoint_files = {
        "epoch_checkpoints": sorted(glob.glob("checkpoint_epoch_*.pt")),
        "latest_checkpoint": "checkpoint_latest.pt" if os.path.exists("checkpoint_latest.pt") else None,
        "final_checkpoint": "distributed_model_checkpoint.pt" if os.path.exists("distributed_model_checkpoint.pt") else None
    }

    # Get info from each checkpoint
    checkpoint_info = {}
    for category, files in checkpoint_files.items():
        if category == "epoch_checkpoints" and files:
            checkpoint_info[category] = []
            for file in files:
                try:
                    checkpoint = torch.load(file, map_location='cpu')
                    info = {
                        "file": file,
                        "epoch": checkpoint.get("epoch", -1),
                        "loss": checkpoint.get("epoch_loss", 0.0),
                        "accuracy": checkpoint.get("epoch_accuracy", 0.0)
                    }
                    checkpoint_info[category].append(info)
                except Exception as e:
                    logger.warning(f"Could not load info from {file}: {e}")
        elif files:  # latest or final checkpoint
            try:
                checkpoint = torch.load(files, map_location='cpu')
                checkpoint_info[category] = {
                    "file": files,
                    "epoch": checkpoint.get("epoch", -1),
                    "loss": checkpoint.get("epoch_loss", 0.0),
                    "accuracy": checkpoint.get("epoch_accuracy", 0.0),
                    "training_complete": checkpoint.get("training_complete", False)
                }
            except Exception as e:
                logger.warning(f"Could not load info from {files}: {e}")

    return checkpoint_info
