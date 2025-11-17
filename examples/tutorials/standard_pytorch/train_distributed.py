#!/usr/bin/env python3
"""
Standard PyTorch Distributed Training Script
This follows the typical PyTorch DDP pattern without any runnable-specific code.

Run with: torchrun --nproc_per_node=4 train_distributed.py --epochs 10 --batch-size 32
Or with: python -m torch.distributed.launch --nproc_per_node=4 train_distributed.py --epochs 10
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SimpleNet(nn.Module):
    """Simple neural network for demonstration."""

    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class SyntheticDataset(Dataset):
    """Synthetic dataset for demonstration."""

    def __init__(self, num_samples=10000, input_size=784, num_classes=10, seed=42):
        torch.manual_seed(seed)
        self.data = torch.randn(num_samples, input_size)
        self.labels = torch.randint(0, num_classes, (num_samples,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def setup_distributed():
    """Initialize distributed training."""
    # Get rank and world size from environment variables (set by torchrun)
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # Initialize process group
    if world_size > 1:
        dist.init_process_group(backend="gloo")  # Use gloo for CPU
        torch.cuda.set_device(local_rank) if torch.cuda.is_available() else None

    return rank, local_rank, world_size


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def train_distributed(args: argparse.Namespace) -> tuple:
    """Main distributed training function."""
    # Setup distributed training
    rank, local_rank, world_size = setup_distributed()

    if rank == 0:
        logger.info(f"Starting distributed training with {world_size} processes")
        logger.info(f"Training parameters: {vars(args)}")

    # Set device (CPU for this example)
    device = torch.device("cpu")

    # Create dataset
    dataset = SyntheticDataset(
        num_samples=args.num_samples,
        input_size=args.input_size,
        num_classes=args.num_classes,
        seed=args.seed,
    )

    # Create distributed sampler and dataloader
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
    )

    # Create model and wrap with DDP
    model = SimpleNet(
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        num_classes=args.num_classes,
    ).to(device)

    if world_size > 1:
        model = DDP(model)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Training metrics
    train_losses = []
    train_accuracies = []

    if rank == 0:
        logger.info(f"Starting training for {args.epochs} epochs")
        os.makedirs(args.output_dir, exist_ok=True)

    start_time = time.time()

    for epoch in range(args.epochs):
        epoch_start = time.time()
        model.train()

        # Set epoch for DistributedSampler
        sampler.set_epoch(epoch)

        epoch_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

            if batch_idx % args.log_interval == 0 and rank == 0:
                logger.info(
                    f"Rank {rank}, Epoch {epoch+1}/{args.epochs}, "
                    f"Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.6f}"
                )

        epoch_time = time.time() - epoch_start
        epoch_loss /= len(dataloader)
        epoch_accuracy = correct / total

        # Gather metrics from all processes
        if world_size > 1:
            # Synchronize metrics across processes
            loss_tensor = torch.tensor([epoch_loss], device=device)
            accuracy_tensor = torch.tensor([epoch_accuracy], device=device)
            total_tensor = torch.tensor([total], device=device)

            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(accuracy_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)

            # Average across processes
            avg_loss = loss_tensor.item() / world_size
            avg_accuracy = accuracy_tensor.item() / world_size
            total_samples = total_tensor.item()
        else:
            avg_loss = epoch_loss
            avg_accuracy = epoch_accuracy
            total_samples = total

        train_losses.append(avg_loss)
        train_accuracies.append(avg_accuracy)

        if rank == 0:
            logger.info(
                f"Epoch {epoch+1}/{args.epochs}: "
                f"Avg Loss: {avg_loss:.4f}, "
                f"Avg Accuracy: {avg_accuracy:.4f}, "
                f"Time: {epoch_time:.2f}s, "
                f"Total Samples: {total_samples}"
            )

            # Save checkpoint
            if (epoch + 1) % args.save_interval == 0 or epoch == args.epochs - 1:
                checkpoint_path = os.path.join(
                    args.output_dir, f"checkpoint_epoch_{epoch+1}.pth"
                )
                model_state = (
                    model.module.state_dict() if world_size > 1 else model.state_dict()
                )
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model_state,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": avg_loss,
                        "accuracy": avg_accuracy,
                        "args": vars(args),
                        "world_size": world_size,
                    },
                    checkpoint_path,
                )
                logger.info(f"Checkpoint saved: {checkpoint_path}")

    total_time = time.time() - start_time

    if rank == 0:
        # Save final model
        final_model_path = os.path.join(args.output_dir, "final_model_distributed.pth")
        model_state = (
            model.module.state_dict() if world_size > 1 else model.state_dict()
        )
        torch.save(
            {
                "model_state_dict": model_state,
                "model_config": {
                    "input_size": args.input_size,
                    "hidden_size": args.hidden_size,
                    "num_classes": args.num_classes,
                },
                "final_loss": train_losses[-1],
                "final_accuracy": train_accuracies[-1],
                "args": vars(args),
                "world_size": world_size,
            },
            final_model_path,
        )

        # Save training results
        results = {
            "training_config": vars(args),
            "distributed_config": {"world_size": world_size, "backend": "gloo"},
            "final_metrics": {
                "loss": train_losses[-1],
                "accuracy": train_accuracies[-1],
                "total_time": total_time,
            },
            "epoch_losses": train_losses,
            "epoch_accuracies": train_accuracies,
            "model_path": final_model_path,
        }

        results_path = os.path.join(
            args.output_dir, "distributed_training_results.json"
        )
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Distributed training completed in {total_time:.2f} seconds")
        logger.info(f"Final loss: {train_losses[-1]:.4f}")
        logger.info(f"Final accuracy: {train_accuracies[-1]:.4f}")
        logger.info(f"Results saved to: {results_path}")

        print(f"\nDistributed Training Summary:")
        print(f"World Size: {world_size}")
        print(f"Final Loss: {train_losses[-1]:.4f}")
        print(f"Final Accuracy: {train_accuracies[-1]:.4f}")
        print(f"Total Time: {total_time:.2f}s")
        print(f"Model saved to: {final_model_path}")

    return train_losses[-1], train_accuracies[-1], total_time


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Standard PyTorch Distributed Training Script"
    )

    # Model parameters
    parser.add_argument(
        "--input-size", type=int, default=784, help="Input size (default: 784)"
    )
    parser.add_argument(
        "--hidden-size", type=int, default=128, help="Hidden layer size (default: 128)"
    )
    parser.add_argument(
        "--num-classes", type=int, default=10, help="Number of classes (default: 10)"
    )

    # Training parameters
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs (default: 10)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size per process (default: 64)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)",
    )

    # Dataset parameters
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10000,
        help="Number of synthetic samples (default: 10000)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )

    # Training settings
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of data loader workers (default: 2)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=50,
        help="Log interval for batches (default: 50)",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=5,
        help="Save checkpoint every N epochs (default: 5)",
    )

    # Output settings
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs_distributed",
        help="Output directory (default: ./outputs_distributed)",
    )

    args = parser.parse_args()

    try:
        final_loss, final_accuracy, total_time = train_distributed(args)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
