#!/usr/bin/env python3
"""
Standard PyTorch Training Script
This is a typical PyTorch training script that follows standard patterns:
- Argparse for command line arguments
- Standard training loop structure
- Basic logging and checkpointing
- No runnable-specific code

This script can be executed directly or through runnable's ShellTask.
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

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


# Note that the annotation of the args is important
def train_model(args: argparse.Namespace) -> dict:
    os.makedirs(args.output_dir, exist_ok=True)
    """Main training function."""
    logger.info(f"Starting training with parameters: {vars(args)}")

    # Set device
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu_only else "cpu"
    )
    logger.info(f"Using device: {device}")

    # Create dataset and dataloader
    train_dataset = SyntheticDataset(
        num_samples=args.num_samples,
        input_size=args.input_size,
        num_classes=args.num_classes,
        seed=args.seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    # Create model
    model = SimpleNet(
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        num_classes=args.num_classes,
    ).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Training metrics
    train_losses = []
    train_accuracies = []

    logger.info(f"Starting training for {args.epochs} epochs")
    start_time = time.time()

    for epoch in range(args.epochs):
        epoch_start = time.time()
        model.train()

        epoch_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
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

            if batch_idx % args.log_interval == 0:
                logger.info(
                    f"Epoch {epoch+1}/{args.epochs}, Batch {batch_idx}/{len(train_loader)}, "
                    f"Loss: {loss.item():.6f}"
                )

        epoch_time = time.time() - epoch_start
        epoch_loss /= len(train_loader)
        epoch_accuracy = correct / total

        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)

        logger.info(
            f"Epoch {epoch+1}/{args.epochs}: "
            f"Loss: {epoch_loss:.4f}, "
            f"Accuracy: {epoch_accuracy:.4f}, "
            f"Time: {epoch_time:.2f}s"
        )

        # Save checkpoint every few epochs
        if (epoch + 1) % args.save_interval == 0 or epoch == args.epochs - 1:
            checkpoint_path = os.path.join(
                args.output_dir, f"checkpoint_epoch_{epoch+1}.pth"
            )
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": epoch_loss,
                    "accuracy": epoch_accuracy,
                    "args": vars(args),
                },
                checkpoint_path,
            )
            logger.info(f"Checkpoint saved: {checkpoint_path}")

    total_time = time.time() - start_time

    # Save final model
    final_model_path = os.path.join(args.output_dir, "final_model.pth")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "input_size": args.input_size,
                "hidden_size": args.hidden_size,
                "num_classes": args.num_classes,
            },
            "final_loss": train_losses[-1],
            "final_accuracy": train_accuracies[-1],
            "args": vars(args),
        },
        final_model_path,
    )

    # Save training results
    results = {
        "training_config": vars(args),
        "final_metrics": {
            "loss": train_losses[-1],
            "accuracy": train_accuracies[-1],
            "total_time": total_time,
        },
        "epoch_losses": train_losses,
        "epoch_accuracies": train_accuracies,
        "model_path": final_model_path,
    }

    results_path = os.path.join(args.output_dir, "training_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Training completed in {total_time:.2f} seconds")
    logger.info(f"Final loss: {train_losses[-1]:.4f}")
    logger.info(f"Final accuracy: {train_accuracies[-1]:.4f}")
    logger.info(f"Results saved to: {results_path}")

    return results


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description="Standard PyTorch Training Script")

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
        "--batch-size", type=int, default=64, help="Batch size (default: 64)"
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
        "--cpu-only",
        action="store_true",
        help="Force CPU usage even if CUDA is available",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of data loader workers (default: 2)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="Log interval for batches (default: 100)",
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
        default="./outputs",
        help="Output directory (default: ./outputs)",
    )

    args = parser.parse_args()

    # Create output directory

    # Train the model
    results = train_model(args)

    print(f"\nTraining Summary:")
    print(f"Final Loss: {results['final_metrics']['loss']:.4f}")
    print(f"Final Accuracy: {results['final_metrics']['accuracy']:.4f}")
    print(f"Total Time: {results['final_metrics']['total_time']:.2f}s")
    print(f"Model saved to: {results['model_path']}")


if __name__ == "__main__":
    main()
