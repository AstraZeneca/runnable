# multi_cpu_ddp_train_with_args.py

import argparse  # New: for command-line arguments
import os
import socket  # To find a free port for inter-process communication
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler


# --- DDP Setup/Cleanup Functions ---
def setup_ddp(rank, world_size, master_addr, master_port, backend="gloo"):
    """
    Sets up the distributed environment for a given process.
    """
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)  # Needs to be a string
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


def cleanup_ddp():
    """
    Cleans up the distributed environment.
    """
    dist.destroy_process_group()


# --- Training Worker Function (Executed by each CPU process) ---
def train_worker(
    rank, world_size, master_addr, master_port, backend, learning_rate, num_epochs
):
    """
    This function will be run by each CPU process.
    """
    setup_ddp(rank, world_size, master_addr, master_port, backend)

    # Log from only one process to avoid duplicate messages
    if rank == 0:
        print(f"--- Starting Multi-CPU DDP Training (Process {rank}/{world_size}) ---")
        print(f"Master Address: {master_addr}, Master Port: {master_port}")
        print(f"Learning Rate: {learning_rate}, Epochs: {num_epochs}")

    # 1. Define a simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 1)  # Input 10 features, output 1

        def forward(self, x):
            return self.linear(x)

    model = SimpleModel()
    device = torch.device("cpu")  # Explicitly set device to CPU for all processes
    model.to(device)

    # 2. Wrap the model with DistributedDataParallel (for CPU)
    ddp_model = DDP(model)  # DDP handles CPU processes directly

    # 3. Create a dummy dataset
    num_samples = 1000
    num_features = 10
    X = torch.randn(num_samples, num_features)
    y = (
        torch.sum(X * torch.arange(1, num_features + 1).float(), dim=1, keepdim=True)
        + torch.randn(num_samples, 1) * 0.1
    )

    dataset = TensorDataset(X, y)

    # 4. Use DistributedSampler to ensure each process gets a unique subset of data
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)

    # 5. Define optimizer and loss function (on the DDP model's parameters)
    optimizer = optim.SGD(ddp_model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    start_time_worker = time.time()  # Per-worker start time

    # 6. Training loop
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)

        ddp_model.train()
        total_loss = 0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = ddp_model(inputs)
            loss = criterion(outputs, targets)

            loss.backward()

            optimizer.step()
            total_loss += loss.item()

        avg_loss_tensor = torch.tensor(total_loss / len(dataloader), device=device)
        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
        global_avg_loss = avg_loss_tensor.item() / world_size

        if rank == 0:  # Only print from the main process
            print(
                f"Epoch {epoch+1}/{num_epochs}, Global Avg Loss: {global_avg_loss:.4f}"
            )

    end_time_worker = time.time()
    if rank == 0:
        print(
            f"\nMulti-CPU DDP Training complete on rank {rank} in {end_time_worker - start_time_worker:.2f} seconds!"
        )

        # Save the model
        model_save_path = (
            f"multi_cpu_ddp_model_lr{learning_rate}_epochs{num_epochs}.pth"
        )
        torch.save(ddp_model.module.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")

    cleanup_ddp()


def run_multi_cpu_training(learning_rate=0.01, num_epochs=50, batch_size=32):
    """
    Main function to run multi-CPU DDP training.
    This function is compatible with runnable and takes parameters directly.
    """
    print(
        f"Parameters: learning_rate={learning_rate}, num_epochs={num_epochs}, batch_size={batch_size}"
    )

    world_size = os.cpu_count()
    if world_size is None or world_size < 1:
        print("Could not determine CPU count or CPU count is zero. Cannot launch DDP.")
        return

    backend = "gloo"
    print(f"Attempting to launch {world_size} CPU processes using '{backend}' backend.")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        master_port = s.getsockname()[1]

    master_addr = "127.0.0.1"
    start_time_global = time.time()

    # Pass parameters to the train_worker function
    mp.spawn(
        train_worker,
        args=(world_size, master_addr, master_port, backend, learning_rate, num_epochs),
        nprocs=world_size,
        join=True,
    )

    end_time_global = time.time()
    print(
        f"\nTotal Multi-CPU DDP training duration: {end_time_global - start_time_global:.2f} seconds across {world_size} processes."
    )
    print("All DDP processes completed successfully.")


# --- Main Launcher Function (Entry point for the script) ---
def main_ddp_launcher(args):
    """
    Launches multiple CPU processes for DDP training.
    This is the legacy function that accepts args object.
    """
    run_multi_cpu_training(
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        batch_size=getattr(args, "batch_size", 32),  # Default to 32 if not provided
    )


if __name__ == "__main__":
    # 1. Set up argument parser
    parser = argparse.ArgumentParser(
        description="Multi-CPU PyTorch DDP Training Example."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,
        help="Learning rate for the optimizer (default: 0.01)",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)",
    )
    # Add more arguments as needed, e.g., --batch_size, --model_arch, etc.

    # 2. Parse command-line arguments
    args = parser.parse_args()

    # 3. Call the main launcher with parsed arguments
    main_ddp_launcher(args)
