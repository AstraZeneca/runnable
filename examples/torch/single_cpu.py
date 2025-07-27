# single_cpu_train_with_args.py

import argparse  # New: for command-line arguments
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def run_single_cpu_training(learning_rate=0.01, num_epochs=50, batch_size=32):
    """
    Runs a simple training loop on a single CPU core.
    Accepts parsed arguments for hyperparameters.
    """
    print(
        f"Parameters: learning_rate={learning_rate}, num_epochs={num_epochs}, batch_size={batch_size}"
    )
    print("--- Starting Single-CPU Training ---")
    print(f"Learning Rate: {learning_rate}, Epochs: {num_epochs}")
    print(f"Batch Size: {batch_size}")

    # 1. Define a simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 1)  # Input 10 features, output 1

        def forward(self, x):
            return self.linear(x)

    model = SimpleModel()
    device = torch.device("cpu")  # Explicitly set device to CPU
    model.to(device)

    # 2. Create a dummy dataset
    num_samples = 1000  # Larger dataset to see the difference in speed later
    num_features = 10
    X = torch.randn(num_samples, num_features)
    y = (
        torch.sum(X * torch.arange(1, num_features + 1).float(), dim=1, keepdim=True)
        + torch.randn(num_samples, 1) * 0.1
    )

    dataset = TensorDataset(X, y)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )  # Use batch_size parameter

    # 3. Define optimizer and loss function
    optimizer = optim.SGD(
        model.parameters(), lr=learning_rate
    )  # Use learning_rate parameter
    criterion = nn.MSELoss()

    start_time = time.time()

    # 4. Training loop
    for epoch in range(num_epochs):  # Use num_epochs parameter
        model.train()
        total_loss = 0
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    end_time = time.time()
    print(f"\nSingle-CPU Training complete in {end_time - start_time:.2f} seconds!")

    # Save the model
    model_save_path = f"single_cpu_model_lr{learning_rate}_epochs{num_epochs}.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single-CPU PyTorch Training Example.")
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
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training (default: 32)",
    )

    args = parser.parse_args()
    run_single_cpu_training(**vars(args))
