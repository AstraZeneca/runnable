import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms


# Simple CNN model definition
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.fc1 = nn.Linear(32 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 32 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def setup():
    # Retrieve rank and world size from environment variables
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    # Initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    return rank, world_size


def cleanup():
    dist.destroy_process_group()


def train():
    # Setup the distributed environment.
    rank, world_size = setup()

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    # Create model and move it to the appropriate device
    model = SimpleCNN().to(device)
    # model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    # Load the FashionMNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.FashionMNIST(
        root="./data", train=True, transform=transform, download=True
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank
    )
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=64, sampler=train_sampler
    )

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Training loop
    model.train()
    for epoch in range(10):  # number of epochs
        train_sampler.set_epoch(epoch)  # Shuffle data
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        print(f"Rank {rank}, Epoch {epoch+1}, Loss: {loss.item()}")

    # Cleanup
    cleanup()


if __name__ == "__main__":
    # world_size = 2  # Number of processes (change according to your setup)
    # mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
    train()
