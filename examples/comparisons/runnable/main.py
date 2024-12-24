# https://github.com/pytorch/examples/blob/main/mnist/main.py
# The main function is modularized into many smaller functions.

# ruff: noqa

"""
The captured data/logs can be found in the.catalog folder.

tree .catalog/pounded-swanson-0626
├── data
│   └── MNIST
│       └── raw
│           ├── t10k-images-idx3-ubyte
│           ├── t10k-images-idx3-ubyte.gz
│           ├── t10k-labels-idx1-ubyte
│           ├── t10k-labels-idx1-ubyte.gz
│           ├── train-images-idx3-ubyte
│           ├── train-images-idx3-ubyte.gz
│           ├── train-labels-idx1-ubyte
│           └── train-labels-idx1-ubyte.gz
├── dataloadersJYZ.execution.log
├── device.dill
├── devicetaskJ6P.execution.log
├── mnist_cnn.pt_epoch_1.pt
├── mnist_cnn.pt_epoch_10.pt
├── mnist_cnn.pt_epoch_11.pt
├── mnist_cnn.pt_epoch_12.pt
├── mnist_cnn.pt_epoch_13.pt
├── mnist_cnn.pt_epoch_14.pt
├── mnist_cnn.pt_epoch_2.pt
├── mnist_cnn.pt_epoch_3.pt
├── mnist_cnn.pt_epoch_4.pt
├── mnist_cnn.pt_epoch_5.pt
├── mnist_cnn.pt_epoch_6.pt
├── mnist_cnn.pt_epoch_7.pt
├── mnist_cnn.pt_epoch_8.pt
├── mnist_cnn.pt_epoch_9.pt
├── test_loader.dill
├── train_loader.dill
└── trainingloop0TF.execution.log

4 directories, 28 files
"""

from __future__ import print_function

# import argparse
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pydantic import BaseModel
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

from runnable import Catalog, Pipeline, PythonTask, pickled


# pydantic model for the convolutional layer parameters.
# This is not essential but it makes the code more readable.
class ConvParams(BaseModel):
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int


class Net(nn.Module):
    # A parameterized version of the Net class would be
    # def __init__(
    #     self,
    #     conv1_params: ConvParams,
    #     conv2_params: ConvParams,
    #     dropout1: float,
    #     dropout2: float,
    #     fc1: List[int],
    #     fc2: List[int],
    # ):
    #     super(Net, self).__init__()
    #     self.conv1 = nn.Conv2d(**conv1_params.model_dump())
    #     self.conv2 = nn.Conv2d(**conv2_params.model_dump())
    #     self.dropout1 = nn.Dropout(dropout1)
    #     self.dropout2 = nn.Dropout(dropout2)
    #     self.fc1 = nn.Linear(*fc1)
    #     self.fc2 = nn.Linear(*fc2)

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


# same as the source but with explicit arguments
def train(
    model,
    device,
    train_loader,
    optimizer,
    epoch,
    dry_run: bool = False,
    log_interval: int = 10,
):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
            if dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


# A function to prepare the dataset.
def prepare_dataset(train_batch_size: Dict[str, Any], test_batch_size: Dict[str, Any]):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    dataset1 = datasets.MNIST("data", train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST("data", train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_batch_size)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_batch_size)

    return train_loader, test_loader


# A function to identify the device to use.
def get_device(use_cuda: bool = False, use_mps: bool = False, seed: int = 1):
    use_cuda = not use_cuda and torch.cuda.is_available()
    use_mps = not use_mps and torch.backends.mps.is_available()

    torch.manual_seed(seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    return device


def fit_model(
    device,
    train_loader,
    test_loader,
    num_epochs: int,
    gamma: float = 0.7,
    dry_run: bool = False,
    lr: float = 1.0,
):
    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}")
        train(model, device, train_loader, optimizer, epoch, dry_run=dry_run)
        test(model, device, test_loader)
        scheduler.step()

        torch.save(model.state_dict(), f"mnist_cnn.pt_epoch_{epoch}.pt")


def main():
    # device = get_device(args.use_cuda, args.use_mps)
    device_task = PythonTask(
        name="device_task",
        function=get_device,
        returns=[pickled("device")],
    )

    data_catalog = Catalog(put=["data/**/*"])

    # train_loader, test_loader = prepare_dataset()
    data_loaders = PythonTask(
        name="data_loaders",
        function=prepare_dataset,
        returns=[pickled("train_loader"), pickled("test_loader")],
        catalog=data_catalog,
    )

    train_catalog = Catalog(put=["mnist_cnn.pt_epoch_*"], get=["data/**/*"])

    # fit_model(
    #     device,
    #     train_loader,
    #     test_loader,
    #     args.epochs,
    #     gamma=args.gamma,
    #     dry_run=args.dry_run,
    #     lr=args.lr,
    # )
    fit_model_task = PythonTask(
        name="training_loop",
        function=fit_model,
        catalog=train_catalog,
        terminate_with_success=True,
    )

    pipeline = Pipeline(steps=[device_task, data_loaders, fit_model_task])
    pipeline.execute(parameters_file="examples/comparisons/runnable_params.yaml")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    # parser.add_argument(
    #     "--batch-size", type=int, default=64, metavar="N", help="input batch size for training (default: 64)"
    # )
    # parser.add_argument(
    #     "--test-batch-size", type=int, default=1000, metavar="N", help="input batch size for testing (default: 1000)"
    # )
    # parser.add_argument("--epochs", type=int, default=14, metavar="N", help="number of epochs to train (default: 14)")
    # parser.add_argument("--lr", type=float, default=1.0, metavar="LR", help="learning rate (default: 1.0)")
    # parser.add_argument("--gamma", type=float, default=0.7, metavar="M", help="Learning rate step gamma (default: 0.7)")
    # parser.add_argument("--no-cuda", action="store_true", default=False, help="disables CUDA training")
    # parser.add_argument("--no-mps", action="store_true", default=False, help="disables macOS GPU training")
    # parser.add_argument("--dry-run", action="store_true", default=False, help="quickly check a single pass")
    # parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    # parser.add_argument(
    #     "--log-interval",
    #     type=int,
    #     default=10,
    #     metavar="N",
    #     help="how many batches to wait before logging training status",
    # )
    # parser.add_argument("--save-model", action="store_true", default=False, help="For Saving the current Model")
    # args = parser.parse_args()
    # main(args)
    main()
