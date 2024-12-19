# Code as seen here: https://outerbounds.com/docs/use-pytorch-with-metaflow/

# ruff: noqa

import torch
import torch.nn as nn
import torch.nn.functional as F
from metaflow import FlowSpec, Parameter, conda_base, step
from torch_utilities import Net, get_data_loaders, test, train
from torchvision import datasets, transforms


class Net(nn.Module):
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


def train(model, train_loader, optimizer, epoch):
    model.train()
    for idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if idx * len(data) % 10000 == 0:
            out = "Train Epoch: " + "{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                epoch,
                idx * len(data),
                len(train_loader.dataset),
                100.0 * idx / len(train_loader),
                loss.item(),
            )
            print(out)


def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    return correct / len(test_loader.dataset)


def get_data_loaders(name="train"):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    if name == "train":
        dataset = datasets.MNIST(
            "../data", train=True, download=True, transform=transform
        )
        train_args = {"batch_size": 32}
        return dataset, train_args
    elif name == "test":
        dataset = datasets.MNIST("../data", train=False, transform=transform)
        test_args = {"batch_size": 32}
        return dataset, test_args


@conda_base(libraries={"pytorch": "1.11.0", "torchvision": "0.12.0"}, python="3.8")
class TorchFlow(FlowSpec):
    lr = Parameter("lr", default=0.01)
    epochs = Parameter("epochs", default=1)

    # It might be better to user a parameters.json?
    # There is an example json in this directory.

    @step
    def start(self):
        self.next(self.get_data)

    @step
    def get_data(self):
        import torch

        train_dataset, train_args = get_data_loaders()
        test_dataset, test_args = get_data_loaders("test")
        self.train_loader = torch.utils.data.DataLoader(train_dataset, **train_args)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, **test_args)
        self.next(self.fit_model)

    @step
    def fit_model(self):
        import torch.optim as optim
        from torch.optim.lr_scheduler import StepLR

        self.model = Net()
        optimizer = optim.Adadelta(self.model.parameters(), lr=self.lr)
        scheduler = StepLR(optimizer, step_size=1)
        for epoch in range(1, self.epochs + 1):
            train(self.model, self.train_loader, optimizer, epoch)
            _ = test(self.model, self.test_loader)
            scheduler.step()
        self.next(self.evaluate_model)

    @step
    def evaluate_model(self):
        self.test_score = test(self.model, self.test_loader)
        print(f"Model scored {100*self.test_score}%")
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    TorchFlow()
