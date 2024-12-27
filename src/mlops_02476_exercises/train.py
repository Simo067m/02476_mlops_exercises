import click
import matplotlib.pyplot as plt
import torch
import os

from typing import Callable, Tuple
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import Optimizer
import torch.nn as nn

from model import MNISTModel
from visualize import visualize_training
from data import get_corrupt_mnist_dataloaders

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# @click.group()
# def cli():
#     """Command line interface."""
#     pass


def train_one_epoch(model: nn.Module, dataloader: DataLoader, optimizer: Optimizer, loss_fn: nn.Module) -> float:
    """
    Train the model for one epoch.
    """

    model.train()
    running_loss = 0.0
    total_samples = 0
    accuracy = 0
    num_batches = len(dataloader)
    for data, targets in dataloader:
        data, targets = data.to(DEVICE), targets.to(DEVICE)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(data)
        loss = loss_fn(outputs, targets)

        # Backward pass
        loss.backward()

        # Update the weights
        optimizer.step()

        # Accumulate loss
        running_loss += loss.item() * data.size(0)
        total_samples += data.size(0)
        accuracy += (outputs.argmax(dim=1) == targets).float().mean().item()

        # Compute the average loss for the epoch
    epoch_loss = running_loss / total_samples
    epoch_accuracy = accuracy / num_batches
    return epoch_loss, epoch_accuracy


# @click.command()
# @click.option("--model", default=MNISTModel(), help="model to train")
# @click.option("--data_loader_func", default=get_corrupt_mnist_dataloaders, help="function to get the data loader")
# @click.option("--lr", default=1e-3, help="learning rate to use for training")
# @click.option("--batch_size", default=32, help="batch size to use for training")
# @click.option("--epochs", default=10, help="number of epochs to train for")
def train(model: nn.Module, data_loader_func: Callable[[Path], int], lr: float, batch_size: int, epochs: int) -> Tuple[list, list]:
    """Train a model."""
    print("Training started")
    print(f"Training model {model.__class__.__name__} for {epochs} epochs with lr={lr} and batch_size={batch_size}")

    # Send the model to the dataloader
    model = model.to(DEVICE)

    # Get the dataloader - it is assumed that the dataloader function returns a tuple of train and test dataloaders, however, we only need the train dataloader
    train_loader, _ = data_loader_func(batch_size=batch_size)

    # Define the loss function and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    statistics = {"train_loss": [], "train_accuracy": []}

    for epoch in range(epochs):
        train_loss, train_accuracy = train_one_epoch(model, train_loader, optimizer, loss_fn)
        print(f"Epoch {epoch + 1}, loss: {train_loss:2f}, accuracy: {train_accuracy:2f}")
        statistics["train_loss"].append(train_loss)
        statistics["train_accuracy"].append(train_accuracy)

    # Save the model
    if not os.path.exists("models/state_dicts"):
        os.makedirs("models/state_dicts")

    torch.save(model.state_dict(), f"models/state_dicts/{model.__class__.__name__}.pth")

    # Visualize the training process
    visualize_training(statistics["train_loss"], statistics["train_accuracy"], model_name=model.__class__.__name__)

    print("Training complete")

    return statistics["train_loss"], statistics["train_accuracy"]


if __name__ == "__main__":
    model = MNISTModel()

    _, _ = train(model, get_corrupt_mnist_dataloaders, lr=1e-3, batch_size=32, epochs=10)
