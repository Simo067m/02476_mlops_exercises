import click
import torch
import torch.nn as nn

from model import MNISTModel

@click.command()
def predict(model: nn.Module, input_img: torch.Tensor) -> torch.Tensor:
    return model(input_img)


if __name__ == "__main__":
    model = MNISTModel()
