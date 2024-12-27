import typer
import torch

from pathlib import Path
from typing import Tuple
from torch.utils.data import Dataset, DataLoader


class MNISTDataset(Dataset):
    """Custom dataset for MNIST."""

    def __init__(self, prefix: str, data_folder: Path = "data/processed") -> None:
        self.X = torch.load(data_folder + f"/{prefix}_images.pt", weights_only=True)
        self.y = torch.load(data_folder + f"/{prefix}_target.pt", weights_only=True)

        assert len(self.X) == len(self.y), "X and y must have the same length."

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def get_corrupt_mnist_dataloaders(
    data_folder: Path = "data/processed", batch_size: int = 32
) -> Tuple[DataLoader, DataLoader]:
    """Return train and test dataloaders for corrupt MNIST."""
    train_set = MNISTDataset("train", data_folder)
    test_set = MNISTDataset("test", data_folder)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


class MyDataset(Dataset):
    """My custom dataset."""

    def __init__(self, raw_data_path: Path) -> None:
        self.data_path = raw_data_path

    def __len__(self) -> int:
        """Return the length of the dataset."""

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""


def preprocess(raw_data_path: Path, output_folder: Path) -> None:
    print("Preprocessing data...")
    dataset = MyDataset(raw_data_path)
    dataset.preprocess(output_folder)


if __name__ == "__main__":
    typer.run(preprocess)
