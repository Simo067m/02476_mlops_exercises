import click
import torch


def normalize(images: torch.Tensor) -> torch.Tensor:
    """Normalize images."""
    return (images - images.mean()) / images.std()


@click.command()
@click.option("--raw_dir", default="data/raw", help="Path to raw data directory")
@click.option("--processed_dir", default="data/processed", help="Path to processed directory")
def make_data(raw_dir: str, processed_dir: str) -> None:
    """
    Processes raw data and save in the processed directory.
    """
    train_images, train_targets = [], []
    for i in range(5):
        train_images.append(torch.load(f"{raw_dir}/train_images_{i}.pt", weights_only=True))
        train_targets.append(torch.load(f"{raw_dir}/train_target_{i}.pt", weights_only=True))
    train_images = torch.cat(train_images)
    train_target = torch.cat(train_targets)

    test_images: torch.Tensor = torch.load(f"{raw_dir}/test_images.pt", weights_only=True)
    test_target: torch.Tensor = torch.load(f"{raw_dir}/test_target.pt", weights_only=True)

    train_images = train_images.unsqueeze(1).float()
    test_images = test_images.unsqueeze(1).float()
    train_target = train_target.long()
    test_target = test_target.long()

    train_images = normalize(train_images)
    test_images = normalize(test_images)

    torch.save(train_images, f"{processed_dir}/train_images.pt")
    torch.save(train_target, f"{processed_dir}/train_target.pt")
    torch.save(test_images, f"{processed_dir}/test_images.pt")
    torch.save(test_target, f"{processed_dir}/test_target.pt")


if __name__ == "__main__":
    make_data()
