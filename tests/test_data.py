import torch
from torch.utils.data import Dataset

from mlops_02476_exercises.data import MyDataset, MNISTDataset
from mlops_02476_exercises.data import get_corrupt_mnist_dataloaders

from tests import _PATH_DATA


def test_my_dataset():
    """Test the MyDataset class."""
    dataset = MNISTDataset("train", _PATH_DATA + "/processed")
    assert isinstance(dataset, Dataset)

def test_data():
    train = MNISTDataset("train", _PATH_DATA + "/processed")
    test = MNISTDataset("test", _PATH_DATA + "/processed")
    assert len(train) == 25000
    assert len(test) == 5000
    # for dataset in [train, test]:
    #     for x, y in dataset:
    #         assert x.shape == (1, 28, 28)
    #         assert y in range(10)
    # train_targets = torch.unique(train.tensors[1])
    # assert (train_targets == torch.arange(0,10)).all()
    # test_targets = torch.unique(test.tensors[1])
    # assert (test_targets == torch.arange(0,10)).all()