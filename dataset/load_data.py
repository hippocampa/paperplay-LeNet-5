import torch
import torchvision
from torch.utils.data import DataLoader
from typing import Tuple


def loader(
    trainset: torchvision.datasets.mnist.MNIST,
    testset: torchvision.datasets.mnist.MNIST,
    batchsize: int = 1,
) -> Tuple[DataLoader, DataLoader]:
    """
    Dataloader for all dataset

    Args:
        trainset: Our training set
        testset: Our test set

    Returns:
        Dataloader
    """

    train_loader = DataLoader(trainset, batch_size=batchsize, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batchsize, shuffle=False)

    return train_loader, test_loader
