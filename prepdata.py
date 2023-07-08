from torchvision.datasets import MNIST
import os
from datatransform import transform_fn


def make_dataset():
    """
    Creating dataset
    """
    os.makedirs("data/")
    train_data = MNIST(root="data/", train=True, download=True, transform=transform_fn)
    test_data = MNIST(root="data/", train=False, download=True, transform=transform_fn)
    return train_data, test_data


if __name__ == "__main__":
    make_dataset()
