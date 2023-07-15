import torchvision
from torch.utils.data import Dataset
from torchvision import datasets
from pathlib import Path
import shutil
from .transform import transform_fn


def create_dir(path: Path) -> None:
    """
    Create a new directory

    Args:
        Directory's path

    Returns:
        None
    """
    path.mkdir(parents=True, exist_ok=True)


def get_data() -> tuple:
    """
    Siapkan dataset training dan testing

    Args:
        Write here

    Returns:
        Training and testing dataset
    """
    DATA_DIR = Path("../data")
    if DATA_DIR.is_dir():  # if directory exist, delete it
        shutil.rmtree(DATA_DIR)

    # create a new dir
    create_dir(DATA_DIR)

    # download the dataset
    train_data = datasets.MNIST(
        root=str(DATA_DIR), train=True, download=True, transform=transform_fn
    )

    test_data = datasets.MNIST(
        root=str(DATA_DIR), train=False, download=True, transform=transform_fn
    )

    return train_data, test_data
