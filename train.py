import fire
from model import LeNet5
from utils import RANDOM_SEED
from loader import load_dataset
from prepdata import make_dataset
import torch


def main(batch_size: int = 32, learning_rate: float = 0.01, epochs: int = 2):
    """
    How to run the code:
        input: python3 train.py --batch_size b --learning_rate lr --epochs e
    ---
    Train the dataset
    return None
    """
    torch.manual_seed(RANDOM_SEED)
    train_data, test_data = make_dataset()
    train_loader = load_dataset(data=train_data, batchsize=batch_size, is_shuffle=True)
    test_loader = load_dataset(data=test_data, batchsize=batch_size, is_shuffle=False)
    model = LeNet5()


if __name__ == "__main__":
    fire.Fire(main)
