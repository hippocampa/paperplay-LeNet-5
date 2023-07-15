import fire
from dataset import data
from dataset import load_data
import torch
import torch.utils.data
import model
import torchmetrics
from torchmetrics import Accuracy
from tqdm import tqdm


def train(
    loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    acc_fn: torchmetrics.Metric,
    device: str,
):
    """
    Train the model

    Args:
        loader: Our data loader
        model: Our deep learning model
        loss_fn: Our loss function
        optimizer: Our optimizer
        acc_fn: Our accuracy metrics function
        device: Our device

    Returns:
        Train loss and Train Accuracy
    """

    for batch, (X, y) in enumerate(loader):
        pass
    return 0, 0


def main(batchsize: int = 1, learningrate: float = 0.01, epochs: int = 2):
    torch.manual_seed(42)
    train_set, test_set = data.get_data()
    train_loader, test_loader = load_data.loader(
        train_set, test_set, batchsize=batchsize
    )

    # model
    LeNet = model.LeNet5()

    loss_f = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=LeNet.parameters(), lr=learningrate)
    acc_f = Accuracy(task="multiclass", num_classes=10)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(type(LeNet), type(loss_f), type(optimizer))

    for i in tqdm(range(epochs)):
        train_loss, train_acc = train(
            loader=train_loader,
            model=LeNet,
            loss_fn=loss_f,
            optimizer=optimizer,
            acc_fn=acc_f,
            device=device,
        )


if __name__ == "__main__":
    fire.Fire(main)
