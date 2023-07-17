import fire
from dataset import data
from dataset import load_data
import torch
import torch.utils.data
import model
import torchmetrics
from torchmetrics import Accuracy
from typing import Tuple, Any
from progress.bar import Bar


def train(
    loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    acc_fn: torchmetrics.Metric,
    device: str,
) -> Tuple[Any, Any]:
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
    t_loss = 0
    t_acc = 0
    model.train()
    with Bar("Training model...", max=len(loader)) as bar:
        for batch, (X, y) in enumerate(loader):
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            t_loss += loss
            t_acc += acc_fn(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            bar.next()
        # Average loss and accuracy per batch
        t_loss = t_loss / len(loader)
        t_acc = t_acc / len(loader)
    return t_loss, t_acc


def test(
    loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    acc_fn: torchmetrics.Metric,
    device: str,
) -> Tuple[Any, Any]:
    """
    Testing the model

    Args:
        model: pytorch model
        loss_fn: pytorch loss function
        acc_fn: torchmetrics multiclass accuracy function
        device: pytorch device (cpu or gpu)

    Returns:
        A tuple of test loss and test accuracy.
    """
    test_loss = 0
    test_acc = 0

    model.eval()
    with torch.inference_mode():
        with Bar("Testing model...", max=len(loader)) as bar:
            for X, y in loader:
                X, y = X.to(device), y.to(device)
                y_pred = model(X)
                loss = loss_fn(y_pred, y)
                test_loss += loss
                test_acc += acc_fn(y_pred, y)
                bar.next()
            test_loss = test_loss / len(loader)
            test_acc = test_acc / len(loader)
    return test_loss, test_acc


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

    for i in range(0, epochs):
        print(f"Epoch {i}")
        train_loss, train_acc = train(
            loader=train_loader,
            model=LeNet.to(device),
            loss_fn=loss_f,
            optimizer=optimizer,
            acc_fn=acc_f,
            device=device,
        )
        test_loss, test_acc = test(test_loader, LeNet, loss_f, acc_f, device)
        print(
            f"Train loss: {train_loss}, Train acc: {train_acc}, Test loss: {test_loss}, Test acc: {test_acc} "
        )
    PATH = "savedmodels/mark1.pt"
    torch.save(LeNet.state_dict(), PATH)
    print(f"Training is complete, model is saved at {PATH}")


if __name__ == "__main__":
    fire.Fire(main)
