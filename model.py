import torch
import torchvision
from torch import nn


class LeNet5(nn.Module):
    """
    LeNet 5 Model
    """

    def __init__(self) -> None:
        super(LeNet5, self).__init__()
        self.C1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.avgpool = nn.AvgPool2d(kernel_size=2)
        self.act = nn.Tanh()
        self.C3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.C5 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)
        self.F6 = nn.Linear(in_features=120, out_features=84)
        self.output = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        out = self.act(self.C1(x))
        out = self.avgpool(out)
        out = self.act(self.C3(out))
        out = self.avgpool(out)
        out = self.act(self.C5(out))
        out = out.reshape(out.shape[0], -1)
        out = self.act(self.F6(out))
        out = self.output(out)

        return out
