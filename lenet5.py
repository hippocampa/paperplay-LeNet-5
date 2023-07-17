import torch
from torch import nn


class LeNetOri(nn.Module):
    """
    Original LeNet-5 Model
    """

    def __init__(self) -> None:
        super(LeNetOri, self).__init__()
        self.C1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.avgpool = nn.AvgPool2d(kernel_size=2)
        self.act = nn.Tanh()
        # conv2D -- For C3 Layer
        self.fsConv2D = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=5)
        self.scConv2D = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=5)
        self.thConv2D = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=5)
        self.foConv2D = nn.Conv2d(in_channels=6, out_channels=1, kernel_size=5)
        self.C5 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)
        self.F6 = nn.Linear(in_features=120, out_features=84)
        self.output = nn.Linear(in_features=84, out_features=10)

    def convC3(self, tensor) -> torch.Tensor:
        """
        Custom C3 layer consist of sub layers
        """
        sublayer_1 = self._get_sub_layer1(tensor)
        sublayer_2 = self._get_sub_layer2(tensor)
        sublayer_3 = self._get_sub_layer3(tensor)
        sublayer_4 = self.foConv2D(tensor)

        concatenated = torch.cat(
            (sublayer_1, sublayer_2, sublayer_3, sublayer_4),
            dim=1,
        )
        return concatenated

    def _get_sub_layer1(self, t) -> torch.Tensor:
        """
        Create sub layer 1

        Args:
            t: tensor

        Return:
            Concatenated layers (sub layer)
        """
        initial_index = torch.tensor([0, 1, 2])
        subl = torch.empty_like(t, dtype=torch.long)
        for i in range(0, 6):
            if i == 0:
                subl = self.fsConv2D(torch.index_select(t, 1, (initial_index % 6)))
            else:
                temp = torch.index_select(t, 1, (initial_index % 6))
                subl = torch.cat((subl, self.fsConv2D(temp)), dim=1)
            initial_index += 1
        return subl

    def _get_sub_layer2(self, t) -> torch.Tensor:
        """
        Create sub layer 1

        Args:
            t: tensor

        Return:
            Concatenated layers (sub layer)
        """
        initial_index = torch.tensor([0, 1, 2, 3])
        subl = torch.empty_like(t, dtype=torch.long)
        for i in range(0, 6):
            if i == 0:
                subl = self.scConv2D(torch.index_select(t, 1, (initial_index % 6)))
            else:
                temp = torch.index_select(t, 1, (initial_index % 6))
                subl = torch.cat((subl, self.scConv2D(temp)), dim=1)
            initial_index += 1
        return subl

    def _get_sub_layer3(self, t) -> torch.Tensor:
        """
        Create sub layer 1

        Args:
            t: tensor

        Return:
            Concatenated layers (sub layer)
        """
        initial_index = torch.tensor([0, 1, 3, 4])
        subl = torch.empty_like(t, dtype=torch.long)
        for i in range(0, 3):
            if i == 0:
                subl = self.thConv2D(torch.index_select(t, 1, (initial_index % 6)))
            else:
                temp = torch.index_select(t, 1, (initial_index % 6))
                subl = torch.cat((subl, self.thConv2D(temp)), dim=1)
            initial_index += 1
        return subl

    def forward(self, x):
        out = self.act(self.C1(x.type(torch.float32)))
        out = self.avgpool(out)
        out = self.act(self.convC3(out))
        out = self.avgpool(out)
        out = self.act(self.C5(out))
        out = out.reshape(out.shape[0], -1)
        out = self.act(self.F6(out))
        out = self.output(out)
        return out
