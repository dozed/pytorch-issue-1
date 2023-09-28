import torch
from torch import nn


class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super().__init__()
        self.conv_1x1 = nn.Conv2d(in_channels, ch1x1, 1)
        self.relu_1x1 = nn.ReLU(inplace=True)
        self.conv_3x3_reduce = nn.Conv2d(in_channels, ch3x3red, 1)
        self.relu_3x3_reduce = nn.ReLU(inplace=True)
        self.conv_3x3 = nn.Conv2d(ch3x3red, ch3x3, 3, padding=1)
        self.relu_3x3 = nn.ReLU(inplace=True)
        self.conv_5x5_reduce = nn.Conv2d(in_channels, ch5x5red, 1)
        self.relu_5x5_reduce = nn.ReLU(inplace=True)
        self.conv_5x5 = nn.Conv2d(ch5x5red, ch5x5, 5, padding=2)
        self.relu_5x5 = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(3, stride=1, padding=1)
        self.pool_proj = nn.Conv2d(in_channels, pool_proj, 1)
        self.relu_pool_proj = nn.ReLU(inplace=True)

    def forward(self, x):
        branch_1 = self.relu_1x1(self.conv_1x1(x))
        branch_2 = self.relu_3x3_reduce(self.conv_3x3_reduce(x))
        branch_2 = self.relu_3x3(self.conv_3x3(branch_2))
        branch_3 = self.relu_5x5_reduce(self.conv_5x5_reduce(x))
        branch_3 = self.relu_5x5(self.conv_5x5(branch_3))
        branch_4 = self.pool(x)
        branch_4 = self.relu_pool_proj(self.pool_proj(branch_4))
        return torch.cat([branch_1, branch_2, branch_3, branch_4], dim=1)
