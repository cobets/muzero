from __future__ import division, absolute_import, print_function

from torch import nn as nn
from torch.nn import functional as F

from conv import Conv


class ResidualBlock(nn.Module):
    def __init__(self, filters):
        super().__init__()
        self.conv = Conv(filters, filters, 3, True)

    def forward(self, x):
        return F.relu(x + (self.conv(x)))
