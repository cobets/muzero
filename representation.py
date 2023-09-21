from __future__ import division, absolute_import, print_function

from torch import nn as nn
from torch.nn import functional as F

from conv import Conv
from mu_zero_config import num_filters, num_blocks
from residual_block import ResidualBlock


class Representation(nn.Module):
    ''' Conversion from observation to inner abstract state '''

    def __init__(self, input_shape):
        super().__init__()
        self.input_shape = input_shape
        self.board_size = self.input_shape[1] * self.input_shape[2]

        self.layer0 = Conv(self.input_shape[0], num_filters, 3, bn=True)
        self.blocks = nn.ModuleList([ResidualBlock(num_filters) for _ in range(num_blocks)])

    def forward(self, x):
        h = F.relu(self.layer0(x))
        for block in self.blocks:
            h = block(h)
        return h
