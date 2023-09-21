from __future__ import division, absolute_import, print_function

import torch
from torch import nn as nn

from conv import Conv
from mu_zero_config import num_filters, num_blocks
from residual_block import ResidualBlock


class Dynamics(nn.Module):
    '''Abstruct state transition'''

    def __init__(self, rp_shape, act_shape):
        super().__init__()
        self.rp_shape = rp_shape
        self.layer0 = Conv(rp_shape[0] + act_shape[0], num_filters, 3, bn=True)
        self.blocks = nn.ModuleList([ResidualBlock(num_filters) for _ in range(num_blocks)])

    def forward(self, rp, a):
        h = torch.cat([rp, a], dim=1)
        h = self.layer0(h)
        for block in self.blocks:
            h = block(h)
        return h
