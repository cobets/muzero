from __future__ import division, absolute_import, print_function

import torch
from torch import nn as nn
from torch.nn import functional as F

from conv import Conv
from mu_zero_config import num_filters


class Prediction(nn.Module):
    # Policy and value prediction from inner abstract state
    def __init__(self, action_shape):
        super().__init__()
        self.board_size = 42
        self.action_size = action_shape

        self.conv_p1 = Conv(num_filters, 4, 1, bn=True)
        self.conv_p2 = Conv(4, 1, 1)

        self.conv_v = Conv(num_filters, 4, 1, bn=True)
        self.fc_v = nn.Linear(self.board_size * 4, 1, bias=False)

    def forward(self, rp):
        h_p = F.relu(self.conv_p1(rp))
        h_p = self.conv_p2(h_p).view(-1, self.action_size)

        h_v = F.relu(self.conv_v(rp))
        h_v = self.fc_v(h_v.view(-1, self.board_size * 4))

        # range of value is -1 ~ 1
        return F.softmax(h_p, dim=-1), torch.tanh(h_v)
