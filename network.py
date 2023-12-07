from __future__ import division, absolute_import, print_function

import typing
from typing import Dict, List

import numpy
import torch
from torch import nn as nn

from action import Action
from dynamics import Dynamics
from prediction import Prediction
from representation import Representation


class NetworkOutput(typing.NamedTuple):
    value: float
    reward: float
    policy_logits: Dict[Action, float]
    hidden_state: List[float]


class Network(nn.Module):

    def __init__(self, action_space_size: int, device, num_filters, num_blocks, width, height):
        super().__init__()
        self.steps = 0
        self.action_space_size = action_space_size
        input_shape = (2, height, width)
        rp_shape = (num_filters, *input_shape[1:])
        self.device = device
        self.representation = Representation(input_shape, num_filters, num_blocks).to(device)
        self.prediction = Prediction(action_space_size, num_filters, width, height).to(device)
        self.dynamics = Dynamics(rp_shape, (2, height, width), num_filters, num_blocks).to(device)
        self.eval()
        self.width = width
        self.height = height

    def predict_initial_inference(self, x):
        assert x.ndim in (3, 4)
        assert x.shape == (2, self.height, self.width) or x.shape[1:] == (2, self.height, self.width)
        orig_x = x
        if x.ndim == 3:
            x = x.reshape(1, 2, self.height, self.width)

        x = torch.Tensor(x).to(self.device)
        h = self.representation(x)
        policy, value = self.prediction(h)

        if orig_x.ndim == 3:
            return h[0], policy[0], value[0]
        else:
            return h, policy, value

    def predict_recurrent_inference(self, x, a):

        if x.ndim == 3:
            x = x.reshape(1, 2, self.height, self.width)

        a = numpy.full((1, 2, self.height, self.width), a)

        g = self.dynamics(x, torch.Tensor(a).to(self.device))
        policy, value = self.prediction(g)

        return g[0], policy[0], value[0]

    def initial_inference(self, image) -> NetworkOutput:
        # representation + prediction function
        h, p, v = self.predict_initial_inference(image.astype(numpy.float32))
        return NetworkOutput(v, 0, p, h)

    def recurrent_inference(self, hidden_state, action) -> NetworkOutput:
        # dynamics + prediction function
        g, p, v = self.predict_recurrent_inference(hidden_state, action)
        return NetworkOutput(v, 0, p, g)

    def training_steps(self) -> int:
        # How many steps / batches the network has been trained for.
        return self.steps
