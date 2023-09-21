from __future__ import division, absolute_import, print_function

from mu_zero_config import make_connect4_config
from network import Network


class SharedStorage(object):
    def __init__(self, device):
        self._networks = {}
        self.device = device

    def latest_network(self) -> Network:
        if self._networks:
            return self._networks[max(self._networks.keys())]
        else:
            # policy -> uniform, value -> 0, reward -> 0
            return make_uniform_network(self.device)

    def old_network(self) -> Network:
        if self._networks:
            return self._networks[min(self._networks.keys())]
        else:
            # policy -> uniform, value -> 0, reward -> 0
            return make_uniform_network(self.device)

    def save_network(self, step: int, network: Network):
        self._networks[step] = network


def make_uniform_network(device):
    return Network(make_connect4_config().action_space_size, device).to(device)
