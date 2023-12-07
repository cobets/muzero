from __future__ import division, absolute_import, print_function
from network import Network


class SharedStorage(object):
    def __init__(self, device, action_space_size, num_filters, num_blocks, width, height):
        self._networks = {}
        self.device = device
        self.action_space_size = action_space_size
        self.num_filters = num_filters
        self.num_blocks = num_blocks
        self.width = width
        self.height = height

    def latest_network(self) -> Network:
        if self._networks:
            return self._networks[max(self._networks.keys())]
        else:
            # policy -> uniform, value -> 0, reward -> 0
            return self.make_uniform_network()

    def old_network(self) -> Network:
        if self._networks:
            return self._networks[min(self._networks.keys())]
        else:
            # policy -> uniform, value -> 0, reward -> 0
            return self.make_uniform_network()

    def save_network(self, step: int, network: Network):
        self._networks[step] = network

    def make_uniform_network(self):
        return Network(
            self.action_space_size,
            self.device,
            self.num_filters,
            self.num_blocks,
            self.height,
            self.width
        ).to(self.device)
