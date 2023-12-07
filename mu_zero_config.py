from __future__ import division, absolute_import, print_function
import collections
from typing import Optional

import torch
KnownBounds = collections.namedtuple('KnownBounds', ['min', 'max'])


class MuZeroConfig(object):
    def __init__(self,
                 action_space_size: int,
                 max_moves: int,
                 discount: float,
                 dirichlet_alpha: float,
                 num_simulations: int,
                 batch_size: int,
                 td_steps: int,
                 num_actors: int,
                 lr_init: float,
                 lr_decay_steps: float,
                 visit_softmax_temperature_fn,
                 game_class,
                 width,
                 height,
                 known_bounds: Optional[KnownBounds] = None
                 ):
        ### Self-Play
        self.action_space_size = action_space_size
        self.num_actors = num_actors

        self.visit_softmax_temperature_fn = visit_softmax_temperature_fn
        self.max_moves = max_moves
        self.num_simulations = num_simulations
        self.discount = discount

        # Root prior exploration noise.
        self.root_dirichlet_alpha = dirichlet_alpha
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        # If we already have some information about which values occur in the
        # environment, we can use them to initialize the rescaling.
        # This is not strictly necessary, but establishes identical behaviour to
        # AlphaZero in board games.
        self.known_bounds = known_bounds

        ### Training
        self.training_steps = int(1e6)
        self.checkpoint_interval = int(100)
        self.window_size = int(1e6)
        self.batch_size = batch_size
        self.num_unroll_steps = 4
        self.td_steps = td_steps
        self.checkpoint_plays = int(10)

        self.weight_decay = 1e-4
        self.momentum = 0.9

        # Exponential learning rate schedule
        self.lr_init = lr_init
        self.lr_decay_rate = 0.1
        self.lr_decay_steps = lr_decay_steps

        self.num_filters = 2
        self.num_blocks = 8

        self.game_class = game_class

        self.width = width
        self.height = height

    def new_game(self):
        return self.game_class(self.action_space_size, self.discount)


def visit_softmax_temperature(num_moves, training_steps):
    if num_moves < 30:
        return 1.0
    else:
        return 0.0  # Play according to the max.


def make_board_game_config(action_space_size: int, max_moves: int,
                           dirichlet_alpha: float,
                           lr_init: float,
                           game_class,
                           width,
                           height,
                           ) -> MuZeroConfig:
    return MuZeroConfig(
        action_space_size=action_space_size,
        max_moves=max_moves,
        discount=1.0,
        dirichlet_alpha=dirichlet_alpha,
        num_simulations=10,
        batch_size=64,
        td_steps=max_moves,  # Always use Monte Carlo return.
        num_actors=6 if torch.cuda.is_available() else 2,
        lr_init=lr_init,
        lr_decay_steps=400e3,
        visit_softmax_temperature_fn=visit_softmax_temperature,
        known_bounds=KnownBounds(-1, 1),
        game_class=game_class,
        width=width,
        height=height
    )
