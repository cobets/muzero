# Lint as: python3
"""Pseudocode description of the MuZero algorithm."""
# pylint: disable=unused-argument
# pylint: disable=missing-docstring
# pylint: disable=assignment-from-no-return

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import queue
import signal
from typing import List
import multiprocessing

import numpy
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from action import Action
from action_history import ActionHistory
from environment import Winner, Player
from connect4_game import Game as Connect4Game
from dots_game import Game as DotsGame
from min_max_stats import MinMaxStats
from mu_zero_config import MuZeroConfig, make_board_game_config
from network import Network, NetworkOutput
from node import Node
from replay_buffer import ReplayBuffer
from shared_storage import SharedStorage

# https://stackoverflow.com/questions/5061582/setting-stacksize-in-a-python-script
# sys.setrecursionlimit(10**6)
# threading.stack_size(2**26)


################################################################################
# Testing the latest net
################################################################################

# Battle against random agents
def vs_random(network, config):
    results = {}
    for i in range(config.checkpoint_plays):
        first_turn = i % 2 == 0
        turn = first_turn
        game = config.new_game()
        r = 0
        while not game.terminal():
            if turn:
                root = Node(0)
                current_observation = game.make_image(-1)
                expand_node(root, game.to_play(), game.legal_actions(),
                            network.initial_inference(current_observation))
                add_exploration_noise(config, root)
                run_mcts(config, root, game.action_history(), network)
                action = select_action(config, len(game.history), root, network)
            else:
                action = numpy.random.choice(game.legal_actions())
            game.apply(action)
            turn = not turn
        if ((game.winner() == Winner.white and first_turn)
                or (game.winner() == Winner.black and not first_turn)):
            r = 1
        elif ((game.winner() == Winner.black and first_turn)
              or (game.winner() == Winner.white and not first_turn)):
            r = -1
        results[r] = results.get(r, 0) + 1
    return results


def random_vs_random(config):
    results = {}
    for i in range(config.checkpoint_plays):
        first_turn = i % 2 == 0
        turn = first_turn
        game = config.new_game()
        r = 0
        while not game.terminal():
            action = numpy.random.choice(game.legal_actions())
            game.apply(action)
            turn = not turn
        if ((game.winner() == Winner.white and first_turn)
                or (game.winner() == Winner.black and not first_turn)):
            r = 1
        elif ((game.winner() == Winner.black and first_turn)
              or (game.winner() == Winner.white and not first_turn)):
            r = -1
        results[r] = results.get(r, 0) + 1
    return results


def latest_vs_older(last, old, config):
    results = {}
    for i in range(config.checkpoint_plays):
        first_turn = i % 2 == 0
        turn = first_turn
        game = config.new_game()
        r = 0
        while not game.terminal():
            if turn:
                root = Node(0)
                current_observation = game.make_image(-1)
                expand_node(root, game.to_play(), game.legal_actions(),
                            last.initial_inference(current_observation))
                add_exploration_noise(config, root)
                run_mcts(config, root, game.action_history(), last)
                action = select_action(config, len(game.history), root, last)
            else:
                root = Node(0)
                current_observation = game.make_image(-1)
                expand_node(root, game.to_play(), game.legal_actions(),
                            old.initial_inference(current_observation))
                add_exploration_noise(config, root)
                run_mcts(config, root, game.action_history(), old)
                action = select_action(config, len(game.history), root, old)
            game.apply(action)
            turn = not turn
        if ((game.winner() == Winner.white and first_turn)
                or (game.winner() == Winner.black and not first_turn)):
            r = 1
        elif ((game.winner() == Winner.black and first_turn)
              or (game.winner() == Winner.white and not first_turn)):
            r = -1
        results[r] = results.get(r, 0) + 1
    return results


# MuZero training is split into two independent parts: Network training and
# self-play data generation.
# These two parts only communicate by transferring the latest network checkpoint
# from the training to the self-play, and the finished games from the self-play
# to the training.
def muzero(config: MuZeroConfig, device):
    storage = SharedStorage(
        device,
        config.action_space_size,
        config.num_filters,
        config.num_blocks,
        config.width,
        config.height
    )
    replay_buffer = ReplayBuffer(config)
    # stop event
    stop_event = multiprocessing.Event()
    signal.signal(signal.SIGINT, lambda sig, frame: stop_event.set())
    # Start n concurrent actor processes
    processes = list()
    network_queues = list()
    game_queue = multiprocessing.Queue()
    for _ in range(config.num_actors):
        q = multiprocessing.Queue()
        network_queues.append(q)
        p = multiprocessing.Process(target=run_selfplay, args=(config, stop_event, q, game_queue))
        processes.append(p)

    # Start all threads
    for x in processes:
        x.start()

    train_network(config, storage, replay_buffer, device, stop_event, network_queues, game_queue)

    return storage.latest_network()


##################################
# Part 1: Self-Play


# Each self-play job is independent of all others; it takes the latest network
# snapshot, produces a game and makes it available to the training job by
# writing it to a shared replay buffer.
def run_selfplay(config: MuZeroConfig, stop_event,
                 network_queue: multiprocessing.Queue,
                 game_queue: multiprocessing.Queue):
    wait_for_network = True

    while True:
        if stop_event.is_set():
            break

        try:
            network = network_queue.get(wait_for_network)
            wait_for_network = False
        except queue.Empty:
            pass

        game = play_game(config, network)
        game_queue.put(game)


# Each game is produced by starting at the initial board position, then
# repeatedly executing a Monte Carlo Tree Search to generate moves until the end
# of the game is reached.
def play_game(config: MuZeroConfig, network: Network):
    game = config.new_game()
    while not game.terminal() and len(game.history) < config.max_moves:
        # At the root of the search tree we use the representation function to
        # obtain a hidden state given the current observation.
        root = Node(0)
        current_observation = game.make_image(-1)
        expand_node(root, game.to_play(), game.legal_actions(),
                    network.initial_inference(current_observation))
        add_exploration_noise(config, root)

        # We then run a Monte Carlo Tree Search using only action sequences and the
        # model learned by the network.
        run_mcts(config, root, game.action_history(), network)
        action = select_action(config, len(game.history), root, network)
        game.apply(action)
        game.store_search_statistics(root)
    return game


# Core Monte Carlo Tree Search algorithm.
# To decide on an action, we run N simulations, always starting at the root of
# the search tree and traversing the tree according to the UCB formula until we
# reach a leaf node.
def run_mcts(config: MuZeroConfig, root: Node, action_history: ActionHistory, network: Network):
    min_max_stats = MinMaxStats(config.known_bounds)

    for _ in range(config.num_simulations):
        history = action_history.clone()
        node = root
        search_path = [node]

        while node.expanded():
            action, node = select_child(config, node, min_max_stats)
            history.add_action(action)
            search_path.append(node)

        # Inside the search tree we use the dynamics function to obtain the next
        # hidden state given an action and the previous hidden state.
        parent = search_path[-2]
        network_output = network.recurrent_inference(parent.hidden_state,
                                                     history.last_action())
        expand_node(node, history.to_play(), history.action_space(), network_output)

        backpropagate(search_path, network_output.value.item(), history.to_play(),
                      config.discount, min_max_stats)


def select_action(config: MuZeroConfig, num_moves: int, node: Node, network: Network):
    visit_counts = [
        (child.visit_count, action) for action, child in node.children.items()
    ]
    t = config.visit_softmax_temperature_fn(
        num_moves=num_moves, training_steps=network.training_steps())
    _, action = softmax_sample(visit_counts, t)
    return action


# Select the child with the highest UCB score.
def select_child(config: MuZeroConfig, node: Node, min_max_stats: MinMaxStats):
    _, action, child = max(
        (ucb_score(config, node, child, min_max_stats), action, child)
        for action, child in node.children.items()
    )
    return action, child


# The score for a node is based on its value, plus an exploration bonus based on
# the prior.
def ucb_score(config: MuZeroConfig, parent: Node, child: Node, min_max_stats: MinMaxStats) -> float:
    pb_c = math.log((parent.visit_count + config.pb_c_base + 1) /
                    config.pb_c_base) + config.pb_c_init
    pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

    prior_score = pb_c * child.prior
    value_score = min_max_stats.normalize(child.value())
    return prior_score + value_score


# We expand a node using the value, reward and policy prediction obtained from
# the neural network.
def expand_node(node: Node, to_play: Player, actions: List[Action],
                network_output: NetworkOutput):
    node.to_play = to_play
    node.hidden_state = network_output.hidden_state
    node.reward = network_output.reward
    policy = {a: math.exp(network_output.policy_logits[a]) for a in actions}
    policy_sum = sum(policy.values())
    for action, p in policy.items():
        node.children[action] = Node(p / policy_sum)


# At the end of a simulation, we propagate the evaluation all the way up the
# tree to the root.
def backpropagate(search_path: List[Node], value: float, to_play: Player,
                  discount: float, min_max_stats: MinMaxStats):
    for node in search_path:
        node.value_sum += value if node.to_play == to_play else -value
        node.visit_count += 1
        min_max_stats.update(node.value())

        value = node.reward + discount * value


# At the start of each search, we add dirichlet noise to the prior of the root
# to encourage the search to explore new actions.
def add_exploration_noise(config: MuZeroConfig, node: Node):
    actions = list(node.children.keys())
    noise = numpy.random.dirichlet([config.root_dirichlet_alpha] * len(actions))
    frac = config.root_exploration_fraction
    for a, n in zip(actions, noise):
        node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac


# End Self-Play
##################################

##################################
# Part 2: Training

def save_network(storage: SharedStorage, step: int, network: Network, network_queues: list):
    storage.save_network(step, network)

    for q in network_queues:
        q.put(network)


def train_network(config: MuZeroConfig, storage: SharedStorage,
                  replay_buffer: ReplayBuffer, device, stop_event,
                  network_queues: list, game_queue: multiprocessing.Queue):

    un = storage.make_uniform_network()
    for q in network_queues:
        q.put(un)

    writer = SummaryWriter()

    network = Network(
        config.action_space_size,
        device,
        config.num_filters,
        config.num_blocks,
        config.width,
        config.height
    ).to(device)

    replay_buffer.save_game(game_queue.get())

    optimizer = optim.SGD(network.parameters(), lr=0.01, weight_decay=config.lr_decay_rate,
                          momentum=config.momentum)

    for i in range(config.training_steps):
        if stop_event.is_set():
            break

        if i % config.checkpoint_interval == 0 and i > 0:
            save_network(storage, i, network, network_queues)
            # Test against random agent
            vs_random_once = vs_random(network, config)
            print(f'iter = {i} network_vs_random = {sorted(vs_random_once.items())}')
            vs_older = latest_vs_older(storage.latest_network(), storage.old_network(), config)
            print(f'iter = {i} lastnet_vs_older = {sorted(vs_older.items())}')
            writer.add_scalars(
                'train/score',
                {
                    'network win vs random': vs_random_once.get(-1, 0),
                    'last network win vs prev': vs_older.get(-1, 0)
                },
                i
            )

        while True:
            try:
                replay_buffer.save_game(game_queue.get_nowait())
            except queue.Empty:
                break

        batch = replay_buffer.sample_batch(config.num_unroll_steps, config.td_steps)
        p_loss, v_loss = update_weights(batch, network, optimizer, device)
        writer.add_scalar('train/p_loss', p_loss, i)
        writer.add_scalar('train/v_loss', v_loss, i)
        writer.add_scalar('train/replay size', len(replay_buffer.buffer), i)

    if not stop_event.is_set():
        save_network(storage, config.training_steps, network, network_queues)


def update_weights(batch, network, optimizer, device):
    network.train()

    p_loss, v_loss = 0, 0

    for image, actions, targets in batch:
        # Initial step, from the real observation.
        value, reward, policy_logits, hidden_state = network.initial_inference(image)
        predictions = [(1.0, value, reward, policy_logits)]

        # Recurrent steps, from action and previous hidden state.
        for action in actions:
            value, reward, policy_logits, hidden_state = network.recurrent_inference(hidden_state, action)
            predictions.append((1.0 / len(actions), value, reward, policy_logits))

        for prediction, target in zip(predictions, targets):
            if len(target[2]) > 0:
                _, value, reward, policy_logits = prediction
                target_value, target_reward, target_policy = target

                p_loss += torch.sum(-torch.Tensor(numpy.array(target_policy)).to(device) * torch.log(policy_logits))
                v_loss += torch.sum((torch.Tensor([target_value]).to(device) - value) ** 2)

    optimizer.zero_grad()
    total_loss = (p_loss + v_loss)
    total_loss.backward()
    optimizer.step()
    network.steps += 1

    return p_loss / len(batch), v_loss / len(batch)


# End Training
##################################

################################################################################
# End of pseudocode
################################################################################

# Stubs to make the typechecker happy.
def softmax_sample(distribution, temperature: float):
    if temperature == 0:
        temperature = 1
    distribution = numpy.array(distribution) ** (1 / temperature)
#   https://github.com/Zeta36/muzero/issues/5
#   p_sum = distribution.sum()
    p_sum = distribution[:, 0].sum()
#   sample_temp = distribution / p_sum
    sample_temp = distribution[:, 0] / p_sum
#   return 0, numpy.argmax(numpy.random.multinomial(1, sample_temp, 1))
    action = distribution[int(numpy.argmax(numpy.random.multinomial(1, sample_temp, 1)))][1]
    return 0, int(action)


def launch_job(f, *args):
    f(*args)


def make_connect4_config() -> MuZeroConfig:
    return make_board_game_config(
        action_space_size=7,
        max_moves=20,
        dirichlet_alpha=0.03,
        lr_init=0.01,
        game_class=Connect4Game,
        width=7,
        height=6
    )


def make_dots_config() -> MuZeroConfig:
    return make_board_game_config(
        action_space_size=64,
        max_moves=50,
        dirichlet_alpha=0.03,
        lr_init=0.001,
        game_class=DotsGame,
        width=8,
        height=8
    )


def main():
    if torch.cuda.is_available():
        multiprocessing.set_start_method('spawn')
        device = 'cuda:0'
    else:
        device = 'cpu'
    print(device)

    # config = make_connect4_config()
    config = make_dots_config()
    vs_random_once = random_vs_random(config)
    print('random_vs_random = ', sorted(vs_random_once.items()), end='\n')
    network = muzero(config, device)
    print(network)


if __name__ == '__main__':
    main()
