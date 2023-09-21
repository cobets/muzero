# Lint as: python3
"""Pseudocode description of the MuZero algorithm."""
# pylint: disable=unused-argument
# pylint: disable=missing-docstring
# pylint: disable=assignment-from-no-return

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import signal
from typing import List

import numpy
import torch
import torch.optim as optim
import threading

from action import Action
from action_history import ActionHistory
from environment import Winner, Player
from game import Game
from min_max_stats import MinMaxStats
from mu_zero_config import MuZeroConfig, make_connect4_config
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
def vs_random(network, config, n=100):
    results = {}
    for i in range(n):
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
        if ((game.environment.winner == Winner.white and first_turn)
                or (game.environment.winner == Winner.black and not first_turn)):
            r = 1
        elif ((game.environment.winner == Winner.black and first_turn)
              or (game.environment.winner == Winner.white and not first_turn)):
            r = -1
        results[r] = results.get(r, 0) + 1
    return results


def random_vs_random(config, n=100):
    results = {}
    for i in range(n):
        first_turn = i % 2 == 0
        turn = first_turn
        game = config.new_game()
        r = 0
        while not game.terminal():
            action = numpy.random.choice(game.legal_actions())
            game.apply(action)
            turn = not turn
        if ((game.environment.winner == Winner.white and first_turn)
                or (game.environment.winner == Winner.black and not first_turn)):
            r = 1
        elif ((game.environment.winner == Winner.black and first_turn)
              or (game.environment.winner == Winner.white and not first_turn)):
            r = -1
        results[r] = results.get(r, 0) + 1
    return results


def latest_vs_older(last, old, config, n=100):
    results = {}
    for i in range(n):
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
        if ((game.environment.winner == Winner.white and first_turn)
                or (game.environment.winner == Winner.black and not first_turn)):
            r = 1
        elif ((game.environment.winner == Winner.black and first_turn)
              or (game.environment.winner == Winner.white and not first_turn)):
            r = -1
        results[r] = results.get(r, 0) + 1
    return results


# MuZero training is split into two independent parts: Network training and
# self-play data generation.
# These two parts only communicate by transferring the latest network checkpoint
# from the training to the self-play, and the finished games from the self-play
# to the training.
def muzero(config: MuZeroConfig, device):
    storage = SharedStorage(device)
    replay_buffer = ReplayBuffer(config)
    # stop event
    stop_event = threading.Event()
    signal.signal(signal.SIGINT, lambda sig, frame: stop_event.set())
    # Start n concurrent actor threads
    threads = list()
    for _ in range(config.num_actors):
        t = threading.Thread(target=launch_job, args=(run_selfplay, config, storage, replay_buffer, stop_event))
        threads.append(t)

    # Start all threads
    for x in threads:
        x.start()

    train_network(config, storage, replay_buffer, device, stop_event)

    return storage.latest_network()


##################################
# Part 1: Self-Play


# Each self-play job is independent of all others; it takes the latest network
# snapshot, produces a game and makes it available to the training job by
# writing it to a shared replay buffer.
def run_selfplay(config: MuZeroConfig, storage: SharedStorage, replay_buffer: ReplayBuffer, stop_event):
    while True:
        if stop_event.is_set():
            break
        network = storage.latest_network()
        game = play_game(config, network)
        replay_buffer.save_game(game)


# Each game is produced by starting at the initial board position, then
# repeatedly executing a Monte Carlo Tree Search to generate moves until the end
# of the game is reached.
def play_game(config: MuZeroConfig, network: Network) -> Game:
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

        backpropagate(search_path, network_output.value, history.to_play(),
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

def train_network(config: MuZeroConfig, storage: SharedStorage,
                  replay_buffer: ReplayBuffer, device, stop_event):
    network = Network(config.action_space_size, device).to(device)

    while True:
        if stop_event.is_set():
            break

        optimizer = optim.SGD(network.parameters(), lr=0.01, weight_decay=config.lr_decay_rate,
                              momentum=config.momentum)

        while not len(replay_buffer.buffer) > 0:
            if stop_event.is_set():
                break

        for i in range(config.training_steps):
            if stop_event.is_set():
                break

            if i % config.checkpoint_interval == 0 and i > 0:
                storage.save_network(i, network)
                # Test against random agent
                vs_random_once = vs_random(network, config)
                print('network_vs_random = ', sorted(vs_random_once.items()), end='\n')
                vs_older = latest_vs_older(storage.latest_network(), storage.old_network(), config)
                print('lastnet_vs_older = ', sorted(vs_older.items()), end='\n')
                # print(hp.heap())

            batch = replay_buffer.sample_batch(config.num_unroll_steps, config.td_steps)
            update_weights(batch, network, optimizer, device)

        if not stop_event.is_set():
            storage.save_network(config.training_steps, network)


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
    print('p_loss %f v_loss %f' % (p_loss / len(batch), v_loss / len(batch)))


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


def main():
    config = make_connect4_config()
    vs_random_once = random_vs_random(config)
    print('random_vs_random = ', sorted(vs_random_once.items()), end='\n')
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(device)
    network = muzero(config, device)
    print(network)


if __name__ == '__main__':
    main()
