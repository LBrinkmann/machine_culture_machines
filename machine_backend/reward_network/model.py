"""
Terminology:
    node: nodes of the network
    action: possible actions from each node (typically 2)
    reward: each action from each node has a related reward
    transition: (node, action) tuple 
    remaining: the number of remaining actions left before the game ends
    state: (node, remaining) tuple 
"""

import numpy as np
import pandas as pd
import hashlib
import uuid


def calculate_reward_transition_matrices(network, n_nodes):
    """
        Calculate the reward and transition matrices.
        R (original node:action): reward on transition from original node by doing the action
        T (original node:desitination node:action): one if action from original node leads to destination node, 0 otherwise

    """
    T = np.zeros((n_nodes, n_nodes, 2))  # original node, destination node, action
    R = np.zeros((n_nodes, 2))  # original node, action
    for j in range(len(network['links'])):
        link = network['links'][j]
        origin = link['source']
        destination = link['target']
        action = 0 if link['action'] == 'L' else 1
        reward = link['weight']
        R[origin-1, action] = reward
        T[origin-1, destination-1, action] = 1
    return T, R
