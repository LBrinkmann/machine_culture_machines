
import numpy as np
from .model import calculate_reward_transition_matrices
import torch
from scipy.special import softmax


def calculate_q_matrix_avpruning(R, T, n_steps, G, device=torch.device('cpu')):
    r"""Calculates the Q matrix under aversive pruning.

    The Q matrix indicates the expected reward (under aversive pruning) 
    from a given node, at a given step for a given action. This method
    is calculating simultaneously the Q matrix for multiple networks and
    participants.

    Args:
        T: The transition matrix. A 4-D `Tensor` of type `torch.float32` and shape
            `[environment, source, target, action]`. A unit value indicates a possible
            transition from a source node to a target node.
        R: The reward matrix. A 3-D `Tensor` of type `torch.float32` and shape
            `[environment, source, action]`. The reward for a given action.
        n_steps: A integer. The number of steps to plan ahead. 
        G: The pruning factor. A 2-D `Tensor` of type `torch.float32` and shape
            `[participant, gamma]`. Each participants has two pruning factors.
            The first value in the second dimension is the general pruning factor, 
            the second value  is the aversive pruning factor. 
        device: A torch device.

    Returns:
        Q matrix
            A 5-D `Tensor` of type `torch.float32` and shape
            `[participant, environment, step, node, action]`
    """

    n_envs = T.shape[0]
    n_nodes = T.shape[1]
    n_participants = G.shape[0]

    R_av = (R <= -100).type(torch.float)  # one if aversive
    Q = torch.zeros((n_participants, n_envs, n_steps, n_nodes, 2), device=device)  # remaining, original node, action
    Q[:, :, n_steps-1] = R
    G_masked = (R_av[np.newaxis, :, :, :] * (1 - G[:, 1, np.newaxis, np.newaxis, np.newaxis]) +
                (1 - R_av[np.newaxis, :, :, :]) * (1-G[:, 0, np.newaxis, np.newaxis, np.newaxis]))
    for k in range(n_steps-1, 0, -1):  # stay
        # Q value of all reachable states (with a single action) from a given state
        Q1 = torch.einsum('lkjb,kija->lkiab', Q[:, :, k], T)
        # for each reachable state, the largest Q value
        Q2 = torch.max(Q1, axis=-1)[0]
        Q[:, :, k-1, :, :] = R[np.newaxis, :, :, :] + G_masked * Q2

    return Q


def calc_T_R(networks, n_nodes=None):
    if n_nodes is None:
        n_nodes = len(networks[0]['nodes'])
    T_list = []
    R_list = []
    for network in networks:
        _T, _R = calculate_reward_transition_matrices(network, n_nodes)
        T_list.append(_T)
        R_list.append(_R)
    T = np.stack(T_list)
    R = np.stack(R_list)
    return T, R


def calculate_traces_stochastic(Q, T, R, starting_node):
    n_steps = Q.shape[0]  # Q: step,node,action
    NT = np.zeros((n_steps + 1), dtype='int64')  # node trace: step, starting node
    AT = np.zeros((n_steps), dtype='int64')  # action trace: step, starting node
    RT = np.zeros((n_steps), dtype='int64')  # reward trace: step, starting node
    NT[0] = starting_node
    PT = softmax(Q, axis=-1)
    for l in range(0, n_steps):
        AT[l] = np.random.choice(a=[0, 1], size=1, p=PT[l, NT[l]])
        NT[l+1] = np.argmax(T[NT[l], :, AT[l]], axis=0)
        RT[l] = R[NT[l], AT[l]]
    RT_tot = np.sum(RT, axis=0)
    return RT_tot, AT, NT, RT


# TODO: rename
def calculate_traces(Q, T, R, ST=None, beta=None, device=torch.device('cpu')):
    r"""Calculates the expected reward.

    This method takes a Q matrix and a temperature `beta` and
    calculates the extected reward for a set of environments.

    Args:
        Q: The Q matrix. A 5-D `Tensor` of type `torch.float32` and shape
            `[participant, environment, step, node, action]`. The expected
            reward of a given action.
        T: The transition matrix. A 4-D `Tensor` of type `torch.float32` and shape
            `[environment, source, target, action]`. A unit value indicates a possible
            transition from a source node to a target node.
        R: The reward matrix. A 3-D `Tensor` of type `torch.float32` and shape
            `[environment, source, action]`. The reward for a given action.
        ST: A 1-D `Tensor` of type `torch.int` and size `environment`.
            Starting nodes to be used for each environment. If None, reward is calculated
            for all starting nodes.
        beta: A float or None. The inverse temperature. Greedy, if None. 
        device: A torch device.

    Returns:
        Expected reward 
            A 3-D `Tensor` of type `torch.float32` and shape
            `[participant, environment, startingNode]` or `[participant, environment]` 
            if ST is defined. 
    """
    n_user = Q.shape[0]
    n_envs = Q.shape[1]
    n_steps = Q.shape[2]
    n_nodes = Q.shape[3]

    if beta is None:
        P = torch.zeros_like(Q, device=device)
        P = torch.where(Q[:, :, :, :, [0]] == Q[:, :, :, :, [1]], torch.tensor(0.5, device=device), P)
        P = torch.where(Q[:, :, :, :, [0]] > Q[:, :, :, :, [1]], torch.tensor([[[[[1., 0.]]]]], device=device), P)
        P = torch.where(Q[:, :, :, :, [0]] < Q[:, :, :, :, [1]], torch.tensor([[[[[0., 1.]]]]], device=device), P)
    else:
        P = torch.nn.functional.softmax(
            Q*beta[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis], dim=-1)

    if ST is None:
        # user, environment, starting node, current node
        p_source = torch.zeros((n_user, n_envs, n_nodes, n_nodes), device=device)
        for i in range(n_nodes):
            p_source[:, :, i, i] = 1
        # user, environment, starting node
        e_reward = torch.zeros((n_user, n_envs, n_nodes), device=device)
    else:
        p_source = torch.zeros((1, n_envs, 1, n_nodes), device=device)
        p_source[0, np.arange(n_envs),0 , ST] = 1
        p_source = p_source.repeat(n_user, 1, 1, 1)

        e_reward = torch.zeros((n_user, n_envs, 1), device=device)
    # print(n_envs, n_steps, n_nodes, p_source.sum())

    for step in range(0, n_steps):
        # P(source, action | step) = P(source | step) * P(action | source, step)
        # u: user, f: starting node, e: env., s: source, a: action
        p_source_action = torch.einsum('uefs,uesa->uefsa', p_source, P[:, :, step])

        # P(source | step + 1) = P(target | step) = P(source, action | step) * T(source, target, action)
        p_source = torch.einsum('uefsa,esta->ueft', p_source_action, T)  # e: env., s: source, a: action, t: target

        # E(reward | step) = P(action, source | step) * R(source, action)
        e_reward += torch.einsum('uefsa,esa->uef', p_source_action, R)

    if ST is None:
        return e_reward
    else:
        return e_reward[:,:,0]


# legacy


def calculate_traces_meanfield(Q, T, R, ST, device=torch.device('cpu')):
    beta = torch.ones(1, device=device)
    return calculate_traces(Q, T, R, ST=ST, beta=beta, device=device)


# testing


def G_fake(n_participants, seed=None):
    np.random.seed(seed)
    gamma_g = np.random.uniform(low=0.1, high=0.3, size=n_participants)
    gamma_s = np.random.uniform(low=0.3, high=0.5, size=n_participants)
    G = np.column_stack((gamma_g, gamma_s))
    return G


def calc_S_fake(networks, n_nodes, n_participants, n_steps, seed=None):
    T, R = calc_T_R(networks, n_nodes)
    G = G_fake(n_participants, seed)
    Q = calc_all_Q(T, R, G, n_steps)
    Q = Q + np.random.normal(100, 30, Q.shape)
    S_fake = calc_Q_correct(Q)
    return G, S_fake


# archive


def calc_all_Q(T, R, G, n_steps):
    T = torch.tensor(T, dtype=torch.float32)
    R = torch.tensor(R, dtype=torch.float32)
    G = torch.tensor(G, dtype=torch.float32)
    Q = calculate_q_matrix_avpruning(R, T, n_steps, G)
    return Q.numpy()


def calc_Q_correct(Q):
    Q_corr = (Q.max(axis=-1, keepdims=1) == Q).astype(int)
    return Q_corr


def calc_Q_correct_torch(Q):
    Q_corr = torch.eq(torch.max(Q, dim=-1, keepdim=True)[0], Q).type(torch.float32)
    return Q_corr
