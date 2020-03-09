from flask import Flask, request, jsonify
import json
import traceback
from mc.reward_networks import optimisation2 as opt2
from mc.reward_networks import optimisation3 as opt3
from mc.reward_networks.parser import parse_actions

from machine_backend.utils import Better_JSON_ENCODER, camelize_dict_keys, snakeize_dict_keys

import numpy as np


# credit: https://github.com/miguelgrinberg/Flask-SocketIO/issues/274


class BetterJsonWrapper(object):
    @staticmethod
    def dumps(*args, **kwargs):
        if 'cls' not in kwargs:
            kwargs['cls'] = Better_JSON_ENCODER
        return json.dumps(*args, **kwargs)

    @staticmethod
    def loads(*args, **kwargs):
        return json.loads(*args, **kwargs)


app = Flask(__name__)
app.json_encoder = Better_JSON_ENCODER

models = [
    {
        'name': 'AdvPruning',
        'parameter': {
            'gamma_s': 0.4,
            'gamma_g': 0.2,
            'beta': 0.01},
        'type': 'pruning'
    },
    {
        'name': 'Lookahead',
        'parameter': {
            'gamma_s': 0.0,
            'gamma_g': 0.0,
            'beta': 1},
        'type': 'pruning'
    },
    {
        'name': 'TakeWorst',  # wrong name, for debug only
        'parameter': {
            'gamma_s': 0.4,
            'gamma_g': 0.2,
            'beta': 0.01},
        'type': 'pruning'
    }
]


model_dict = {m['name']: m for m in models}


def calculate_reward_transition_matrices(actions, n_nodes):
    """
        Calculate the reward and transition matrices.
        R (original node:action): reward on transition from original node by doing the action
        T (original node:desitination node:action): one if action from original node leads to destination node, 0 otherwise

    """
    T = np.zeros((n_nodes, n_nodes, 2))  # original node, destination node, action
    R = np.zeros((n_nodes, 2))  # original node, action

    # counter of actions from each source
    n_source_actions = {i: 0 for i in range(n_nodes)}
    for a in actions:
        actionIdx = n_source_actions[a['source_id']]
        n_source_actions[a['source_id']] += 1
        R[a['source_id'], actionIdx] = a['reward']
        T[a['source_id'], a['target_id'], actionIdx] = 1
    return T, R


def calculate_q_matrix(R, T, n_steps, beta, gamma_s, gamma_g):
    G = np.array([[gamma_s, gamma_g]])
    Q = opt2.calculate_q_matrix_avpruning(R[np.newaxis], T[np.newaxis], n_steps, G)[0, 0]
    print(Q.shape)
    return beta * Q


def create_action_trace(environment, model_type, model_parameter):
    T, R = calculate_reward_transition_matrices(
        environment['actions'], n_nodes=len(environment['nodes']))
    n_steps = environment["required_solution_length"]
    starting_node = environment["starting_node_id"]
    if model_type == 'pruning':
        Q_temp = calculate_q_matrix(R, T, n_steps, **model_parameter)
        RT_tot, AT, NT, RT = opt3.calculate_traces_stochastic(Q_temp, T, R, starting_node)
    else:
        raise NotImplementedError('Model type is not implemented.')

    action_trace = parse_actions(NT, RT)

    return action_trace, RT_tot


def _machine_solution(request):
    request_snake = snakeize_dict_keys(request)

    environment = request_snake['data']['environment']
    model_name = request_snake['data']['model_name']
    model = model_dict[model_name]

    actions, total_reward = create_action_trace(
        environment=environment, model_type=model['type'], model_parameter=model['parameter'])

    prev_solution = request_snake['data']['previous_solution']

    if prev_solution is not None and prev_solution['total_reward'] > total_reward:
        new_actions = prev_solution['actions']
        new_total_reward = prev_solution['total_reward']
        solution_type = 'COPY'
    else:
        new_actions = actions
        new_total_reward = total_reward
        solution_type = 'NEW'

    data = {
        'actions': new_actions,
        "environmentId": environment['environment_id'],
        "networkId": environment['network_id'],
        "modelName": model_name,
        "totalReward": new_total_reward,
        "solutionType": solution_type
    }

    response = {'requestId': request_snake['request_id'], 'data': data}
    return response


def test():
    with open('./machine_backend/test_request.json', 'r') as f:
        request = json.load(f)

    import time

    t0 = time.time()
    response = _machine_solution(request)
    t1 = time.time()
    print(response)
    print(f'Request took {(t1-t0)*1000} ms')


@app.route('/', methods=['POST'])
def machine_solution():
    return _machine_solution(request.get_json())


@app.errorhandler(Exception)
def handle_error(e):
    error = getattr(e, "original_exception", None)
    message = [str(x) for x in error.args]
    status_code = 500
    success = False
    response = {
        'success': success,
        'error': {
            'type': error.__class__.__name__,
            'message': message,
            'stacktrace': traceback.format_exc()
        }
    }

    return jsonify(response), status_code


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000)
