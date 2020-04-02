from flask import Flask, request
import json
import yaml

from mc.reward_networks.models import torch_model as tm
from mc.reward_networks.utils.parser import parse_actions

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


def load_yaml(file):
    with open(file) as f:
        return yaml.safe_load(f)


def store_yaml(obj, file):
    with open(file, 'w') as f:
        return yaml.dump(obj, f)

models = load_yaml('./models.yaml')


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
    Q = tm.calc_all_Q(T[np.newaxis], R[np.newaxis], G, n_steps)[0, 0]
    if beta is not None:
        Q_temp = beta * Q
    else:
        Q_temp = Q
    return Q_temp


def create_action_trace(environment, model_type, model_parameter):
    n_nodes = len(environment['nodes'])
    T, R = calculate_reward_transition_matrices(
        environment['actions'], n_nodes=n_nodes)
    n_steps = environment["required_solution_length"]
    starting_node = environment["starting_node_id"]
    if model_type == 'pruning':
        Q_temp = calculate_q_matrix(R, T, n_steps, **model_parameter)
        RT_tot, AT, NT, RT = tm.calculate_traces_stochastic(Q_temp, T, R, starting_node)
    elif model_type == 'random':
        Q_random = np.zeros((n_nodes*n_steps, 2))
        Q_random_idx = np.random.choice(2, n_nodes*n_steps)
        Q_random[np.arange(n_nodes*n_steps), Q_random_idx] = 1
        Q_random = Q_random.reshape(n_steps, n_nodes, 2)
        RT_tot, AT, NT, RT = tm.calculate_traces_stochastic(Q_random, T, R, starting_node)
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
        environment=environment, model_type=model['type'], 
        model_parameter=model.get('parameter', {}))
    previous_solution = request_snake['data']['previous_solution']
    if (previous_solution is not None) and (previous_solution['total_reward'] > total_reward):
        new_actions = previous_solution['actions']
        new_total_reward = previous_solution['total_reward']
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

    print('request.get_json', request.get_json())
    return _machine_solution(request.get_json())


@app.route('/config', methods=['PUT'])
def put_config():
    data = request.get_data()
    print(data)
    models = yaml.safe_load(data)
    store_yaml(models, './models.yaml')
    global model_dict
    model_dict = {m['name']: m for m in models}
    return (yaml.dump(models), 200)


@app.route('/config', methods=['GET'])
def get_config():
    models = load_yaml('./models.yaml')
    return (yaml.dump(models), 200)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=8085)
