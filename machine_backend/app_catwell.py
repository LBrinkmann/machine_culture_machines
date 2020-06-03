from flask import Flask, request, jsonify, send_file
import json
from io import BytesIO
import traceback
from PIL import Image
from machine_backend.catwell.non_discrete_landscape import gen_cubic_noise_functional

from machine_backend.utils import Better_JSON_ENCODER, camelize_dict_keys, snakeize_dict_keys

import numpy as np


app = Flask(__name__)
app.json_encoder = Better_JSON_ENCODER


experiments = {
    'test': {
        'w_min': 3,
        'w_tot': 3,
        'persistence': 0.7
    }
}

def _action_eval(environment, action):
    func = gen_cubic_noise_functional(
        **experiments[environment['experiment_name']], seed=environment['seed'])
    reward = func(action['x'], action['y'])

    reward = (reward + 0.2) * 50

    return {**action, 'reward': reward}


def serve_pil_image(pil_img):
    img_io = BytesIO()
    pil_img.save(img_io, 'JPEG', quality=70)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpeg')


def _get_image(size, **kwargs):
    func = gen_cubic_noise_functional(**kwargs)
    net_points = np.linspace(0,1,size)
    x, y = np.meshgrid(net_points, net_points)
    x, y = x.reshape(-1), y.reshape(-1)
    landscape = func(x, y).reshape(size, size)
    landscape = (((landscape - landscape.min()) / (landscape.max() - landscape.min())) * 255).astype(np.uint8)


    img = Image.fromarray(landscape, 'L')
    return serve_pil_image(img)

def random_actions(environment):
    step_size=4
    data=[]
    x_array = np.random.random(step_size)
    y_array = np.random.random(step_size)
    for i in range(step_size):
        action = {'x': x_array[i], 'y': y_array[i],'step':i+1}
        data.append( _action_eval(environment,action))
    return data

@app.route('/eval', methods=['POST'])
def action_eval():
    request_snake = snakeize_dict_keys(request.get_json())
    data = _action_eval(**request_snake['data'])
    resp = {'data': data, 'request_id': request_snake['request_id']}
    return camelize_dict_keys(resp)


@app.route('/image', methods=['GET'])
def get_image():
    exp_name = request.args.get('experiment-name')
    size = int(request.args.get('size'))
    seed = int(request.args.get('seed'))
    return _get_image(**experiments[exp_name], seed=seed, size=size)

@app.route('/play', methods=['POST'])
def action_play():
    request_snake = snakeize_dict_keys(request.get_json())
    data ={'solution': random_actions(request_snake['data']['environment'])}
    resp = {'data': data,'request_id': request_snake['request_id']}
    return camelize_dict_keys(resp)

# @app.route('/config', methods=['PUT'])
# def put_config():
#     data = request.get_data()
#     print(data)
#     models = yaml.safe_load(data)
#     store_yaml(models, './models.yaml')
#     global model_dict
#     model_dict = {m['name']: m for m in models}
#     return (yaml.dump(models), 200)


# @app.route('/config', methods=['GET'])
# def get_config():
#     models = load_yaml('./models.yaml')
#     return (yaml.dump(models), 200)

@app.errorhandler(Exception)
def handle_error(error):
    # import ipdb; ipdb.set_trace()
    # error = getattr(e, "original_exception", None)
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
    print(message)
    print(traceback.format_exc())
    return jsonify(response), status_code


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=8090)
