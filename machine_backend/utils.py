import re
import json
import numpy as np
import datetime


def camelize(snake_str):
    components = snake_str.split('_')
    return components[0] + ''.join(x.capitalize() for x in components[1:])


def snakeize(camel_str):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', camel_str)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def transform_dict_keys(d, keyer):
    if isinstance(d, dict):
        nd = {}
        for key, value in d.items():
            transformed_key = keyer(key)
            nd[transformed_key] = transform_dict_keys(value, keyer)
    elif isinstance(d, list):
        nd = [transform_dict_keys(el, keyer) for el in d]
    else:
        nd = d
    return nd


def camelize_dict_keys(d):
    return transform_dict_keys(d, keyer=camelize)


def snakeize_dict_keys(d):
    return transform_dict_keys(d, keyer=snakeize)


# credit: https://stackoverflow.com/questions/44146087/pass-user-built-json-encoder-into-flasks-jsonify
class Better_JSON_ENCODER(json.JSONEncoder):
    '''
    Used to help jsonify numpy arrays or lists that contain numpy data types.
    '''

    def default(self, obj):
        # print(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (datetime.datetime)):
            if obj.tzinfo is None:
                return obj.isoformat() + 'Z'
            else:
                return obj.isoformat()
        elif isinstance(obj, (datetime.datetime, datetime.date)):
            # print('datetime', obj)
            return obj.isoformat()
        else:
            return super(Better_JSON_ENCODER, self).default(obj)
