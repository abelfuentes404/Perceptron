import json
import numpy as np

def save_weights(weights, filename):
    with open(filename, "w") as f:
        json.dump(weights, f)

def load_weights(filename):
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def ensure_python_types(data):
    if isinstance(data, (np.integer, np.floating)):
        return int(data) if isinstance(data, np.integer) else float(data)
    elif isinstance(data, np.bool_):
        return bool(data)
    elif isinstance(data, (list, tuple)):
        return [ensure_python_types(x) for x in data]
    elif isinstance(data, dict):
        return {k: ensure_python_types(v) for k, v in data.items()}
    return data