import json

def save_weights(weights, filename="weights.json"):
    with open(filename, "w") as f:
        json.dump(weights, f)

def load_weights(filename="weights.json"):
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
