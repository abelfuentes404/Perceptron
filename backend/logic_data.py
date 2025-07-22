import numpy as np

def get_and_data():
    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    y = np.array([0, 0, 0, 1])
    return X, y

def get_or_data():
    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    y = np.array([0, 1, 1, 1])
    return X, y

def get_logic_data(logic_type):
    if logic_type == "and":
        return get_and_data()
    elif logic_type == "or":
        return get_or_data()
    else:
        raise ValueError("Tipo de lógica no soportada")