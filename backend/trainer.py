from perceptron import SimplePerceptron
from logic_data import get_logic_data
from utils import save_weights

def train_logic_model(logic_type="and", epochs=20):
    X, y = get_logic_data(logic_type)
    
    model = SimplePerceptron(input_size=2)
    model.train(X, y, epochs)
    save_weights(model.get_weights(), "logic_weights.json")
    
    truth_table = []
    for xi, expected in zip(X, y):
        prediction = model.predict(xi)
        truth_table.append({
            "input": xi.tolist(),
            "expected": int(expected),
            "predicted": round(float(prediction), 4)
        })
    
    return model, {
        "message": f"Modelo {logic_type.upper()} entrenado",
        "errors": model.errors,
        "weights": model.get_weights(),
        "truth_table": truth_table
    }