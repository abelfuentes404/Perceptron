from perceptron import SimplePerceptron
from logic_data import get_and_data, get_or_data
from utils import save_weights

def train_model(logic_type="and", epochs=20):
    X, y = get_and_data() if logic_type == "and" else get_or_data()
    model = SimplePerceptron(input_size=2)
    model.train(X, y, epochs)
    save_weights(model.get_weights())
    return model
