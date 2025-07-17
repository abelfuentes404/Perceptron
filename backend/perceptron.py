import numpy as np

class SimplePerceptron:
    def __init__(self, input_size, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.weights = np.random.uniform(-1, 1, input_size + 1)  # +1 for bias
        self.errors = []

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def predict(self, x):
        x = np.insert(x, 0, 1)
        return self.sigmoid(np.dot(x, self.weights))

    def train(self, X, y, epochs=20):
        self.errors.clear()
        for _ in range(epochs):
            total_error = 0
            for xi, target in zip(X, y):
                xi = np.insert(xi, 0, 1)
                z = np.dot(xi, self.weights)
                y_hat = self.sigmoid(z)
                error = target - y_hat
                total_error += error ** 2
                gradient = error * y_hat * (1 - y_hat)
                self.weights += self.learning_rate * gradient * xi
            self.errors.append(total_error)

    def get_weights(self):
        return self.weights.tolist()

    def set_weights(self, weights_list):
        self.weights = np.array(weights_list)
