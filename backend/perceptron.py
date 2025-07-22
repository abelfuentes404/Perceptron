import numpy as np

class SimplePerceptron:
    def __init__(self, input_size, learning_rate=0.1):
        self.learning_rate = learning_rate
        # Pesos iniciales más conservadores
        self.weights = np.array([-0.5, 0.7, 0.2])  # [bias, loan_weight, income_weight]
        self.errors = []
    
    def predict(self, x):
        """Predicción con función escalón (mejor para lógica binaria)"""
        x_with_bias = np.insert(x, 0, 1)  # Add bias term
        weighted_sum = np.dot(x_with_bias, self.weights)
        return 1 if weighted_sum > 0 else 0
    
    def train(self, X, y, epochs=20):
        """Entrenamiento con regla de perceptrón estándar"""
        self.errors = []
        for _ in range(epochs):
            total_error = 0
            for xi, target in zip(X, y):
                xi_with_bias = np.insert(xi, 0, 1)
                prediction = self.predict(xi)
                error = target - prediction
                total_error += abs(error)
                # Actualizar pesos
                self.weights += self.learning_rate * error * xi_with_bias
            self.errors.append(total_error)
    
    def get_weights(self):
        return self.weights.tolist()
    
    def set_weights(self, weights_list):
        self.weights = np.array(weights_list)