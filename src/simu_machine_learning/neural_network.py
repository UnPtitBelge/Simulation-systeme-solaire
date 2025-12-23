import numpy as np


class Neural_network:
    def __init__(self, input_dim, hidden_dim, learning_rate) -> None:
        self.mu = learning_rate

        # Couche d'entrée de dimension input_dim x hidden_dim
        self.W1 = np.random.randn(hidden_dim, input_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)

        # 2ème couche cachée de dimension hidden_dim x hidden_dim
        self.W2 = np.random.randn(hidden_dim, hidden_dim) * 0.1
        self.b2 = np.zeros(hidden_dim)

        # Couche de sortie de dimension hidden_dim x output_dim
        self.W3 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b3 = np.zeros(input_dim)

    def MSE(self, x: np.ndarray, y: np.ndarray) -> np.float32:
        return np.mean((x - y) ** 2)

    def relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def relu_derivative(self, x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x_1 = self.W1 @ x + self.b1
        self.x_hat = self.relu(self.x_1)

        self.x_2 = self.W2 @ self.x_hat + self.b2
        self.x_hat2 = self.relu(self.x_2)

        self.output = self.W3 @ self.x_hat2 + self.b3
        return self.output

    def backward(self, x: np.ndarray, y: np.ndarray) -> None:
        # Calcul du gradient de la perte par rapport à la sortie
        dL_output = self.output - y  # Δx, Δy

        dW3 = np.outer(dL_output, self.x_hat2)
        db3 = dL_output
        d_hidden2 = self.W3.T @ dL_output

        d_hidden2 = d_hidden2 * self.relu_derivative(self.x_2)
        dW2 = np.outer(d_hidden2, self.x_hat)
        db2 = d_hidden2
        d_hidden1 = self.W2.T @ d_hidden2

        d_hidden1 = d_hidden1 * self.relu_derivative(self.x_1)
        dW1 = np.outer(d_hidden1, x)
        db1 = d_hidden1

        self.W3 -= self.mu * dW3
        self.b3 -= self.mu * db3
        self.W2 -= self.mu * dW2
        self.b2 -= self.mu * db2
        self.W1 -= self.mu * dW1
        self.b1 -= self.mu * db1
