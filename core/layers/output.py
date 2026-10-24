import numpy as np

class Output:
    def __init__(self, input_size):
        self.input_size = input_size

        self.weights = np.random.randn(input_size, 1) * np.sqrt(2.0 / input_size)

        self.input = None
        self.dbiases = None

    def forward(self, x):
        self.input = x

        return x @ self.weights

    def backward(self, grad_output):
        self.dweights = self.input.T @ grad_output
        grad_input = grad_output @ self.weights.T
        return grad_input
