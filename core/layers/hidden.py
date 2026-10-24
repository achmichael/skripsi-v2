import numpy as np

class Hidden:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)

        self.biases = np.zeros((1, output_size))

        self.input = None
        # delta weights and biases
        self.dweights = None
        self.dbiases = None

    def forward(self, x):
        self.input = x

        return x @ self.weights + self.biases


    def backward(self, grad_output):
        self.dweights = self.input.T @ grad_output
        self.dbiases = np.sum(grad_output, axis=0, keepdims=True)

        grad_input = grad_output @ self.weights.T

        # print('grad_input', grad_input)

        return grad_input
