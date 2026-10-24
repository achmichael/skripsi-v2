import numpy as np

class ReLU:
    def __init__(self):
        self.input = None

    def forward(self, x):
        self.input = x

        return np.maximum(0, x)

    def backward(self, grad_output):
        if self.input is None:
            raise RuntimeError(
                "Forward harus dipanggil sebelum backward()"
            )

        grad_input = grad_output.copy()

        grad_input[self.input <= 0] = 0

        return grad_input


class Linear:
    def __init__(self):
        pass

    def forward(self, x):
        return x

    def backward(self, grad_output):
        return grad_output
