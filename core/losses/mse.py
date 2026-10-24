import numpy as np

class MSE:
    def forward(self, y_pred, y_actual):
        return np.mean((y_pred - y_actual) ** 2)

    def backward(self, y_pred, y_actual):
        n = y_actual.shape[0]

        return (2.0/n) * (y_pred - y_actual)
