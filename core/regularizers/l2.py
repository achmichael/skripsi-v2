import numpy as np


class L2Regularizer:
    """
    L2 Regularization (Weight Decay).

    Menambahkan penalty term ke loss function:
    L_total = L_original + (lambda/2) * sum(W^2)

    Gradient tambahan untuk weight:
    dW_regularized = dW_original + lambda * W
    """

    def __init__(self, lambda_: float = 0.01):
        """
        Args:
            lambda_: Regularization strength (default: 0.01)
        """
        if lambda_ < 0:
            raise ValueError("Lambda harus non-negatif")

        self.lambda_ = lambda_

    def compute_loss(self, weights: list, inputs) -> float:
        if self.lambda_ == 0:
            return 0.0

        total = 0.0
        for W in weights:
            if W is not None:
                total += np.sum(W ** 2)

        m = len(inputs)
        return (self.lambda_ / (2.0 * m)) * total

    def compute_gradient(self, weight: np.ndarray) -> np.ndarray:
        if self.lambda_ == 0 or weight is None:
            return np.zeros_like(weight)

        return self.lambda_ * weight

    def apply_to_gradients(self, layers: list, inputs):
        if self.lambda_ == 0:
            return

        for layer in layers:
            if hasattr(layer, 'weights') and hasattr(layer, 'dweights'):
                if layer.weights is not None and layer.dweights is not None:
                    m = len(inputs)
                    layer.dweights += (self.lambda_ / m) * layer.weights
