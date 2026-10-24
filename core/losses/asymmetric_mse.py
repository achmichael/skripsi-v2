from matplotlib.pylab import beta
import numpy as np

class AsymmetricMSE:
    def __init__(self, beta, alpha=0.6):
        if not 0 <= alpha <= 1:
            raise ValueError("alpha harus berada di antara 0 dan 1")

        self.alpha = alpha
        self.beta = beta

    def forward(self, y_pred, y_actual):
        diff = y_actual - y_pred

        # Mask untuk underestimation (prediction < actual)
        underestimation_mask = (diff >= 0).astype(float)
        # Mask untuk overestimation (prediction >= actual)
        overestimation_mask = (diff < 0).astype(float)

        # Asymmetric weight
        weights = self.alpha * underestimation_mask + self.beta * overestimation_mask

        # Weighted squared error
        weighted_squared_errors = weights * (diff ** 2)

        return np.mean(weighted_squared_errors)

    def backward(self, y_pred, y_actual):
        n = y_actual.shape[0]
        #  Definisi diff = y_actual - y_pred ini berarti diff > 0 (underestimation) diff < 0 (overestimation) diff = 0 (estimasi tepat)
        diff = y_actual - y_pred

        # Pilih bobot langsung berdasarkan kondisi diff
        weights = np.where(diff >= 0, self.alpha, self.beta)

        # Gradient: dL/dy_pred
        return -2.0 * weights * diff / n
