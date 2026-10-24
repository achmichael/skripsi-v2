import numpy as np

class Adam:
    def __init__(
        self,
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8
    ):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.m = {}
        self.v = {}

        self.t = 0

    def update(self, layers):
        self.t += 1

        for index, layer in enumerate(layers):

            if index not in self.m:
                self.m[index] = {
                    "weights": np.zeros_like(layer.weights)
                }

                self.v[index] = {
                    "weights": np.zeros_like(layer.weights)
                }

                if hasattr(layer, 'biases'):
                    self.m[index]["biases"] = np.zeros_like(layer.biases)
                    self.v[index]["biases"] = np.zeros_like(layer.biases)

            self.m[index]["weights"] = (
                self.beta1 * self.m[index]["weights"]
                + (1 - self.beta1) * layer.dweights
            )

            self.v[index]["weights"] = (
                self.beta2 * self.v[index]["weights"]
                + (1 - self.beta2) * (layer.dweights ** 2)
            )

            m_hat_w = self.m[index]["weights"] / (
                1 - self.beta1 ** self.t
            )

            v_hat_w = self.v[index]["weights"] / (
                1 - self.beta2 ** self.t
            )

            layer.weights -= self.learning_rate * (
                m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)
            )

            if hasattr(layer, 'biases'):
                self.m[index]["biases"] = (
                    self.beta1 * self.m[index]["biases"]
                    + (1 - self.beta1) * layer.dbiases
                )

                self.v[index]["biases"] = (
                    self.beta2 * self.v[index]["biases"]
                    + (1 - self.beta2) * (layer.dbiases ** 2)
                )

                m_hat_b = self.m[index]["biases"] / (
                    1 - self.beta1 ** self.t
                )

                v_hat_b = self.v[index]["biases"] / (
                    1 - self.beta2 ** self.t
                )

                layer.biases -= self.learning_rate * (
                    m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)
                )
