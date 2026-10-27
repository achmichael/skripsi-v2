import copy

class EarlyStopping:
    def __init__(
        self,
        patience=15,
        min_delta=1e-5,
        mode="min"
    ):
        if patience < 1:
            raise ValueError("patience harus minimal 1")
        if min_delta < 0:
            raise ValueError("min_delta tidak boleh negatif")
        if mode not in ("min", "max"):
            raise ValueError("mode harus 'min' atau 'max'")

        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_value = None
        self.best_epoch = 0
        self.wait = 0
        self.should_stop = False
        self.best_parameters = None

    def _is_improvement(self, current_value):
        if self.best_value is None:
            return True
        if self.mode == "min":
            return current_value < self.best_value - self.min_delta
        return current_value > self.best_value + self.min_delta

    def _save_parameters(self, model):
        parameters = []
        for layer in model.layers:
            if hasattr(layer, "weights") and hasattr(layer, "biases"):
                parameters.append({
                    "weights": layer.weights.copy(),
                    "biases": layer.biases.copy()
                })
            elif hasattr(layer, "weights"):
                parameters.append({
                    "weights": layer.weights.copy(),
                })
            else:
                parameters.append(None)
        return parameters

    def _restore_parameters(self, model):
        if self.best_parameters is None:
            return
        for layer, saved in zip(
            model.layers, self.best_parameters
        ):
            if saved is not None:
                if hasattr(layer, "weights"):
                    layer.weights = saved["weights"].copy()
                if hasattr(layer, "biases") and "biases" in saved:
                    layer.biases = saved["biases"].copy()

    def step(self, current_value, epoch, model):
        if self._is_improvement(current_value):
            self.best_value = current_value
            self.best_epoch = epoch
            self.wait = 0
            self.best_parameters = self._save_parameters(model)

        else:
            self.wait += 1

        if self.wait >= self.patience:
            self.should_stop = True
            self._restore_parameters(model)

        return self.should_stop
