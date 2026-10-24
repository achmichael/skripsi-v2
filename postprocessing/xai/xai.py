from matplotlib.pylab import axis
from typing import Tuple
import numpy as np

from core.models.mlp import MLP

class IntegratedGradients:
    def __init__(
        self,
        model: MLP,
        baseline_strategy: str = "mean",
        steps: int = 100
    ):
        if steps < 1:
            raise ValueError("Steps minimal 1")

        self.model = model
        self.baseline_strategy = baseline_strategy
        self.steps = steps

    def _input_gradient(self, x: np.ndarray):
        prediction = self.model.forward(x)
        output_gradient = np.ones_like(prediction)
        return self.model.backward(output_gradient)

    def explain(self, x: np.ndarray, baseline: np.ndarray | None = None):
        if x.ndim == 1:
            x = x.reshape(1, -1)

        if baseline is None:
            baseline = np.zeros_like(x)
        else:
            baseline = np.asarray(baseline, float)

            if baseline.ndim == 1:
                baseline = baseline.reshape(1, -1)

        if baseline.shape != x.shape:
            raise ValueError("Dimensi baseline harus sama dengan dimensi fitur")

        difference = x - baseline

        # trapezoidal integration
        total_gradient = np.zeros_like(x)

        for k in range(self.steps + 1):
            alpha = k / self.steps
            interpolated_x = baseline + alpha * difference

            gradient = self._input_gradient(interpolated_x)
            weight = 0.5 if k == 0 or k == self.steps else 1.0
            total_gradient += weight * gradient

        average_gradient = total_gradient / self.steps

        attributions = difference * average_gradient

        baseline_prediction = float(
            self.model.forward(baseline)[0, 0]
        )

        prediction = float(
            self.model.forward(x)[0, 0]
        )

        attribution_sum = float(np.sum(attributions))

        prediction_difference = prediction - baseline_prediction

        completeness_delta = (
            prediction_difference - attribution_sum
        )

        return {
            "attributions": attributions.reshape(-1),
            "prediction": prediction,
            "baseline_prediction": baseline_prediction,
            "attribution_sum": attribution_sum,
            "prediction_difference": prediction_difference,
            "completeness_delta": completeness_delta,
        }

    def feature_importance(
        self,
        X: np.ndarray,
        feature_names: Tuple[str],
        baseline: np.ndarray | None = None
    ):
        if X.ndim != 2:
            raise ValueError("Fitur harus berupa 2 dimensi")

        if len(feature_names) != X.shape[1]:
            raise ValueError("Jumlah nama fitur harus sama dengan jumlah kolom x")

        all_attributions = []

        for sample in X:
            result = self.explain(sample, baseline)
            all_attributions.append(result["attributions"])

        all_attributions = np.asarray(all_attributions)
        mean_absolute = np.mean(np.abs(all_attributions), axis=0)

        mean_signed = np.mean(all_attributions, axis=0)

        total = np.sum(mean_absolute)

        if total > 0:
            relative_importance = mean_absolute / total * 100
        else:
            relative_importance = np.zeros_like(mean_absolute)

        order = np.argsort(mean_absolute)[::-1]

        return [
            {
                "feature": feature_names[i],
                "mean_absolute_ig": float(mean_absolute[i]),
                "mean_signed_ig": float(mean_signed[i]),
                "relative_importance_pct": float(
                    relative_importance[i]
                ),
            }
            for i in order
        ]
