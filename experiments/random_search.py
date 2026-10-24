"""
Random Search Hyperparameter Tuning

Module ini melakukan hyperparameter tuning menggunakan random search.
Perankingan berdasarkan MAE pada data skala asli (setelah inverse transform).
"""

import csv
import math
import random
from copy import deepcopy
from typing import Dict, List, Tuple, Callable, Optional, Any

import numpy as np


# ---------------------------------------------------------------------------
# SEARCH SPACE
# ---------------------------------------------------------------------------
SEARCH_SPACE = {
    # Arsitektur
    "n_hidden_layers": [1, 2],
    "hidden_1": [8, 16, 24, 32, 48, 64, 96, 128, 192, 256],
    "hidden_2": [4, 8, 12, 16, 24, 32, 48, 64, 96, 128],
    # Optimasi
    "learning_rate": [1e-4, 3e-4, 5e-4, 1e-3, 3e-3, 5e-3, 1e-2, 3e-2, 5e-2],
    "batch_size": [16, 32, 64, 128, 256],
    # Regularisasi
    "l2": [0.0, 1e-7, 1e-6, 1e-5, 1e-4, 5e-4, 1e-3, 1e-2],
    # Early stopping
    "patience": [5, 10, 15, 20, 30, 50],
    # Gradient clipping
    "clip_value": [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
}


class HyperparameterSampler:
    """
    Random sampler untuk hyperparameter dengan constraint validation.

    Constraint:
    - Jika n_hidden_layers=1, hidden_2 diabaikan (set None)
    - Jika enforce_funnel=True dan n_hidden_layers=2, maka hidden_2 <= hidden_1
    """

    ARCH_KEYS = ("n_hidden_layers", "hidden_1", "hidden_2")

    def __init__(
        self,
        search_space: Dict[str, List],
        enforce_funnel: bool = True,
        seed: int = 42
    ):
        self.search_space = search_space
        self.enforce_funnel = enforce_funnel
        self.rng = random.Random(seed)

    def sample(self, n_samples: int) -> List[Dict[str, Any]]:
        """Generate n_samples kombinasi hyperparameter unik yang valid."""
        samples = []
        seen = set()
        max_attempts = n_samples * 200
        attempts = 0

        while len(samples) < n_samples and attempts < max_attempts:
            attempts += 1
            params = self._sample_one()

            if not self._is_valid(params):
                continue

            params = self._normalize(params)
            key = self._params_to_hashable(params)

            if key in seen:
                continue

            seen.add(key)
            samples.append(params)

        return samples

    def _sample_one(self) -> Dict[str, Any]:
        """Sample satu kombinasi hyperparameter secara random."""
        return {
            key: self.rng.choice(values)
            for key, values in self.search_space.items()
        }

    def _is_valid(self, params: Dict[str, Any]) -> bool:
        """Validasi constraint funnel architecture."""
        if not self.enforce_funnel:
            return True

        n_layers = params.get("n_hidden_layers", 2)
        if n_layers == 2:
            h1 = params.get("hidden_1")
            h2 = params.get("hidden_2")
            if h1 is not None and h2 is not None:
                return h2 <= h1

        return True

    def _normalize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize params: jika 1 layer, set hidden_2 = None."""
        params = dict(params)
        if params.get("n_hidden_layers", 2) == 1:
            params["hidden_2"] = None
        return params

    def _params_to_hashable(self, params: Dict[str, Any]) -> Tuple:
        """Convert params dict ke tuple untuk hashing."""
        keys = sorted(self.search_space.keys())
        return tuple(params.get(k) for k in keys)


class ModelEvaluator:
    """
    Evaluator untuk training dan evaluasi model dengan hyperparameter tertentu.
    """

    def __init__(
        self,
        model_factory: Callable,
        trainer_factory: Callable,
        input_size: int,
        trainer_param_map: Optional[Dict[str, str]] = None
    ):
        self.model_factory = model_factory
        self.trainer_factory = trainer_factory
        self.input_size = input_size
        self.trainer_param_map = trainer_param_map or {}

    def evaluate(
        self,
        params: Dict[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Tuple[Any, Dict[str, Any], str]:
        """
        Train dan evaluasi model dengan hyperparameter tertentu.

        Returns:
            (model, metrics_dict, status_string)
        """
        try:
            # Build model
            hidden_layers = self._extract_hidden_layers(params)
            model = self.model_factory(
                input_size=self.input_size,
                hidden_layers=hidden_layers
            )

            # Build trainer
            trainer_params = self._extract_trainer_params(params)
            trainer = self.trainer_factory(**trainer_params)

            # Train
            model, history, best_epoch = trainer.fit(
                model=model,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val
            )

            # Get predictions
            y_pred = model.forward(X_val)

            metrics = {
                "model": model,
                "y_pred": y_pred,
                "best_epoch": best_epoch,
                "history": history
            }

            return model, metrics, "ok"

        except Exception as exc:
            return None, {}, f"failed: {type(exc).__name__}: {exc}"

    def _extract_hidden_layers(self, params: Dict[str, Any]) -> List[int]:
        """Extract hidden layer sizes dari params."""
        n_layers = params.get("n_hidden_layers", 2)
        hidden_layers = [params["hidden_1"]]
        if n_layers == 2:
            hidden_layers.append(params["hidden_2"])
        return hidden_layers

    def _extract_trainer_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Extract trainer parameters dari params dengan mapping."""
        arch_keys = ("n_hidden_layers", "hidden_1", "hidden_2")
        trainer_params = {}

        for key, value in params.items():
            if key in arch_keys:
                continue
            mapped_key = self.trainer_param_map.get(key, key)
            trainer_params[mapped_key] = value

        return trainer_params


class MetricCalculator:
    """
    Calculator untuk metrik evaluasi dengan support inverse transform.
    """

    def __init__(
        self,
        y_scaler: Any = None,
        inverse_fn: Optional[Callable] = None
    ):
        self.y_scaler = y_scaler
        self.inverse_fn = inverse_fn

    def calculate_mae(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        use_original_scale: bool = True
    ) -> float:
        """
        Hitung MAE. Jika use_original_scale=True, inverse transform dulu.

        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            use_original_scale: Jika True, inverse transform ke skala asli

        Returns:
            MAE score
        """
        if use_original_scale:
            y_true = self._inverse_transform(y_true)
            y_pred = self._inverse_transform(y_pred)

        y_true = np.asarray(y_true).reshape(-1)
        y_pred = np.asarray(y_pred).reshape(-1)

        if y_true.shape != y_pred.shape:
            raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")

        mae = float(np.mean(np.abs(y_true - y_pred)))

        if not math.isfinite(mae):
            return float("inf")

        return mae

    def _inverse_transform(self, y_scaled: np.ndarray) -> np.ndarray:
        """Inverse transform dari skala normalisasi ke skala asli."""
        if self.inverse_fn is None or self.y_scaler is None:
            return y_scaled

        flat = np.asarray(y_scaled).reshape(-1).tolist()
        return np.array(
            self.inverse_fn(flat, self.y_scaler),
            dtype=np.float64
        ).reshape(y_scaled.shape if hasattr(y_scaled, "shape") else (-1,))


class RandomSearch:
    """
    Random Search untuk hyperparameter tuning neural network.

    Perankingan berdasarkan MAE pada data skala asli (inverse transformed).
    """

    def __init__(
        self,
        search_space: Dict[str, List],
        model_factory: Callable,
        trainer_factory: Callable,
        input_size: int,
        n_trials: int = 100,
        seed: int = 42,
        enforce_funnel: bool = True,
        trainer_param_map: Optional[Dict[str, str]] = None,
        y_scaler: Any = None,
        inverse_fn: Optional[Callable] = None,
        verbose: bool = True
    ):
        """
        Args:
            search_space: Dictionary berisi hyperparameter dan nilai kandidatnya
            model_factory: Function(input_size, hidden_layers) -> model
            trainer_factory: Function(**trainer_params) -> trainer
            input_size: Jumlah fitur input
            n_trials: Jumlah kombinasi yang akan dicoba
            seed: Random seed untuk reproducibility
            enforce_funnel: Jika True, enforce hidden_2 <= hidden_1
            trainer_param_map: Mapping hyperparameter name ke trainer param name
            y_scaler: Scaler untuk inverse transform target
            inverse_fn: Function untuk inverse transform
            verbose: Print progress
        """
        self.search_space = search_space
        self.n_trials = n_trials
        self.verbose = verbose

        # Components
        self.sampler = HyperparameterSampler(
            search_space=search_space,
            enforce_funnel=enforce_funnel,
            seed=seed
        )

        self.evaluator = ModelEvaluator(
            model_factory=model_factory,
            trainer_factory=trainer_factory,
            input_size=input_size,
            trainer_param_map=trainer_param_map
        )

        self.metric_calculator = MetricCalculator(
            y_scaler=y_scaler,
            inverse_fn=inverse_fn
        )

        # Results
        self.results: List[Dict[str, Any]] = []
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_score: float = float("inf")
        self.best_model: Any = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> "RandomSearch":
        """
        Jalankan random search.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target

        Returns:
            self
        """
        self._reset_results()

        # Generate random combinations
        combinations = self.sampler.sample(self.n_trials)
        n_total = len(combinations)

        if self.verbose:
            print(f"Random Search | trials={n_total}")
            print("=" * 70)

        # Pre-compute inverse transformed y_val (tetap untuk semua trial)
        y_val_original = self.metric_calculator._inverse_transform(y_val)

        # Evaluate each combination
        for trial_idx, params in enumerate(combinations, start=1):
            result = self._evaluate_trial(
                trial_idx=trial_idx,
                n_total=n_total,
                params=params,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                y_val_original=y_val_original
            )

            self.results.append(result)
            self._update_best(result)

            if self.verbose:
                self._print_trial_result(result, trial_idx, n_total)

        return self

    def _reset_results(self):
        """Reset hasil pencarian."""
        self.results = []
        self.best_params = None
        self.best_score = float("inf")
        self.best_model = None

    def _evaluate_trial(
        self,
        trial_idx: int,
        n_total: int,
        params: Dict[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        y_val_original: np.ndarray
    ) -> Dict[str, Any]:
        """Evaluate satu trial."""
        # Train model
        model, metrics, status = self.evaluator.evaluate(
            params=params,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val
        )

        # Calculate MAE on original scale
        if status == "ok":
            y_pred = metrics["y_pred"]
            y_pred_original = self.metric_calculator._inverse_transform(y_pred)
            mae = self.metric_calculator.calculate_mae(
                y_val_original,
                y_pred_original,
                use_original_scale=False  # Already inverse transformed
            )
            best_epoch = metrics.get("best_epoch")
        else:
            mae = float("inf")
            best_epoch = None

        # Build result
        hidden_layers = self.evaluator._extract_hidden_layers(params)

        return {
            "trial": trial_idx,
            "params": deepcopy(params),
            "hidden_layers": hidden_layers,
            "mae": mae,
            "best_epoch": best_epoch,
            "status": status,
            "model": model
        }

    def _update_best(self, result: Dict[str, Any]):
        """Update best result jika lebih baik."""
        mae = result["mae"]
        if mae < self.best_score:
            self.best_score = mae
            self.best_params = deepcopy(result["params"])
            self.best_model = result["model"]

    def _print_trial_result(self, result: Dict[str, Any], trial_idx: int, n_total: int):
        """Print hasil satu trial."""
        print(
            f"[{trial_idx}/{n_total}] "
            f"MAE={result['mae']:.6f} | "
            f"layers={result['hidden_layers']} | "
            f"epoch={result['best_epoch']} | "
            f"{result['status']}"
        )

    def get_results(self, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Dapatkan hasil yang sudah diranking berdasarkan MAE (ascending).

        Args:
            top_k: Jika diberikan, return top k results

        Returns:
            List of result dicts, sorted by MAE
        """
        ranked = sorted(self.results, key=lambda r: r["mae"])
        return ranked[:top_k] if top_k else ranked

    def summarize_by_depth(self) -> Dict[int, Dict[str, Any]]:
        """
        Ringkasan statistik MAE berdasarkan jumlah hidden layer.

        Returns:
            Dictionary: {n_layers: {n_trials, best_mae, mean_mae}}
        """
        summary = {}

        for n_layers in sorted({r["params"]["n_hidden_layers"] for r in self.results}):
            scores = [
                r["mae"]
                for r in self.results
                if r["params"]["n_hidden_layers"] == n_layers
                and math.isfinite(r["mae"])
            ]

            if scores:
                summary[n_layers] = {
                    "n_trials": len(scores),
                    "best_mae": min(scores),
                    "mean_mae": float(np.mean(scores)),
                    "std_mae": float(np.std(scores))
                }

        return summary

    def save_results(self, path: str = "random_search_results.csv"):
        """
        Save hasil ke CSV file.

        Args:
            path: Path output CSV file
        """
        ranked = self.get_results()
        if not ranked:
            print("No results to save.")
            return

        param_keys = sorted(self.search_space.keys())

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # Header
            header = ["rank", "trial", "mae", "best_epoch", "status"] + param_keys
            writer.writerow(header)

            # Rows
            for rank, result in enumerate(ranked, start=1):
                row = [
                    rank,
                    result["trial"],
                    result["mae"],
                    result["best_epoch"],
                    result["status"]
                ]
                row.extend([result["params"].get(k) for k in param_keys])
                writer.writerow(row)


def main():
    """Entry point untuk command line interface."""
    import argparse
    import sys
    import os

    # Add parent directory to path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from preprocessing.preprocess import preprocess, train_test_split
    from feature_extraction.feature_extraction import extractor
    from transformation.feature import fit_standard_scaler, transform_standard
    from transformation.target import fit_minmax_scaler, transform_minmax
    from transformation.inverse_transform import inverse_transform_target
    from core.models.mlp import MLP
    from core.losses.asymmetric_mse import AsymmetricMSE
    from core.optimizers.adam import Adam
    from training.trainer import Trainer
    from training.early_stopping import EarlyStopping
    from training.gradient_clipping import clip_gradients
    from core.regularizers.l2 import L2Regularizer

    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Random Search Hyperparameter Tuning"
    )
    parser.add_argument(
        "--capacity",
        type=str,
        default="900",
        choices=["900", "1300", "2200"],
        help="Capacity VA (default: 900)"
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=150,
        help="Number of trials (default: 150)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="random_search_results.csv",
        help="Output CSV file (default: random_search_results.csv)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )

    args = parser.parse_args()

    # Print header
    print("=" * 70)
    print("RANDOM SEARCH HYPERPARAMETER TUNING")
    print("=" * 70)
    print(f"Capacity: {args.capacity} VA")
    print(f"Trials: {args.trials}")
    print(f"Output: {args.output}")
    print(f"Seed: {args.seed}")
    print("=" * 70)
    print()

    # Load and preprocess data
    print("[1/4] Loading and preprocessing data...")
    dataset_path = f"data/prabayar_{args.capacity}.csv"
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found: {dataset_path}")
        sys.exit(1)

    rows = preprocess(dataset_path)
    x_data, y_data, feature_columns, target_col = extractor(rows, args.capacity)

    print(f"   Features: {len(feature_columns)}")
    print(f"   Samples: {len(x_data)}")

    # Split data
    print("[2/4] Splitting data...")
    x_train, y_train, x_test, y_test = train_test_split(x_data, y_data, 0.2, 42)

    # Validation split from training data
    val_size = int(len(x_train) * 0.15)
    x_val = x_train[:val_size]
    y_val = y_train[:val_size]
    x_train = x_train[val_size:]
    y_train = y_train[val_size:]

    print(f"   Train: {len(x_train)}, Val: {len(x_val)}, Test: {len(x_test)}")

    # Scale data
    print("[3/4] Scaling data...")
    x_scaler = fit_standard_scaler(x_train)
    x_train_scaled = transform_standard(x_train, x_scaler)
    x_val_scaled = transform_standard(x_val, x_scaler)

    y_scaler = fit_minmax_scaler(y_train)
    y_train_scaled = transform_minmax(y_train, y_scaler)
    y_val_scaled = transform_minmax(y_val, y_scaler)

    # Convert to numpy
    X_train = np.array(x_train_scaled, dtype=np.float32)
    y_train_np = np.array(y_train_scaled, dtype=np.float32).reshape(-1, 1)
    X_val = np.array(x_val_scaled, dtype=np.float32)
    y_val_np = np.array(y_val_scaled, dtype=np.float32).reshape(-1, 1)

    input_size = len(feature_columns)

    # Define model factory
    def model_factory(input_size: int, hidden_layers: List[int]):
        """Create MLP model dengan arsitektur tertentu."""
        layer_sizes = [input_size] + hidden_layers + [1]
        return MLP(layer_sizes)

    # Define trainer factory
    def trainer_factory(learning_rate, batch_size, l2, patience, clip_value):
        """Create trainer dengan hyperparameter tertentu."""

        class SimpleTrainer:
            """Trainer wrapper untuk grid/random search."""

            def __init__(self):
                self.lr = learning_rate
                self.batch_size = batch_size
                self.l2_lambda = l2
                self.patience = patience
                self.clip_value = clip_value

            def fit(self, model, X_train, y_train, X_val, y_val):
                """Train model dengan early stopping."""
                loss_fn = AsymmetricMSE(beta=0.4, alpha=0.6)
                optimizer = Adam(self.lr)
                regularizer = L2Regularizer(lambda_=self.l2_lambda)
                early_stopping = EarlyStopping(
                    patience=self.patience,
                    min_delta=1e-5,
                    mode="min"
                )

                history = {"train_loss": [], "val_loss": []}

                for epoch in range(1, 1001):
                    # Shuffle training data
                    indices = np.random.permutation(len(X_train))
                    X_shuffled = X_train[indices]
                    y_shuffled = y_train[indices]

                    # Mini-batch training
                    epoch_losses = []
                    num_batches = (len(X_train) + self.batch_size - 1) // self.batch_size

                    for i in range(num_batches):
                        start_idx = i * self.batch_size
                        end_idx = min(start_idx + self.batch_size, len(X_train))
                        x_batch = X_shuffled[start_idx:end_idx]
                        y_batch = y_shuffled[start_idx:end_idx]

                        # Forward pass
                        y_pred = model.forward(x_batch)
                        batch_loss = loss_fn.forward(y_pred, y_batch)

                        # Backward pass
                        grad_loss = loss_fn.backward(y_pred, y_batch)
                        model.backward(grad_loss)

                        # Apply regularization
                        trainable_params = model.get_trainable_layers()
                        regularizer.apply_to_gradients(trainable_params, y_batch)

                        # Gradient clipping
                        clip_gradients(model, self.clip_value)

                        # Update weights
                        optimizer.update(trainable_params)

                        epoch_losses.append(batch_loss)

                    # Calculate training loss
                    train_loss = np.mean(epoch_losses)

                    # Add L2 penalty to loss
                    if self.l2_lambda > 0:
                        weights_list = [
                            layer.weights
                            for layer in model.layers
                            if hasattr(layer, 'weights') and layer.weights is not None
                        ]
                        l2_penalty = regularizer.compute_loss(weights_list, y_batch)
                        train_loss += l2_penalty

                    history["train_loss"].append(float(train_loss))

                    # Calculate validation loss
                    y_pred_val = model.forward(X_val)
                    val_loss = loss_fn.forward(y_pred_val, y_val)
                    history["val_loss"].append(float(val_loss))

                    # Check early stopping
                    should_stop = early_stopping.step(val_loss, epoch, model)
                    if should_stop:
                        break

                return model, history, early_stopping.best_epoch

        return SimpleTrainer()

    # Run random search
    print("[4/4] Running random search...")
    print()

    random_search = RandomSearch(
        search_space=SEARCH_SPACE,
        model_factory=model_factory,
        trainer_factory=trainer_factory,
        input_size=input_size,
        n_trials=args.trials,
        seed=args.seed,
        enforce_funnel=True,
        trainer_param_map={"l2": "l2"},
        y_scaler=y_scaler,
        inverse_fn=inverse_transform_target,
        verbose=True
    )

    random_search.fit(X_train, y_train_np, X_val, y_val_np)

    # Print results
    print()
    print("=" * 70)
    print("RANDOM SEARCH RESULTS")
    print("=" * 70)
    print(f"Best MAE: {random_search.best_score:.4f}")
    print(f"Best params: {random_search.best_params}")

    # Summary by depth
    print()
    print("=" * 70)
    print("SUMMARY BY DEPTH")
    print("=" * 70)
    depth_summary = random_search.summarize_by_depth()
    for n_layers, stats in depth_summary.items():
        print(f"\n{n_layers} hidden layer(s):")
        print(f"   Trials: {stats['n_trials']}")
        print(f"   Best MAE: {stats['best_mae']:.4f}")
        print(f"   Mean MAE: {stats['mean_mae']:.4f}")
        print(f"   Std MAE: {stats['std_mae']:.4f}")

    # Top 10 results
    print()
    print("=" * 70)
    print("TOP 10 RESULTS")
    print("=" * 70)
    top_10 = random_search.get_results(top_k=10)
    for rank, result in enumerate(top_10, start=1):
        print(f"\nRank {rank}:")
        print(f"   MAE: {result['mae']:.4f}")
        print(f"   Hidden layers: {result['hidden_layers']}")
        print(f"   Best epoch: {result['best_epoch']}")
        print(f"   Params: {result['params']}")

    # Save results
    random_search.save_results(args.output)
    print()
    print("=" * 70)
    print(f"Results saved to: {args.output}")
    print("=" * 70)


if __name__ == "__main__":
    main()
