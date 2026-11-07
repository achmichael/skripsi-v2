"""
Random Search Hyperparameter Tuning

Module ini melakukan hyperparameter tuning menggunakan random search.
Setiap kombinasi dievaluasi dengan k-fold cross-validation yang sama seperti
pipeline utama (train.py): scaling fit per fold, model baru per fold, dan
metrik dihitung pada skala target asli.
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


class RandomSearch:
    """
    Random Search untuk hyperparameter tuning neural network.

    Perankingan berdasarkan rata-rata MAE k-fold cross-validation pada skala
    target asli (inverse transformed).
    """

    ARCH_KEYS = ("n_hidden_layers", "hidden_1", "hidden_2")

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
        verbose: bool = True,
        k_folds: int = 5,
        feature_columns: Optional[List[str]] = None,
        encoding_cols: Optional[List[str]] = None,
        use_log: bool = False
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
            verbose: Print progress
            k_folds: Jumlah fold cross-validation
            feature_columns: Nama fitur (untuk selective feature scaling)
            encoding_cols: Kolom yang tidak ikut di-scale
            use_log: Apakah target memakai log transform (sama seperti train.py)
        """
        self.search_space = search_space
        self.n_trials = n_trials
        self.verbose = verbose
        self.k_folds = k_folds
        self.seed = seed
        self.feature_columns = list(
            range(input_size) if feature_columns is None else feature_columns
        )
        self.encoding_cols = (
            set() if encoding_cols is None else set(encoding_cols)
        )
        self.use_log = use_log

        self.sampler = HyperparameterSampler(
            search_space=search_space,
            enforce_funnel=enforce_funnel,
            seed=seed
        )

        self.trainer_param_map = trainer_param_map or {}
        self.model_factory = model_factory
        self.trainer_factory = trainer_factory
        self.input_size = input_size

        # Results
        self.results: List[Dict[str, Any]] = []
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_score: float = float("inf")
        self.best_model: Any = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomSearch":
        """
        Jalankan random search dengan k-fold cross-validation.

        Args:
            X: Fitur mentah (scaling dilakukan per fold)
            y: Target mentah

        Returns:
            self
        """
        self._reset_results()

        # Generate random combinations
        combinations = self.sampler.sample(self.n_trials)
        n_total = len(combinations)

        if self.verbose:
            print(f"Random Search | trials={n_total} | {self.k_folds}-fold CV")
            print("=" * 70)

        # Evaluate each combination
        for trial_idx, params in enumerate(combinations, start=1):
            result = self._evaluate_trial(
                trial_idx=trial_idx,
                n_total=n_total,
                params=params,
                X=X,
                y=y,
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
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, Any]:
        """Evaluate satu trial dengan k-fold cross-validation."""
        from experiments.k_fold_cv import KFoldCrossValidation

        hidden_layers = self._extract_hidden_layers(params)
        trainer_params = self._extract_trainer_params(params)

        try:
            kfold = KFoldCrossValidation(
                k=self.k_folds,
                shuffle=True,
                random_state=self.seed,
            )
            fold_results = kfold.evaluate(
                X=X,
                y=y,
                model_factory=self.model_factory,
                trainer_factory=self.trainer_factory,
                model_params={
                    "input_size": self.input_size,
                    "hidden_layers": hidden_layers,
                },
                trainer_params=trainer_params,
                feature_columns=self.feature_columns,
                encoding_cols=self.encoding_cols,
                use_log=self.use_log,
                seed=self.seed,
                verbose=False,
            )

            fold_scores = [float(row["mae"]) for row in fold_results]
            fold_best_epochs = [int(row["best_epoch"]) for row in fold_results]

            mae = float(np.mean(fold_scores))
            mae_std = (
                float(np.std(fold_scores, ddof=1))
                if len(fold_scores) > 1
                else 0.0
            )
            best_epoch = int(round(np.mean(fold_best_epochs)))
            status = "ok"

            if not math.isfinite(mae):
                mae = float("inf")
                status = "failed: MAE NaN/inf (training diverge)"
        except Exception as exc:
            fold_scores = []
            mae = float("inf")
            mae_std = float("nan")
            best_epoch = None
            status = f"failed: {type(exc).__name__}: {exc}"

        return {
            "trial": trial_idx,
            "params": deepcopy(params),
            "hidden_layers": hidden_layers,
            "mae": mae,
            "mae_std": mae_std,
            "fold_maes": fold_scores,
            "best_epoch": best_epoch,
            "status": status,
            "model": None,
        }

    def _extract_hidden_layers(self, params: Dict[str, Any]) -> List[int]:
        """Extract hidden layer sizes dari params."""
        n_layers = params.get("n_hidden_layers", 2)
        hidden_layers = [params["hidden_1"]]
        if n_layers == 2:
            hidden_layers.append(params["hidden_2"])
        return hidden_layers

    def _extract_trainer_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Extract trainer parameters dari params dengan mapping."""
        trainer_params = {}

        for key, value in params.items():
            if key in self.ARCH_KEYS:
                continue
            mapped_key = self.trainer_param_map.get(key, key)
            trainer_params[mapped_key] = value

        return trainer_params

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
            f"CV MAE={result['mae']:.6f} ± {result['mae_std']:.6f} | "
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
            Dictionary: {n_layers: {n_trials, best_mae, mean_mae, std_mae}}
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
            header = (
                ["rank", "trial", "mae", "mae_std", "best_epoch", "status"]
                + param_keys
            )
            writer.writerow(header)

            # Rows
            for rank, result in enumerate(ranked, start=1):
                row = [
                    rank,
                    result["trial"],
                    result["mae"],
                    result["mae_std"],
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

    from core.config.config import configs
    from preprocessing.preprocess import preprocess
    from feature_extraction.feature_extraction import extractor
    from core.models.mlp import MLP
    from core.losses.asymmetric_mse import AsymmetricMSE
    from training.trainer import Trainer

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
        "--folds",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5)"
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
    print(f"Cross-validation: {args.folds}-fold")
    print(f"Output: {args.output}")
    print(f"Seed: {args.seed}")
    print("=" * 70)
    print()

    cfg = configs["models_config"][args.capacity]

    # Load and preprocess data
    print("[1/3] Loading and preprocessing data...")
    dataset_path = cfg["dataset_path"]
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found: {dataset_path}")
        sys.exit(1)

    rows = preprocess(dataset_path)
    x_data, y_data, feature_columns, target_col = extractor(rows, args.capacity)

    print(f"   Features: {len(feature_columns)}")
    print(f"   Samples: {len(x_data)}")

    # Data mentah; scaling dilakukan per fold di dalam cross-validation.
    print("[2/3] Preparing raw data for cross-validation...")
    X = np.array(x_data, dtype=np.float32)
    y = np.array(y_data, dtype=np.float32).reshape(-1, 1)
    encoding_cols = set(configs["preprocessing"]["no_scaling_cols"])

    input_size = len(feature_columns)

    # Define model factory
    def model_factory(input_size: int, hidden_layers: List[int]):
        """Create MLP model dengan arsitektur tertentu."""
        layer_sizes = [input_size] + hidden_layers + [1]
        return MLP(layer_sizes)

    # Define trainer factory
    def trainer_factory(
        learning_rate,
        batch_size,
        patience,
        clip_value,
    ):
        """Create trainer dengan hyperparameter tertentu."""
        # Hyperparameter yang tidak dituning mengikuti konfigurasi train.py.
        cfg_train = {
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "patience": patience,
            "min_delta": cfg["min_delta"],
            "clip_value": clip_value,
        }

        class CVTrainer:
            """Trainer wrapper yang memakai logika Trainer._train_fold."""

            def fit(self, model, X_train, y_train, X_val, y_val):
                loss_fn = AsymmetricMSE(
                    beta=cfg["asymmetric_beta"],
                    alpha=cfg["asymmetric_alpha"],
                )
                history, early_stopping = Trainer._train_fold(
                    model=model,
                    loss=loss_fn,
                    x_train_np=X_train,
                    y_train_np=y_train,
                    x_val_np=X_val,
                    y_val_np=y_val,
                    cfg=cfg_train,
                    epochs=1000,
                    verbose=False,
                )
                return model, history, early_stopping.best_epoch

        return CVTrainer()

    # Run random search
    print(f"[3/3] Running random search ({args.folds}-fold CV)...")
    print()

    random_search = RandomSearch(
        search_space=SEARCH_SPACE,
        model_factory=model_factory,
        trainer_factory=trainer_factory,
        input_size=input_size,
        n_trials=args.trials,
        seed=args.seed,
        enforce_funnel=True,
        verbose=True,
        k_folds=args.folds,
        feature_columns=feature_columns,
        encoding_cols=encoding_cols,
        use_log=cfg["use_log_transform"],
    )

    random_search.fit(X, y)

    # Print results
    print()
    print("=" * 70)
    print("RANDOM SEARCH RESULTS")
    print("=" * 70)
    print(f"Best CV MAE: {random_search.best_score:.4f}")
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
        print(f"   CV MAE: {result['mae']:.4f} ± {result['mae_std']:.4f}")
        print(f"   Hidden layers: {result['hidden_layers']}")
        print(f"   Best epoch (mean): {result['best_epoch']}")
        print(f"   Params: {result['params']}")

    # Save results
    random_search.save_results(args.output)
    print()
    print("=" * 70)
    print(f"Results saved to: {args.output}")
    print("=" * 70)


if __name__ == "__main__":
    main()
