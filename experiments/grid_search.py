# experiments/grid_search.py

import csv
import math
from copy import deepcopy
from itertools import product
from experiments.k_fold_cv import KFoldCrossValidation

import numpy as np


# ---------------------------------------------------------------------------
# SEARCH SPACE
# ---------------------------------------------------------------------------
# Setiap hyperparameter punya banyak kandidat nilai agar pencarian beragam.
# Learning rate dan clip value memakai skala logaritmik / bertingkat
# karena pengaruhnya biasanya terlihat pada orde besaran, bukan selisih kecil.
SEARCH_SPACE = {
    # Arsitektur: pencarian ikut menentukan 1 atau 2 hidden layer.
    # Jika n_hidden_layers=1, hidden_2 diabaikan (dicatat sebagai None).
    "n_hidden_layers": [1, 2],
    "hidden_1": [4, 8, 12, 16, 24, 32, 48, 64, 96, 128],
    "hidden_2": [8, 16, 24, 32, 48, 64, 96, 128, 192, 256],
    # Optimasi
    "learning_rate": [1e-4, 3e-4, 5e-4, 1e-3, 3e-3, 5e-3, 1e-2, 3e-2, 5e-2],
    "batch_size": [8, 16, 32, 64, 128, 256],
    # Early stopping: toleransi jumlah epoch tanpa perbaikan (patience)
    "patience": [5, 10, 15, 20, 30, 50],
    # Gradient clipping
    "clip_value": [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
}


class GridSearch:
    """
    Hyperparameter tuning untuk neural network (1 atau 2 hidden layer)
    menggunakan pendekatan Full Grid Search.

    Setiap kombinasi dievaluasi dengan k-fold cross-validation yang sama
    seperti pipeline utama (train.py): scaling fit per fold, model baru per
    fold, dan metrik dihitung pada skala target asli.

    Kontrak yang diasumsikan:
        model_factory(input_size=..., hidden_layers=[h1] atau [h1, h2]) -> model
        trainer_factory(**trainer_params)                               -> trainer
        trainer.fit(model=, X_train=, y_train=, X_val=, y_val=)
            -> (model, history, best_epoch)

    trainer_param_map:
        Pemetaan nama hyperparameter di search space -> nama argumen di
        Trainer Anda. Contoh: {"patience": "tolerance"}.
        Nama yang tidak ada di map dipakai apa adanya.
    """

    ARCH_KEYS = ("n_hidden_layers", "hidden_1", "hidden_2")

    def __init__(
        self,
        search_space,
        model_factory,
        trainer_factory,
        input_size,
        max_grid_trials=999999999999,
        seed=42,
        enforce_funnel=True,
        trainer_param_map=None,
        scoring="mae",
        verbose=True,
        k_folds=5,
        feature_columns=None,
        encoding_cols=None,
        use_log=False,
    ):
        if scoring != "mae":
            raise ValueError("Implementasi ini hanya mendukung scoring='mae'")

        self.search_space = search_space
        self.model_factory = model_factory
        self.trainer_factory = trainer_factory
        self.input_size = input_size
        self.max_grid_trials = max_grid_trials
        self.seed = seed
        self.enforce_funnel = enforce_funnel  # hidden_2 <= hidden_1 (2 layer)
        self.trainer_param_map = trainer_param_map or {}
        self.scoring = scoring
        self.verbose = verbose
        self.k_folds = k_folds
        self.feature_columns = list(
            range(input_size) if feature_columns is None else feature_columns
        )
        self.encoding_cols = (
            set() if encoding_cols is None else set(encoding_cols)
        )
        self.use_log = use_log

        self.results = []
        self.best_params = None
        self.best_score = float("inf")
        self.best_model = None

    # ------------------------------------------------------------------
    # Pembangkitan kombinasi
    # ------------------------------------------------------------------
    def _normalize(self, params):
        """Jika hanya 1 hidden layer, hidden_2 tidak relevan -> None."""
        params = dict(params)
        if params.get("n_hidden_layers", 2) == 1:
            params["hidden_2"] = None
        return params

    def _is_valid(self, params):
        # Constraint "funnel" (hidden_2 <= hidden_1) hanya untuk 2 layer.
        if (
            self.enforce_funnel
            and params.get("n_hidden_layers", 2) == 2
            and "hidden_1" in params
            and "hidden_2" in params
        ):
            return params["hidden_2"] <= params["hidden_1"]
        return True

    def _effective_size(self):
        """Jumlah kombinasi unik (setelah normalisasi & constraint)."""
        space = self.search_space
        other = 1
        for key, values in space.items():
            if key not in self.ARCH_KEYS:
                other *= len(values)

        layer_options = space.get("n_hidden_layers", [2])
        h1_values = space["hidden_1"]
        h2_values = space["hidden_2"]

        arch = 0
        if 1 in layer_options:
            arch += len(h1_values)
        if 2 in layer_options:
            if self.enforce_funnel:
                arch += sum(1 for a in h1_values for b in h2_values if b <= a)
            else:
                arch += len(h1_values) * len(h2_values)
        return arch * other

    def _generate_combinations(self):
        total = self._effective_size()
        if total > self.max_grid_trials:
            raise ValueError(
                f"Full grid = {total} kombinasi (> max_grid_trials="
                f"{self.max_grid_trials}). Harap perkecil search space."
            )

        keys = list(self.search_space.keys())
        values = [self.search_space[k] for k in keys]
        seen = set()

        for combo in product(*values):
            params = dict(zip(keys, combo))
            if not self._is_valid(params):
                continue
            params = self._normalize(params)
            key = tuple(params[k] for k in keys)
            if key in seen:
                continue
            seen.add(key)
            yield params

    # ------------------------------------------------------------------
    # Pencarian
    # ------------------------------------------------------------------
    def _split_params(self, params):
        n_layers = params.get("n_hidden_layers", 2)
        hidden_layers = [params["hidden_1"]]
        if n_layers == 2:
            hidden_layers.append(params["hidden_2"])

        trainer_params = {}
        for key, value in params.items():
            if key in self.ARCH_KEYS:
                continue
            trainer_params[self.trainer_param_map.get(key, key)] = value
        return hidden_layers, trainer_params

    def _cross_validate(self, hidden_layers, trainer_params, X, y):
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
        return [float(row["mae"]) for row in fold_results], [
            int(row["best_epoch"]) for row in fold_results
        ]

    def fit(self, X, y):
        self.results = []
        self.best_params = None
        self.best_score = float("inf")
        self.best_model = None

        combos = list(self._generate_combinations())
        n_total = len(combos)

        if self.verbose:
            print(
                f"Mode=Grid Search | total kombinasi={n_total} | "
                f"ukuran search space efektif={self._effective_size()} | "
                f"{self.k_folds}-fold CV"
            )

        for trial, params in enumerate(combos, start=1):
            hidden_layers, trainer_params = self._split_params(params)

            try:
                fold_scores, fold_best_epochs = self._cross_validate(
                    hidden_layers=hidden_layers,
                    trainer_params=trainer_params,
                    X=X,
                    y=y,
                )
                score = float(np.mean(fold_scores))
                score_std = (
                    float(np.std(fold_scores, ddof=1))
                    if len(fold_scores) > 1
                    else 0.0
                )
                best_epoch = int(round(np.mean(fold_best_epochs)))
                status = "ok"
            except Exception as exc:  # training diverge / error numerik
                fold_scores = []
                score = float("inf")
                score_std = float("nan")
                best_epoch = None
                status = f"gagal: {type(exc).__name__}: {exc}"

            if not math.isfinite(score):
                score = float("inf")
                if status == "ok":
                    status = "gagal: MAE NaN/inf (training diverge)"

            self.results.append(
                {
                    "trial": trial,
                    "params": deepcopy(params),
                    "hidden_layers": hidden_layers,
                    "mae": score,
                    "mae_std": score_std,
                    "fold_maes": fold_scores,
                    "best_epoch": best_epoch,
                    "status": status,
                }
            )

            if self.verbose:
                print(
                    f"[{trial}/{n_total}] CV MAE={score:.6f} ± {score_std:.6f} | "
                    f"hidden_layers={hidden_layers} | "
                    f"best_epoch={best_epoch} | {params} | {status}"
                )

            if score < self.best_score:
                self.best_score = score
                self.best_params = deepcopy(params)

        return self

    # ------------------------------------------------------------------
    # Utilitas
    # ------------------------------------------------------------------
    def get_results(self, top_k=None):
        ranked = sorted(self.results, key=lambda row: row["mae"])
        return ranked[:top_k] if top_k else ranked

    def summarize_by_depth(self):
        """Ringkasan MAE terbaik & rata-rata untuk 1 vs 2 hidden layer."""
        summary = {}
        for n in sorted({r["params"].get("n_hidden_layers", 2) for r in self.results}):
            scores = [
                r["mae"]
                for r in self.results
                if r["params"].get("n_hidden_layers", 2) == n and math.isfinite(r["mae"])
            ]
            if scores:
                summary[n] = {
                    "n_trials": len(scores),
                    "best_mae": min(scores),
                    "mean_mae": float(np.mean(scores)),
                    "std_mae": float(np.std(scores)),
                }
        return summary

    def save_results(self, path="grid_search_results.csv"):
        ranked = self.get_results()
        if not ranked:
            return
        param_keys = list(self.search_space.keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["rank", "trial", "mae", "mae_std", "best_epoch", "status"]
                + param_keys
            )
            for rank, row in enumerate(ranked, start=1):
                writer.writerow(
                    [
                        rank,
                        row["trial"],
                        row["mae"],
                        row["mae_std"],
                        row["best_epoch"],
                        row["status"],
                    ]
                    + [row["params"].get(k) for k in param_keys]
                )


if __name__ == "__main__":
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

    parser = argparse.ArgumentParser(description="Grid Search Hyperparameter Tuning")
    parser.add_argument("--capacity", type=str, default="900", choices=["900", "1300", "2200"],
                        help="Capacity VA (default: 900)")
    parser.add_argument("--folds", type=int, default=5,
                        help="Number of cross-validation folds (default: 5)")
    parser.add_argument("--output", type=str,        default="grid_search_results.csv",
                        help="Output CSV file (default: grid_search_results.csv)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")

    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"GRID SEARCH HYPERPARAMETER TUNING")
    print(f"{'='*60}")
    print(f"Capacity: {args.capacity} VA")
    print(f"Mode: Full Grid Search ({args.folds}-fold CV)")
    print(f"Output: {args.output}")
    print(f"{'='*60}\n")

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

    # Define factories
    def model_factory(input_size, hidden_layers):
        layer_sizes = [input_size] + hidden_layers + [1]
        return MLP(layer_sizes)

    def trainer_factory(
        learning_rate,
        batch_size,
        patience,
        clip_value,
    ):
        # Hyperparameter yang tidak dituning mengikuti konfigurasi train.py.
        cfg_train = {
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "patience": patience,
            "min_delta": cfg["min_delta"],
            "clip_value": clip_value,
        }

        class CVTrainer:
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

    # Run grid search
    print(f"[3/3] Running full grid search ({args.folds}-fold CV)...\n")

    grid_search = GridSearch(
        search_space=SEARCH_SPACE,
        model_factory=model_factory,
        trainer_factory=trainer_factory,
        input_size=input_size,
        seed=args.seed,
        enforce_funnel=True,
        scoring="mae",
        verbose=True,
        k_folds=args.folds,
        feature_columns=feature_columns,
        encoding_cols=encoding_cols,
        use_log=cfg["use_log_transform"],
    )

    grid_search.fit(X, y)

    # Print results
    print(f"\n{'='*60}")
    print(f"GRID SEARCH RESULTS")
    print(f"{'='*60}")
    print(f"Best CV MAE: {grid_search.best_score:.4f}")
    print(f"Best params: {grid_search.best_params}")

    # Summary by depth
    print(f"\n{'='*60}")
    print(f"SUMMARY BY DEPTH")
    print(f"{'='*60}")
    depth_summary = grid_search.summarize_by_depth()
    for n_layers, stats in depth_summary.items():
        print(f"\n{n_layers} hidden layer(s):")
        print(f"   Trials: {stats['n_trials']}")
        print(f"   Best MAE: {stats['best_mae']:.4f}")
        print(f"   Mean MAE: {stats['mean_mae']:.4f}")
        print(f"   Std MAE: {stats['std_mae']:.4f}")

    # Top 10 results
    print(f"\n{'='*60}")
    print(f"TOP 10 RESULTS")
    print(f"{'='*60}")
    top_10 = grid_search.get_results(top_k=10)
    for rank, result in enumerate(top_10, start=1):
        print(f"\nRank {rank}:")
        print(f"   CV MAE: {result['mae']:.4f} ± {result['mae_std']:.4f}")
        print(f"   Hidden layers: {result['hidden_layers']}")
        print(f"   Params: {result['params']}")
        print(f"   Best epoch (mean): {result['best_epoch']}")

    # Save results
    grid_search.save_results(args.output)
    print(f"\n{'='*60}")
    print(f"Results saved to: {args.output}")
    print(f"{'='*60}")
