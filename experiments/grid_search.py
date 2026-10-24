# experiments/grid_search.py

import csv
import math
from copy import deepcopy
from itertools import product

import numpy as np


# ---------------------------------------------------------------------------
# SEARCH SPACE
# ---------------------------------------------------------------------------
# Setiap hyperparameter punya banyak kandidat nilai agar pencarian beragam.
# Learning rate, L2, dan clip value memakai skala logaritmik / bertingkat
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
    # Regularisasi
    "l2": [0.0, 1e-7, 1e-6, 1e-5, 1e-4, 5e-4, 1e-3, 1e-2],
    # Early stopping: toleransi jumlah epoch tanpa perbaikan (patience)
    "patience": [5, 10, 15, 20, 30, 50],
    # Gradient clipping
    "clip_value": [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
}


class GridSearch:
    """
    Hyperparameter tuning untuk neural network (1 atau 2 hidden layer)
    menggunakan pendekatan Full Grid Search.

    Kontrak yang diasumsikan:
        model_factory(input_size=..., hidden_layers=[h1] atau [h1, h2]) -> model
        trainer_factory(**trainer_params)                               -> trainer
        trainer.fit(model=, X_train=, y_train=, X_val=, y_val=)
            -> (model, history, best_epoch)

    trainer_param_map:
        Pemetaan nama hyperparameter di search space -> nama argumen di
        Trainer Anda. Contoh: {"l2": "l2_lambda", "patience": "tolerance"}.
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
        y_scaler=None,
        inverse_fn=None,
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
        self.y_scaler = y_scaler
        self.inverse_fn = inverse_fn

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

    def _denormalize(self, y_scaled):
        """Inverse transform prediksi/aktual dari skala normalisasi ke skala asli."""
        if self.inverse_fn is None or self.y_scaler is None:
            return y_scaled
        flat = np.asarray(y_scaled).reshape(-1).tolist()
        return np.array(self.inverse_fn(flat, self.y_scaler), dtype=np.float64).reshape(
            y_scaled.shape if hasattr(y_scaled, "shape") else (-1,)
        )

    def fit(self, X_train, y_train, X_val, y_val):
        self.results = []
        self.best_params = None
        self.best_score = float("inf")
        self.best_model = None

        # Pre-compute denormalized y_val (tetap, tidak berubah tiap trial)
        y_val_denorm = self._denormalize(y_val)

        combos = list(self._generate_combinations())
        n_total = len(combos)

        if self.verbose:
            print(
                f"Mode=Grid Search | total kombinasi={n_total} | "
                f"ukuran search space efektif={self._effective_size()}"
            )

        for trial, params in enumerate(combos, start=1):
            hidden_layers, trainer_params = self._split_params(params)

            try:
                model = self.model_factory(
                    input_size=self.input_size,
                    hidden_layers=hidden_layers,
                )
                trainer = self.trainer_factory(**trainer_params)

                model, history, best_epoch = trainer.fit(
                    model=model,
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                )

                y_pred = model.forward(X_val)

                # Denormalisasi prediksi lalu hitung MAE di skala asli
                y_pred_denorm = self._denormalize(y_pred)
                score = self._mae(y_val_denorm, y_pred_denorm)
                status = "ok"
            except Exception as exc:  # training diverge / error numerik
                score = float("inf")
                best_epoch = None
                model = None
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
                    "best_epoch": best_epoch,
                    "status": status,
                }
            )

            if self.verbose:
                print(
                    f"[{trial}/{n_total}] MAE={score:.6f} | "
                    f"hidden_layers={hidden_layers} | "
                    f"best_epoch={best_epoch} | {params} | {status}"
                )

            if score < self.best_score:
                self.best_score = score
                self.best_params = deepcopy(params)
                self.best_model = model

        return self

    # ------------------------------------------------------------------
    # Utilitas
    # ------------------------------------------------------------------
    @staticmethod
    def _mae(y_true, y_pred):
        y_true = np.asarray(y_true).reshape(-1)
        y_pred = np.asarray(y_pred).reshape(-1)

        if y_true.shape != y_pred.shape:
            raise ValueError("Dimensi y_true dan y_pred tidak sama")

        return float(np.mean(np.abs(y_true - y_pred)))

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
                }
        return summary

    def save_results(self, path="grid_search_results.csv"):
        ranked = self.get_results()
        if not ranked:
            return
        param_keys = list(self.search_space.keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["rank", "trial", "mae", "best_epoch", "status"] + param_keys)
            for rank, row in enumerate(ranked, start=1):
                writer.writerow(
                    [rank, row["trial"], row["mae"], row["best_epoch"], row["status"]]
                    + [row["params"].get(k) for k in param_keys]
                )


if __name__ == "__main__":
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

    parser = argparse.ArgumentParser(description="Grid Search Hyperparameter Tuning")
    parser.add_argument("--capacity", type=str, default="900", choices=["900", "1300", "2200"],
                        help="Capacity VA (default: 900)")
    parser.add_argument("--output", type=str,        default="grid_search_results.csv",
                        help="Output CSV file (default: grid_search_results.csv)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")

    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"GRID SEARCH HYPERPARAMETER TUNING")
    print(f"{'='*60}")
    print(f"Capacity: {args.capacity} VA")
    print(f"Mode: Full Grid Search")
    print(f"Output: {args.output}")
    print(f"{'='*60}\n")

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

    # Validation split
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

    # Define factories
    def model_factory(input_size, hidden_layers):
        layer_sizes = [input_size] + hidden_layers + [1]
        return MLP(layer_sizes)

    def trainer_factory(learning_rate, batch_size, l2, patience, clip_value):
        class SimpleTrainer:
            def __init__(self):
                self.lr = learning_rate
                self.batch_size = batch_size
                self.l2_lambda = l2
                self.patience = patience
                self.clip_value = clip_value

            def fit(self, model, X_train, y_train, X_val, y_val):
                loss_fn = AsymmetricMSE(beta=0.4, alpha=0.6)
                optimizer = Adam(self.lr)
                regularizer = L2Regularizer(lambda_=self.l2_lambda)
                early_stopping = EarlyStopping(patience=self.patience, min_delta=1e-5, mode="min")

                history = {"train_loss": [], "val_loss": []}

                for epoch in range(1, 1001):
                    # Shuffle
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

                        # Forward
                        y_pred = model.forward(x_batch)
                        batch_loss = loss_fn.forward(y_pred, y_batch)

                        # Backward
                        grad_loss = loss_fn.backward(y_pred, y_batch)
                        model.backward(grad_loss)

                        # Regularization
                        trainable_params = model.get_trainable_layers()
                        regularizer.apply_to_gradients(trainable_params, y_batch)

                        # Gradient clipping
                        clip_gradients(model, self.clip_value)

                        # Update
                        optimizer.update(trainable_params)

                        epoch_losses.append(batch_loss)

                    # Training loss
                    train_loss = np.mean(epoch_losses)

                    # Add L2 penalty
                    if self.l2_lambda > 0:
                        weights_list = [layer.weights for layer in model.layers
                                        if hasattr(layer, 'weights') and layer.weights is not None]
                        l2_penalty = regularizer.compute_loss(weights_list, y_batch)
                        train_loss += l2_penalty

                    history["train_loss"].append(float(train_loss))

                    # Validation loss
                    y_pred_val = model.forward(X_val)
                    val_loss = loss_fn.forward(y_pred_val, y_val)
                    history["val_loss"].append(float(val_loss))

                    # Early stopping
                    should_stop = early_stopping.step(val_loss, epoch, model)
                    if should_stop:
                        break

                return model, history, early_stopping.best_epoch

        return SimpleTrainer()

    # Run grid search
    print("[4/4] Running full grid search...\n")

    grid_search = GridSearch(
        search_space=SEARCH_SPACE,
        model_factory=model_factory,
        trainer_factory=trainer_factory,
        input_size=input_size,
        seed=args.seed,
        enforce_funnel=True,
        trainer_param_map={"l2": "l2"},
        scoring="mae",
        verbose=True,
        y_scaler=y_scaler,
        inverse_fn=inverse_transform_target,
    )

    grid_search.fit(X_train, y_train_np, X_val, y_val_np)

    # Print results
    print(f"\n{'='*60}")
    print(f"GRID SEARCH RESULTS")
    print(f"{'='*60}")
    print(f"Best MAE: {grid_search.best_score:.4f}")
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

    # Top 10 results
    print(f"\n{'='*60}")
    print(f"TOP 10 RESULTS")
    print(f"{'='*60}")
    top_10 = grid_search.get_results(top_k=10)
    for rank, result in enumerate(top_10, start=1):
        print(f"\nRank {rank}:")
        print(f"   MAE: {result['mae']:.4f}")
        print(f"   Hidden layers: {result['hidden_layers']}")
        print(f"   Params: {result['params']}")
        print(f"   Best epoch: {result['best_epoch']}")

    # Save results
    grid_search.save_results(args.output)
    print(f"\n{'='*60}")
    print(f"Results saved to: {args.output}")
    print(f"{'='*60}")
