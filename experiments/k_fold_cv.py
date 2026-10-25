import numpy as np
import matplotlib.pyplot as plt

class KFoldCrossValidation:
    def __init__(
        self,
        k=5,
        shuffle=True,
        random_state=42
    ):
        if k < 2:
            raise ValueError("k fold minimal harus 2")

        self.k = k
        self.shuffle=shuffle
        self.random_state = random_state

        self.fold_results = []

    def split(self, n_samples):
        """
            Menghasilkan pasangan index (train_indices, val indices) untuk setiap fold
        """
        if n_samples < self.k:
            raise ValueError("Jumlah sampel harus >= jumlah fold")

        indices = np.arange(n_samples)

        if self.shuffle:
            rng = np.random.default_rng(self.random_state)
            rng.shuffle(indices)

        folds = np.array_split(indices, self.k)

        for idx in range(self.k):
            val_indices = folds[idx]

            # menggabungkan seluruh data kecuali yang sedang dipakai sebagai validation set
            train_indices = np.concatenate([
                folds[i]
                for i in range(self.k)
                if i != idx
            ])

            yield train_indices, val_indices

    @staticmethod
    def calculate_metrics(y_true, y_pred):
        # convert both variable to numpy array one dimension
        y_true = np.asarray(y_true).reshape(-1)
        y_pred = np.asarray(y_pred).reshape(-1)

        if y_true.shape != y_pred.shape:
            raise ValueError("Dimensi y_true dan y_pred tidak sama")

        errors = y_true - y_pred

        mae = np.mean(np.abs(errors))

        rmse = np.sqrt(np.mean(errors ** 2))

        nonzero = y_true != 0
        # Mape tidak terdefinisi untuk nilai target 0 (karena perhitungan MAPE melibatkan pembagian dengan nilai target tersebut)
        if np.any(nonzero):
            mape = np.mean(np.abs(errors[nonzero] / y_true[nonzero])) * 100
        else:
            mape = float("nan")

        ss_res = np.sum(errors ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

        if ss_tot == 0:
            r2 = float("nan")
        else:
            r2 = 1 - (ss_res / ss_tot)

        return {
            "mae": float(mae),
            "rmse": float(rmse),
            "mape": float(mape),
            "r2": float(r2),
        }

    @staticmethod
    def _denormalize(y_scaled, y_scaler, inverse_fn):
        if y_scaler is None or inverse_fn is None:
            return y_scaled
        flat = np.asarray(y_scaled).reshape(-1).tolist()
        return np.array(
            inverse_fn(flat, y_scaler), dtype=np.float64
        ).reshape(np.asarray(y_scaled).shape)

    def evaluate(
            self,
            X,
            y,
            model_factory,
            trainer_factory,
            model_params,
            trainer_params,
            y_scaler=None,
            inverse_fn=None,
        ):
            """
            Setiap fold membuat model dan Trainer baru.

            model_factory:
                Contoh: MLP

            trainer_factory:
                Class Trainer yang menyediakan fit(...).

            y_scaler, inverse_fn:
                Dipakai untuk mengembalikan target & prediksi ke skala asli
                sebelum menghitung metrik.
            """
            X = np.asarray(X)
            y = np.asarray(y)

            if len(X) != len(y):
                raise ValueError(
                    "Jumlah baris X dan y harus sama"
                )

            self.fold_results = []

            for fold_number, (train_idx, val_idx) in enumerate(
                self.split(len(X)),
                start=1,
            ):
                X_train = X[train_idx]
                y_train = y[train_idx]

                X_val = X[val_idx]
                y_val = y[val_idx]

                # Buat model baru agar bobot antarfold tidak terbawa.
                model = model_factory(**model_params)
                trainer = trainer_factory(**trainer_params)

                model, history, best_epoch = trainer.fit(
                    model=model,
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                )

                y_pred = model.forward(X_val)

                # Kembalikan ke skala asli sebelum menghitung metrik.
                y_val_eval = self._denormalize(y_val, y_scaler, inverse_fn)
                y_pred_eval = self._denormalize(y_pred, y_scaler, inverse_fn)

                metrics = self.calculate_metrics(y_val_eval, y_pred_eval)

                result = {
                    "fold": fold_number,
                    "train_samples": len(train_idx),
                    "validation_samples": len(val_idx),
                    "best_epoch": best_epoch,
                    "history": history,
                    "y_val": y_val_eval,
                    "y_pred": y_pred_eval,
                    **metrics,
                }

                self.fold_results.append(result)

                print(
                    f"Fold {fold_number}: "
                    f"MAE={metrics['mae']:.4f}, "
                    f"RMSE={metrics['rmse']:.4f}, "
                    f"MAPE={metrics['mape']:.2f}%, "
                    f"R2={metrics['r2']:.4f}"
                )

            return self.fold_results

    def summary(self):
        if not self.fold_results:
            raise ValueError(
                "Jalankan evaluate() terlebih dahulu"
            )
        metric_names = ["mae", "rmse", "mape", "r2"]
        summary = {}
        for metric in metric_names:
            values = np.array([
                row[metric]
                for row in self.fold_results
            ], dtype=float)
            valid_values = values[np.isfinite(values)]
            summary[metric] = {
                "mean": (
                    float(np.mean(valid_values))
                    if len(valid_values) else float("nan")
                ),
                "std": (
                    float(np.std(valid_values, ddof=1))
                    if len(valid_values) > 1 else float("nan")
                ),
            }
        return summary


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
    from core.config.config import configs
    from training.early_stopping import EarlyStopping
    from training.gradient_clipping import clip_gradients
    from core.regularizers.l2 import L2Regularizer

    parser = argparse.ArgumentParser(description="K-Fold Cross Validation")
    parser.add_argument("--capacity", type=str, default="900", choices=["900", "1300", "2200"],
                        help="Capacity VA (default: 900)")
    parser.add_argument("--k", type=int, default=5,
                        help="Number of folds (default: 5)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--no-shuffle", action="store_true",
                        help="Disable shuffling")

    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"K-FOLD CROSS VALIDATION")
    print(f"{'='*60}")
    print(f"Capacity: {args.capacity} VA")
    print(f"K-Folds: {args.k}")
    print(f"Shuffle: {not args.no_shuffle}")
    print(f"Random Seed: {args.seed}")
    print(f"{'='*60}\n")

    # Load configuration
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

    # Scale data
    print("[2/3] Scaling data...")
    x_scaler = fit_standard_scaler(x_data)
    x_scaled = transform_standard(x_data, x_scaler)

    y_scaler = fit_minmax_scaler(y_data)
    y_scaled = transform_minmax(y_data, y_scaler)

    # Convert to numpy
    X = np.array(x_scaled, dtype=np.float32)
    y = np.array(y_scaled, dtype=np.float32).reshape(-1, 1)

    input_size = len(feature_columns)
    hidden_layers = cfg["hidden_layers"]

    print(f"   Input size: {input_size}")
    print(f"   Hidden layers: {hidden_layers}")

    # Define factories
    def model_factory(input_size, hidden_layers):
        layer_sizes = [input_size] + hidden_layers + [1]
        return MLP(layer_sizes)

    def trainer_factory(learning_rate, batch_size, l2_lambda, patience, clip_value):
        class SimpleTrainer:
            def __init__(self):
                self.lr = learning_rate
                self.batch_size = batch_size
                self.l2_lambda = l2_lambda
                self.patience = patience
                self.clip_value = clip_value

            def fit(self, model, X_train, y_train, X_val, y_val):
                loss_fn = AsymmetricMSE(alpha=cfg["asymmetric_alpha"], beta=cfg["asymmetric_beta"])
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

    # Model and trainer parameters
    model_params = {
        "input_size": input_size,
        "hidden_layers": hidden_layers
    }

    trainer_params = {
        "learning_rate": cfg["learning_rate"],
        "batch_size": cfg["batch_size"],
        "l2_lambda": cfg["l2_lambda"],
        "patience": cfg["patience"],
        "clip_value": cfg["clip_value"]
    }

    print(f"\nTrainer parameters:")
    for key, value in trainer_params.items():
        print(f"   {key}: {value}")

    # Run K-Fold CV
    print(f"\n[3/3] Running {args.k}-Fold Cross Validation...\n")

    kfold = KFoldCrossValidation(
        k=args.k,
        shuffle=not args.no_shuffle,
        random_state=args.seed
    )

    kfold.evaluate(
        X=X,
        y=y,
        model_factory=model_factory,
        trainer_factory=trainer_factory,
        model_params=model_params,
        trainer_params=trainer_params,
        y_scaler=y_scaler,
        inverse_fn=inverse_transform_target,
    )

    output_dir = f"results/kfold_{args.capacity}VA"
    os.makedirs(output_dir, exist_ok=True)

    for result in kfold.fold_results:
        plt.figure(figsize=(8, 5))
        plt.plot(result['history']['train_loss'], label='Train Loss')
        plt.plot(result['history']['val_loss'], label='Validation Loss')
        plt.axvline(result['best_epoch'], color='r', linestyle='--', alpha=0.5, label=f"Best Epoch ({result['best_epoch']})")
        plt.title(f"Fold {result['fold']} - Loss Curve")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/fold_{result['fold']}.png", dpi=150)
        plt.close()

    print(f"\nVisualizations saved in: {output_dir}/")

    # Print summary
    print(f"\n{'='*60}")
    print(f"K-FOLD CROSS VALIDATION SUMMARY")
    print(f"{'='*60}")

    summary = kfold.summary()

    print(f"\nMAE:")
    print(f"   Mean: {summary['mae']['mean']:.4f} hari")
    print(f"   Std:  {summary['mae']['std']:.4f} hari")

    print(f"\nRMSE:")
    print(f"   Mean: {summary['rmse']['mean']:.4f} hari")
    print(f"   Std:  {summary['rmse']['std']:.4f} hari")

    print(f"\nMAPE:")
    print(f"   Mean: {summary['mape']['mean']:.2f} %")
    print(f"   Std:  {summary['mape']['std']:.2f} %")

    print(f"\nR²:")
    print(f"   Mean: {summary['r2']['mean']:.4f}")
    print(f"   Std:  {summary['r2']['std']:.4f}")

    print(f"\n{'='*60}")
    print(f"All folds completed successfully!")
    print(f"{'='*60}")
