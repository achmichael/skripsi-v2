import os
import json
import numpy as np

from core.config.config import configs
from preprocessing.preprocess import preprocess

from feature_extraction.feature_extraction import extractor
from transformation.target import (
    fit_minmax_scaler,
    transform_minmax
)

from transformation.inverse_transform import inverse_transform_target
from utils.helper import selective_feature_scaling, create_model
from .early_stopping import EarlyStopping
from .gradient_clipping import clip_gradients
from core.optimizers.adam import Adam
from experiments.k_fold_cv import KFoldCrossValidation
from visualization.plots import save_loss_curve

class Trainer:
    def __init__(
        self,
        model,
        loss,
        optimizer
    ):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer

    def train_batch(self, x, y):
        y_pred = self.model.forward(x)

        loss_value = self.loss.forward(y_pred, y)

        grad_loss = self.loss.backward(y_pred, y)

        self.model.backward(grad_loss)

        trainable_params = self.model.get_trainable_layers()

        self.optimizer.update(trainable_params)

        return loss_value

    def evaluate(self, x, y):
        """Evaluasi model tanpa update parameter."""
        y_pred = self.model.forward(x)
        loss_value = self.loss.forward(y_pred, y)
        return loss_value, y_pred

    @staticmethod
    def _train_fold(
        model,
        loss,
        x_train_np,
        y_train_np,
        x_val_np,
        y_val_np,
        cfg,
        epochs,
        verbose
    ):
        optimizer = Adam(learning_rate=cfg["learning_rate"])
        trainer = Trainer(model, loss, optimizer)
        early_stopping = EarlyStopping(
            patience=cfg["patience"],
            min_delta=cfg["min_delta"],
            mode="min"
        )

        batch_size = cfg["batch_size"]
        clip_value = cfg["clip_value"]

        history = {
            "train_loss": [],
            "val_loss": []
        }

        if verbose:
            print(f"\n{'Epoch':<8} {'Train Loss':<15} {'Val Loss':<15} {'Status':<20}")
            print("-" * 60)

        for epoch in range(1, epochs + 1):
            # Shuffle training data
            indices = np.random.permutation(len(x_train_np))
            x_train_shuffled = x_train_np[indices]
            y_train_shuffled = y_train_np[indices]

            # Mini-batch training
            epoch_losses = []
            num_batches = (len(x_train_np) + batch_size - 1) // batch_size

            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, len(x_train_np))

                x_batch = x_train_shuffled[start_idx:end_idx]
                y_batch = y_train_shuffled[start_idx:end_idx]

                # Forward, backward
                y_pred = model.forward(x_batch)
                batch_loss = loss.forward(y_pred, y_batch)

                # Backward pass
                grad_loss = loss.backward(y_pred, y_batch)
                model.backward(grad_loss)

                trainable_params = model.get_trainable_layers()

                # Gradient clipping
                clip_gradients(model, clip_value)

                # Update parameters
                optimizer.update(trainable_params)

                epoch_losses.append(batch_loss)

            # Average training loss
            train_loss = np.mean(epoch_losses)

            history["train_loss"].append(float(train_loss))

            # Validation loss
            val_loss, _ = trainer.evaluate(x_val_np, y_val_np)
            history["val_loss"].append(float(val_loss))

            # Early stopping check
            should_stop = early_stopping.step(val_loss, epoch, model)

            # Print progress
            if verbose and (epoch % 10 == 0 or epoch == 1 or should_stop):
                status = "Early Stop!" if should_stop else ""
                print(f"{epoch:<8} {train_loss:<15.6f} {val_loss:<15.6f} {status:<20}")

            if should_stop:
                if verbose:
                    print(f"\nEarly stopping triggered at epoch {epoch}")
                    print(f"Best epoch: {early_stopping.best_epoch}")
                    print(f"Best val loss: {early_stopping.best_value:.6f}")
                break

        return history, early_stopping

    @staticmethod
    def fit(
        model,
        loss,
        capacity: str,
        epochs: int = 1000,
        k_folds: int = 5,
        verbose: bool = True
    ):
        if capacity not in configs["models_config"]:
            raise ValueError(f"Capacity {capacity} not found in config")

        cfg = configs["models_config"][capacity]

        if not os.path.exists(cfg["dataset_path"]):
            raise FileNotFoundError(f"Dataset tidak ditemukan: {cfg['dataset_path']}")

        os.makedirs(os.path.dirname(cfg["model_path"]), exist_ok=True)
        os.makedirs(cfg["metrics_dir"], exist_ok=True)

        if verbose:
            print(f"\n{'='*60}")
            print(f"Training Model {capacity} VA")
            print(f"{'='*60}")
            print(f"Dataset: {cfg['dataset_path']}")
            print(f"Hidden Layers: {cfg['hidden_layers']}")
            print(f"Learning Rate: {cfg['learning_rate']}")
            print(f"Batch Size: {cfg['batch_size']}")
            print(f"Asymmetric Alpha: {cfg['asymmetric_alpha']}")
            print(f"Validation: {k_folds}-fold cross-validation")

        if verbose:
            print("\n[1/6] Preprocessing data...")
        rows = preprocess(cfg["dataset_path"])

        if verbose:
            print("[2/6] Feature extraction...")
        x_data, y_data, feature_columns, target_col = extractor(rows, capacity)

        if verbose:
            print(f"   Features: {len(feature_columns)}")
            print(f"   Samples: {len(x_data)}")

        no_scaling_cols = set(configs["preprocessing"]["no_scaling_cols"])
        use_log = cfg["use_log_transform"]

        # 3. K-Fold Cross Validation
        if verbose:
            print(f"[3/6] {k_folds}-fold cross-validation...")

        kfold = KFoldCrossValidation(
            k=k_folds,
            shuffle=True,
            random_state=42
        )

        fold_results = []
        best_fold = None

        for fold_number, (train_idx, val_idx) in enumerate(
            kfold.split(len(x_data)),
            start=1,
        ):
            x_train = [x_data[i] for i in train_idx]
            y_train = [y_data[i] for i in train_idx]
            x_val = [x_data[i] for i in val_idx]
            y_val = [y_data[i] for i in val_idx]

            # 4. Feature Scaling (fit hanya pada training fold)
            x_train_scaled, x_val_scaled, _, x_scaler = selective_feature_scaling(
                x_train=x_train,
                x_val=x_val,
                x_test=x_val,
                feature_columns=feature_columns,
                encoding_cols=no_scaling_cols
            )

            # 5. Target Scaling
            y_scaler = fit_minmax_scaler(y_train)
            y_train_scaled = transform_minmax(y_train, y_scaler)
            y_val_scaled = transform_minmax(y_val, y_scaler)

            x_train_np = np.array(x_train_scaled, dtype=np.float32)
            y_train_np = np.array(y_train_scaled, dtype=np.float32).reshape(-1, 1)
            x_val_np = np.array(x_val_scaled, dtype=np.float32)
            y_val_np = np.array(y_val_scaled, dtype=np.float32).reshape(-1, 1)

            # 6. Training fold (model baru agar bobot antarfold tidak terbawa)
            fold_model = create_model(capacity, model.layer_sizes)
            np.random.seed(42 + fold_number)

            history, early_stopping = Trainer._train_fold(
                model=fold_model,
                loss=loss,
                x_train_np=x_train_np,
                y_train_np=y_train_np,
                x_val_np=x_val_np,
                y_val_np=y_val_np,
                cfg=cfg,
                epochs=epochs,
                verbose=verbose
            )

            # Evaluasi fold pada skala asli
            y_pred_original = inverse_transform_target(
                fold_model.forward(x_val_np).flatten().tolist(),
                y_scaler,
                use_log
            )
            y_true_original = inverse_transform_target(
                y_val_np.flatten().tolist(),
                y_scaler,
                use_log
            )
            metrics = KFoldCrossValidation.calculate_metrics(
                y_true_original,
                y_pred_original
            )

            fold_result = {
                "fold": fold_number,
                "train_samples": len(train_idx),
                "validation_samples": len(val_idx),
                "best_epoch": early_stopping.best_epoch,
                "best_val_loss": float(early_stopping.best_value),
                "history": history,
                **metrics,
            }
            fold_results.append(fold_result)

            save_loss_curve(
                history=history,
                save_path=os.path.join(
                    cfg["metrics_dir"],
                    f"loss_curve_fold{fold_number}.png"
                ),
                model_type=f"{capacity}VA fold {fold_number}"
            )

            if verbose:
                print(
                    f"\nFold {fold_number}: "
                    f"MAE={metrics['mae']:.4f} hari, "
                    f"RMSE={metrics['rmse']:.4f} hari, "
                    f"MAPE={metrics['mape']:.2f}%, "
                    f"R2={metrics['r2']:.4f}"
                )

            # Simpan fold terbaik sebagai model final
            if best_fold is None or early_stopping.best_value < best_fold["best_val_loss"]:
                best_fold = {
                    **fold_result,
                    "model": fold_model,
                    "x_scaler": x_scaler,
                    "y_scaler": y_scaler,
                    "data": {
                        "x_train": x_train_np,
                        "y_train": y_train_np,
                        "x_val": x_val_np,
                        "y_val": y_val_np,
                        "x_test": x_val_np,
                        "y_test": y_val_np,
                    },
                }

        kfold.fold_results = fold_results
        cv_summary = kfold.summary()

        model = best_fold["model"]
        history = best_fold["history"]
        x_scaler = best_fold["x_scaler"]
        y_scaler = best_fold["y_scaler"]

        if verbose:
            print(f"\nBest fold: {best_fold['fold']} "
                  f"(val loss={best_fold['best_val_loss']:.6f})")

        # Save model weights
        model_weights = []
        for layer in model.layers:
            if hasattr(layer, 'weights') and hasattr(layer, 'biases'):
                model_weights.append({
                    'weights': layer.weights.tolist(),
                    'biases': layer.biases.tolist()
                })
            elif hasattr(layer, 'weights'):
                model_weights.append({
                    'weights': layer.weights.tolist()
                })

        model_data = {
            'layer_sizes': model.layer_sizes,
            'weights': model_weights,
            'capacity': capacity
        }

        with open(cfg["model_path"], 'w') as f:
            json.dump(model_data, f, indent=2)

        if verbose:
            print(f"\nModel saved to: {cfg['model_path']}")

        # Save scalers
        scalers_path = os.path.join(cfg['metrics_dir'], 'scalers.json')
        scalers_data = {
            'x_scaler': x_scaler,
            'y_scaler': y_scaler,
            'use_log_transform': use_log
        }
        with open(scalers_path, 'w') as f:
            json.dump(scalers_data, f, indent=2)

        if verbose:
            print(f"Scalers saved to: {scalers_path}")

        # Prepare result
        result = {
            'model': model,
            'history': history,
            'scalers': {
                'x_scaler': x_scaler,
                'y_scaler': y_scaler,
                'use_log': use_log
            },
            'data': best_fold["data"],
            'metadata': {
                'capacity': capacity,
                'feature_columns': feature_columns,
                'target_col': target_col,
                'use_log_transform': use_log,
                'epochs_trained': len(history['train_loss']),
                'best_epoch': best_fold['best_epoch'],
                'best_val_loss': best_fold['best_val_loss'],
                'k_folds': k_folds,
                'cv_summary': cv_summary,
                'cv_folds': [
                    {
                        key: fold[key]
                        for key in (
                            "fold",
                            "train_samples",
                            "validation_samples",
                            "best_epoch",
                            "best_val_loss",
                            "mae",
                            "rmse",
                            "mape",
                            "r2",
                        )
                    }
                    for fold in fold_results
                ],
            }
        }

        if verbose:
            print(f"\n{'='*60}")
            print("Training completed!")
            print(f"{'='*60}\n")

        return result
