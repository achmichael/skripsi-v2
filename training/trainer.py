import os
import json
import numpy as np

from core.config.config import configs
from preprocessing.preprocess import (
    preprocess,
    train_test_split
)

from feature_extraction.feature_extraction import extractor
from transformation.target import (
    fit_minmax_scaler,
    transform_minmax
)

from transformation.inverse_transform import inverse_transform_target
from utils.helper import selective_feature_scaling
from .early_stopping import EarlyStopping
from .gradient_clipping import clip_gradients
from core.regularizers.l2 import L2Regularizer

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
    def fit(
        model,
        loss,
        optimizer,
        capacity: str,
        epochs: int = 1000,
        val_split: float = 0.15,
        test_split: float = 0.15,
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

        if verbose:
            print("\n[1/8] Preprocessing data...")
        rows = preprocess(cfg["dataset_path"])

        if verbose:
            print("[2/8] Feature extraction...")
        x_data, y_data, feature_columns, target_col = extractor(rows, capacity)

        if verbose:
            print(f"   Features: {len(feature_columns)}")
            print(f"   Samples: {len(x_data)}")

        # 3. Train-Test Split
        if verbose:
            print("[3/8] Train-test split...")
        x_train, y_train, x_test, y_test = train_test_split(
            x_data=x_data,
            y_data=y_data,
            test_ratio=test_split,
            seed=42
        )

        # 4. Train-Validation Split
        if verbose:
            print("[4/8] Train-validation split...")
        val_size = int(len(x_train) * val_split)
        x_val = x_train[:val_size]
        y_val = y_train[:val_size]
        x_train = x_train[val_size:]
        y_train = y_train[val_size:]

        if verbose:
            print(f"   Train: {len(x_train)}, Val: {len(x_val)}, Test: {len(x_test)}")

        no_scaling_cols = set(configs["preprocessing"]["no_scaling_cols"])

        # 5. Feature Scaling
        if verbose:
            print("[5/8] Feature scaling (StandardScaler)...")

        x_train_scaled, x_val_scaled, x_test_scaled, x_scaler = (
            selective_feature_scaling(
                x_train=x_train,
                x_test=x_test,
                x_val=x_val,
                feature_columns=feature_columns,
                encoding_cols=no_scaling_cols
            )
        )

        # 6. Target Scaling
        if verbose:
            print("[6/8] Target scaling (MinMaxScaler)...")
        y_scaler = fit_minmax_scaler(y_train)
        y_train_scaled = transform_minmax(y_train, y_scaler)
        y_val_scaled = transform_minmax(y_val, y_scaler)
        y_test_scaled = transform_minmax(y_test, y_scaler)

        # Convert to numpy arrays
        x_train_np = np.array(x_train_scaled, dtype=np.float32)
        y_train_np = np.array(y_train_scaled, dtype=np.float32).reshape(-1, 1)
        x_val_np = np.array(x_val_scaled, dtype=np.float32)
        y_val_np = np.array(y_val_scaled, dtype=np.float32).reshape(-1, 1)
        x_test_np = np.array(x_test_scaled, dtype=np.float32)
        y_test_np = np.array(y_test_scaled, dtype=np.float32).reshape(-1, 1)

        # 7. Training Setup
        if verbose:
            print("[7/8] Training setup...")

        # Initialize L2 regularizer
        l2_lambda = cfg.get("l2_lambda", 0.0)
        regularizer = L2Regularizer(lambda_=l2_lambda)

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

        # 8. Training Loop
        if verbose:
            print("[8/8] Training model...")
            print(f"\n{'Epoch':<8} {'Train Loss':<15} {'Val Loss':<15} {'Status':<20}")
            print("-" * 60)

        # Set random seed for reproducibility
        np.random.seed(42)

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

                # Apply L2 regularization to gradients
                trainable_params = model.get_trainable_layers()
                regularizer.apply_to_gradients(trainable_params, y_batch)

                # Gradient clipping
                clip_gradients(model, clip_value)

                # Update parameters
                optimizer.update(trainable_params)

                epoch_losses.append(batch_loss)

            # Average training loss
            train_loss = np.mean(epoch_losses)

            # Add L2 regularization penalty to loss for tracking
            if l2_lambda > 0:
                weights_list = [layer.weights for layer in model.layers if hasattr(layer, 'weights') and layer.weights is not None]
                l2_penalty = regularizer.compute_loss(weights_list, y_batch)
                train_loss_with_reg = train_loss + l2_penalty
            else:
                train_loss_with_reg = train_loss

            history["train_loss"].append(float(train_loss_with_reg))

            # Validation loss
            val_loss, _ = trainer.evaluate(x_val_np, y_val_np)
            history["val_loss"].append(float(val_loss))

            # Early stopping check
            should_stop = early_stopping.step(val_loss, epoch, model)

            # Print progress
            if verbose and (epoch % 10 == 0 or epoch == 1 or should_stop):
                status = "Early Stop!" if should_stop else ""
                print(f"{epoch:<8} {train_loss_with_reg:<15.6f} {val_loss:<15.6f} {status:<20}")

            if should_stop:
                if verbose:
                    print(f"\nEarly stopping triggered at epoch {epoch}")
                    print(f"Best epoch: {early_stopping.best_epoch}")
                    print(f"Best val loss: {early_stopping.best_value:.6f}")
                break

        # Save model weights
        model_weights = []
        for layer in model.layers:
            if hasattr(layer, 'weights') and hasattr(layer, 'biases'):
                model_weights.append({
                    'weights': layer.weights.tolist(),
                    'biases': layer.biases.tolist()
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
            'use_log_transform': cfg['use_log_transform']
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
                'use_log': cfg['use_log_transform']
            },
            'data': {
                'x_train': x_train_np,
                'y_train': y_train_np,
                'x_val': x_val_np,
                'y_val': y_val_np,
                'x_test': x_test_np,
                'y_test': y_test_np
            },
            'metadata': {
                'capacity': capacity,
                'feature_columns': feature_columns,
                'target_col': target_col,
                'use_log_transform': cfg['use_log_transform'],
                'epochs_trained': len(history['train_loss']),
                'best_epoch': early_stopping.best_epoch,
                'best_val_loss': early_stopping.best_value
            }
        }

        if verbose:
            print(f"\n{'='*60}")
            print("Training completed!")
            print(f"{'='*60}\n")

        return result
