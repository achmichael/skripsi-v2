import os
import json
import numpy as np

from evaluations.metrics import rmse, mae, mape, r2
from transformation.inverse_transform import inverse_transform_target
from visualization.plots import (
    save_loss_curve,
    save_prediction_scatter,
    plot_error_distribution,
    save_metrics_json
)


class Evaluator:
    def __init__(self, model, scalers, metadata):
        """
        Evaluator untuk mengevaluasi model dan generate visualisasi.
        
        Args:
            model: Trained model
            scalers: Dict berisi x_scaler, y_scaler, dan use_log
            metadata: Dict berisi informasi capacity, feature_columns, dll
        """
        self.model = model
        self.x_scaler = scalers['x_scaler']
        self.y_scaler = scalers['y_scaler']
        self.use_log = scalers.get('use_log_transform', metadata.get('use_log_transform', False))
        self.metadata = metadata
        self.capacity = metadata['capacity']
        
    def predict(self, x_scaled):
        """Prediksi dari data yang sudah di-scale."""
        y_pred_scaled = self.model.forward(x_scaled)
        return y_pred_scaled
    
    def evaluate_set(self, x_scaled, y_scaled, set_name="test"):
        """
        Evaluasi pada satu set data (train/val/test).
        
        Returns:
            dict: metrics dan predictions
        """
        # Prediksi (scaled)
        y_pred_scaled = self.predict(x_scaled)
        
        # Inverse transform ke skala asli
        y_pred_scaled_list = y_pred_scaled.flatten().tolist()
        y_scaled_list = y_scaled.flatten().tolist()
        
        y_pred_original = inverse_transform_target(y_pred_scaled_list, self.y_scaler, self.use_log)
        y_true_original = inverse_transform_target(y_scaled_list, self.y_scaler, self.use_log)
        
        # Hitung metrics
        y_true_np = np.array(y_true_original)
        y_pred_np = np.array(y_pred_original)
        
        metrics = {
            'rmse': rmse(y_true_np, y_pred_np),
            'mae': mae(y_true_np, y_pred_np),
            'mape': mape(y_true_np, y_pred_np),
            'r2': r2(y_true_np, y_pred_np),
            'set_name': set_name,
            'n_samples': len(y_true_np)
        }
        
        return {
            'metrics': metrics,
            'y_true': y_true_original,
            'y_pred': y_pred_original
        }
    
    def evaluate_all(self, data, save_dir=None, verbose=True):
        """
        Evaluasi pada train, validation, dan test set.
        Generate visualisasi dan save metrics.
        
        Args:
            data: Dict berisi x_train, y_train, x_val, y_val, x_test, y_test
            save_dir: Directory untuk menyimpan hasil
            verbose: Print hasil evaluasi
            
        Returns:
            dict: Hasil evaluasi lengkap
        """
        results = {}
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Evaluating Model {self.capacity} VA")
            print(f"{'='*60}\n")
        
        # Evaluate Train Set
        if verbose:
            print("[1/3] Evaluating training set...")
        train_result = self.evaluate_set(data['x_train'], data['y_train'], "train")
        results['train'] = train_result
        
        # Evaluate Validation Set
        if verbose:
            print("[2/3] Evaluating validation set...")
        val_result = self.evaluate_set(data['x_val'], data['y_val'], "validation")
        results['val'] = val_result
        
        # Evaluate Test Set
        if verbose:
            print("[3/3] Evaluating test set...")
        test_result = self.evaluate_set(data['x_test'], data['y_test'], "test")
        results['test'] = test_result
        
        # Print metrics
        if verbose:
            print(f"\n{'='*60}")
            print("Evaluation Results")
            print(f"{'='*60}\n")
            
            for set_name in ['train', 'val', 'test']:
                metrics = results[set_name]['metrics']
                print(f"{set_name.upper()} SET ({metrics['n_samples']} samples):")
                print(f"  RMSE: {metrics['rmse']:.4f} hari")
                print(f"  MAE:  {metrics['mae']:.4f} hari")
                print(f"  MAPE: {metrics['mape']:.4f} %")
                print(f"  R²:   {metrics['r2']:.4f}")
                print()
        
        # Save results jika save_dir diberikan
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            
            if verbose:
                print("Saving evaluation results and visualizations...")
            
            # Save metrics JSON
            metrics_summary = {
                'capacity': self.capacity,
                'train': results['train']['metrics'],
                'validation': results['val']['metrics'],
                'test': results['test']['metrics'],
                'metadata': self.metadata
            }
            
            metrics_path = os.path.join(save_dir, f"metrics_{self.capacity}va.json")
            save_metrics_json(metrics_summary, metrics_path)
            
            # Save prediction scatter untuk test set
            scatter_path = os.path.join(save_dir, f"prediction_scatter_{self.capacity}va.png")
            save_prediction_scatter(
                y_actual=results['test']['y_true'],
                y_predicted=results['test']['y_pred'],
                save_path=scatter_path,
                model_type=f"{self.capacity}VA",
                label="durasi token (hari)"
            )
            
            # Save error distribution untuk test set
            error_dist_path = os.path.join(save_dir, f"error_distribution_{self.capacity}va.png")
            plot_error_distribution(
                y_true=results['test']['y_true'],
                y_pred=results['test']['y_pred'],
                mae_value=results['test']['metrics']['mae'],
                save_path=error_dist_path,
                show=False
            )
            
            if verbose:
                print(f"\nResults saved to: {save_dir}")
        
        if verbose:
            print(f"\n{'='*60}")
            print("Evaluation completed!")
            print(f"{'='*60}\n")
        
        return results
    
    def save_training_curves(self, history, save_path):
        """Save training loss curves."""
        save_loss_curve(
            history=history,
            save_path=save_path,
            model_type=f"{self.capacity}VA"
        )
