import json
import numpy as np
import pandas as pd
import argparse
import os

from utils.helper import create_model, transform_selective
from preprocessing.preprocess import preprocess
from feature_extraction.feature_extraction import extractor
from transformation.inverse_transform import inverse_transform_target
from core.config.config import configs


def load_model(model_path, capacity):
    """
    Load trained model dari file JSON.
    
    Args:
        model_path: Path ke model JSON
        capacity: Capacity VA
        
    Returns:
        model: Loaded model dengan weights
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model tidak ditemukan: {model_path}")
    
    with open(model_path, 'r') as f:
        model_data = json.load(f)
    
    # Create model
    layer_sizes = model_data['layer_sizes']
    model = create_model(capacity, layer_sizes)
    
    # Load weights
    weight_index = 0
    for layer in model.layers:
        if hasattr(layer, 'weights'):
            weight_data = model_data['weights'][weight_index]
            layer.weights = np.array(weight_data['weights'], dtype=np.float32)
            if 'biases' in weight_data:
                layer.biases = np.array(weight_data['biases'], dtype=np.float32)
            weight_index += 1
    
    return model


def load_scalers(metrics_dir):
    """
    Load scalers dari file training results.
    
    Args:
        metrics_dir: Directory tempat metrics disimpan
        
    Returns:
        dict: x_scaler dan y_scaler
    """
    scalers_path = os.path.join(metrics_dir, "scalers.json")
    
    if not os.path.exists(scalers_path):
        raise FileNotFoundError(
            f"Scalers tidak ditemukan: {scalers_path}\n"
            "Silakan train model terlebih dahulu dengan train.py"
        )
    
    with open(scalers_path, 'r') as f:
        scalers = json.load(f)
    
    return scalers


def predict(model, x_data, x_scaler, y_scaler, use_log=False):
    """
    Predict durasi token dari input data.
    
    Args:
        model: Trained model
        x_data: Input features (list of list)
        x_scaler: Feature scaler
        y_scaler: Target scaler
        use_log: Whether log transform was used
        
    Returns:
        list: Predictions dalam skala asli (hari)
    """
    # Scale features
    x_scaled = transform_selective(x_data, x_scaler)
    x_np = np.array(x_scaled, dtype=np.float32)
    
    # Predict
    y_pred_scaled = model.forward(x_np)
    y_pred_scaled_list = y_pred_scaled.flatten().tolist()
    
    # Inverse transform
    y_pred_original = inverse_transform_target(y_pred_scaled_list, y_scaler, use_log)
    
    return y_pred_original


def predict_from_csv(csv_path, capacity, verbose=True):
    """
    Predict dari file CSV baru.
    
    Args:
        csv_path: Path ke CSV file
        capacity: Capacity VA (900, 1300, 2200)
        verbose: Print progress
        
    Returns:
        pd.DataFrame: DataFrame dengan kolom prediksi
    """
    if capacity not in configs["models_config"]:
        raise ValueError(f"Capacity {capacity} tidak tersedia")
    
    cfg = configs["models_config"][capacity]
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Prediction - Model {capacity} VA")
        print(f"{'='*60}\n")
    
    # Load model
    if verbose:
        print(f"[1/6] Loading model from {cfg['model_path']}...")
    model = load_model(cfg['model_path'], capacity)
    
    # Load scalers
    if verbose:
        print(f"[2/6] Loading scalers from {cfg['metrics_dir']}...")
    scalers = load_scalers(cfg['metrics_dir'])
    
    # Preprocess data
    if verbose:
        print(f"[3/6] Preprocessing data from {csv_path}...")
    df = preprocess(csv_path)
    
    # Extract features
    if verbose:
        print(f"[4/6] Extracting features...")
    x_data, _, feature_columns, _ = extractor(df, capacity)
    
    if verbose:
        print(f"   Features: {len(feature_columns)}")
        print(f"   Samples: {len(x_data)}")
    
    # Predict
    if verbose:
        print(f"[5/6] Making predictions...")
    predictions = predict(
        model=model,
        x_data=x_data,
        x_scaler=scalers['x_scaler'],
        y_scaler=scalers['y_scaler'],
        use_log=cfg['use_log_transform']
    )
    
    # Add predictions to dataframe
    df['Prediksi_Durasi_Token_Hari'] = predictions
    df['Prediksi_Durasi_Token_Hari'] = df['Prediksi_Durasi_Token_Hari'].round(2)
    
    if verbose:
        print(f"[6/6] Prediction completed!")
        print(f"\nPrediction Statistics:")
        print(f"   Mean: {np.mean(predictions):.2f} hari")
        print(f"   Median: {np.median(predictions):.2f} hari")
        print(f"   Min: {np.min(predictions):.2f} hari")
        print(f"   Max: {np.max(predictions):.2f} hari")
        print(f"\n{'='*60}\n")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Prediction Durasi Token Listrik menggunakan trained model"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path ke CSV file input"
    )
    parser.add_argument(
        "--capacity",
        required=True,
        choices=["900", "1300", "2200"],
        help="Capacity VA model yang akan digunakan"
    )
    parser.add_argument(
        "--output",
        help="Path untuk menyimpan hasil prediksi (optional)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print detailed progress"
    )
    
    args = parser.parse_args()
    
    try:
        # Predict
        result_df = predict_from_csv(
            csv_path=args.input,
            capacity=args.capacity,
            verbose=args.verbose
        )
        
        # Save output if specified
        if args.output:
            result_df.to_csv(args.output, index=False, encoding='utf-8-sig')
            print(f"Results saved to: {args.output}")
        else:
            # Print first few predictions
            print("\nFirst 5 predictions:")
            print(result_df[['Prediksi_Durasi_Token_Hari']].head())
        
    except Exception as e:
        print(f"\nError during prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
