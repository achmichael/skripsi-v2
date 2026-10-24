import os
import sys
import argparse

from core.config.config import configs
from core.losses.mse import MSE
from core.losses.asymmetric_mse import AsymmetricMSE
from core.optimizers.adam import Adam
from utils.helper import create_model

from training.trainer import Trainer
from evaluations.evaluator import Evaluator
from postprocessing.xai.xai import IntegratedGradients

import json
import numpy as np


def train_model(capacity, use_asymmetric=True, verbose=True):
    if capacity not in configs["models_config"]:
        raise ValueError(f"Capacity {capacity} tidak tersedia")

    cfg = configs["models_config"][capacity]

    # Build model architecture
    num_features = len(cfg["features"])
    layer_sizes = [num_features] + cfg["hidden_layers"] + [1]

    model = create_model(capacity, layer_sizes)

    # Setup loss function
    if use_asymmetric:
        loss = AsymmetricMSE(beta=cfg["asymmetric_beta"], alpha=cfg["asymmetric_alpha"])
        if verbose:
            print(f"Using Asymmetric MSE Loss (alpha={cfg['asymmetric_alpha']})")
    else:
        loss = MSE()
        if verbose:
            print("Using Standard MSE Loss")

    # Setup optimizer
    optimizer = Adam(learning_rate=cfg["learning_rate"])

    # Train model
    result = Trainer.fit(
        model=model,
        loss=loss,
        optimizer=optimizer,
        capacity=capacity,
        epochs=1000,
        val_split=0.1,
        verbose=verbose
    )

    return result


def evaluate_model(result, capacity, verbose=True):
    cfg = configs["models_config"][capacity]

    # Create evaluator
    evaluator = Evaluator(
        model=result['model'],
        scalers=result['scalers'],
        metadata=result['metadata']
    )

    # Evaluate all sets
    eval_results = evaluator.evaluate_all(
        data=result['data'],
        save_dir=cfg['metrics_dir'],
        verbose=verbose
    )

    # Save training curves
    loss_curve_path = os.path.join(cfg['metrics_dir'], f"loss_curve_{capacity}va.png")
    evaluator.save_training_curves(
        history=result['history'],
        save_path=loss_curve_path
    )

    return eval_results


def analyze_features(result, capacity, n_samples=50, verbose=True):
    if verbose:
        print(f"\n{'='*60}")
        print(f"Feature Importance Analysis - {capacity} VA")
        print(f"{'='*60}\n")

    cfg = configs["models_config"][capacity]

    # Ambil sample dari test set
    x_test = result['data']['x_test']
    n_samples = min(n_samples, len(x_test))
    x_sample = x_test[:n_samples]

    # Compute baseline (mean dari training set)
    x_train = result['data']['x_train']
    baseline = np.mean(x_train, axis=0, keepdims=True)  # Shape: (1, n_features)

    # Integrated Gradients
    ig = IntegratedGradients(
        model=result['model'],
        baseline_strategy="mean",
        steps=50
    )

    if verbose:
        print(f"Computing Integrated Gradients for {n_samples} samples...")

    feature_names = result['metadata']['feature_columns']
    importance = ig.feature_importance(
        X=x_sample,
        feature_names=tuple(feature_names),
        baseline=baseline
    )

    # Save feature importance
    importance_path = os.path.join(cfg['metrics_dir'], f"feature_importance_{capacity}va.json")
    with open(importance_path, 'w') as f:
        json.dump(importance, f, indent=2, ensure_ascii=False)

    if verbose:
        print(f"\nTop 10 Most Important Features:")
        print(f"{'Rank':<6} {'Feature':<50} {'Importance (%)':<15}")
        print("-" * 75)
        for i, feat in enumerate(importance[:72], 1):
            print(f"{i:<6} {feat['feature']:<50} {feat['relative_importance_pct']:<15.2f}")

        print(f"\nFeature importance saved to: {importance_path}")

    return importance


def main():
    parser = argparse.ArgumentParser(
        description="Training Neural Network untuk Prediksi Durasi Token Listrik"
    )
    parser.add_argument(
        "--capacity",
        nargs="+",
        choices=["900", "1300", "2200"],
        help="Capacity VA yang akan di-train (bisa multiple)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Train semua capacity"
    )
    parser.add_argument(
        "--loss",
        choices=["mse", "asymmetric"],
        default="asymmetric",
        help="Jenis loss function (default: asymmetric)"
    )
    parser.add_argument(
        "--no-analysis",
        action="store_true",
        help="Skip feature importance analysis"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print detailed progress"
    )

    args = parser.parse_args()

    # Determine capacities to train
    if args.all:
        capacities = ["900", "1300", "2200"]
    elif args.capacity:
        capacities = args.capacity
    else:
        print("Error: Harap specify --capacity atau --all")
        parser.print_help()
        sys.exit(1)

    use_asymmetric = (args.loss == "asymmetric")

    print("\n" + "="*60)
    print("TRAINING PIPELINE - PREDIKSI DURASI TOKEN LISTRIK")
    print("="*60)
    print(f"Capacities: {', '.join(capacities)} VA")
    print(f"Loss Function: {args.loss.upper()}")
    print(f"Feature Analysis: {'Enabled' if not args.no_analysis else 'Disabled'}")
    print("="*60 + "\n")

    results = {}

    # Train each capacity
    for capacity in capacities:
        try:
            # Training
            train_result = train_model(
                capacity=capacity,
                use_asymmetric=use_asymmetric,
                verbose=args.verbose
            )

            # Evaluation
            eval_result = evaluate_model(
                result=train_result,
                capacity=capacity,
                verbose=args.verbose
            )

            # Feature importance analysis
            if not args.no_analysis:
                importance = analyze_features(
                    result=train_result,
                    capacity=capacity,
                    n_samples=50,
                    verbose=args.verbose
                )
                train_result['feature_importance'] = importance

            results[capacity] = {
                'training': train_result,
                'evaluation': eval_result
            }

        except Exception as e:
            print(f"\nError training model {capacity} VA: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    # Summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)

    for capacity, res in results.items():
        eval_res = res['evaluation']
        test_metrics = eval_res['test']['metrics']

        print(f"\n{capacity} VA:")
        print(f"  Test RMSE: {test_metrics['rmse']:.4f} hari")
        print(f"  Test MAE:  {test_metrics['mae']:.4f} hari")
        print(f"  Test MAPE: {test_metrics['mape']:.4f} %")
        print(f"  Test R²:   {test_metrics['r2']:.4f}")
        print(f"  Epochs:    {res['training']['metadata']['epochs_trained']}")

    print("\n" + "="*60)
    print("Training pipeline completed successfully!")
    print("="*60 + "\n")

    return results


if __name__ == "__main__":
    main()
