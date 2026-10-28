import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import json

def save_loss_curve(history: dict, save_path: str, model_type: str):
    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])

    if not train_loss:
        raise ValueError("Train loss kosong, Tidak bisa membuat kurva loss")

    trained_epochs = len(train_loss)
    epoch_axis = list(range(1, trained_epochs + 1))

    marker_size = max(1.5, 5.0 - trained_epochs * 0.008)

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(
        epoch_axis, train_loss,
        color="#1565C0", linewidth=1.5,
        marker="o", markersize=marker_size, markerfacecolor="#1565C0",
        label="Train Loss",
        zorder=3,
    )

    if val_loss:
        ax.plot(
            epoch_axis, val_loss,
            color="#E53935", linewidth=1.5,
            marker="o", markersize=marker_size, markerfacecolor="#E53935",
            label="Validation Loss",
            zorder=3,
        )

    if val_loss:
        best_epoch = val_loss.index(min(val_loss)) + 1
        best_val   = min(val_loss)
        ax.scatter(
            [best_epoch], [best_val],
            color="#FFB300", s=60, zorder=5,
            label=f"Best Val Loss (epoch {best_epoch}: {best_val:.6f})",
        )
    else:
        best_epoch = train_loss.index(min(train_loss)) + 1
        best_val   = min(train_loss)
        ax.scatter(
            [best_epoch], [best_val],
            color="#FFB300", s=60, zorder=5,
            label=f"Best Train Loss (epoch {best_epoch}: {best_val:.6f})",
        )

    ax.axvline(
        x=trained_epochs, color="gray", linestyle=":", linewidth=1.2,
        label=f"Early stop (epoch {trained_epochs})",
    )

    ax.set_title(
        f"Training vs Validation Loss — {model_type.upper()} ({trained_epochs} epoch)",
        fontsize=14, fontweight="bold",
    )
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("MSE Loss", fontsize=12)
    ax.set_xlim(1, trained_epochs)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"Loss curve disimpan ke: {save_path} (epoch aktual: {trained_epochs})")

def save_prediction_scatter(y_actual, y_predicted, save_path: str, model_type: str, label: str):
    """Plot scatter prediksi vs aktual dengan zona toleransi dan statistik."""
    actual_vals = list(y_actual)
    pred_vals   = list(y_predicted)

    mean_actual = sum(actual_vals) / len(actual_vals)
    ss_tot = sum((a - mean_actual) ** 2 for a in actual_vals)
    ss_res = sum((a - p) ** 2 for a, p in zip(actual_vals, pred_vals))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    within_20pct = sum(
        1 for a, p in zip(actual_vals, pred_vals)
        if a != 0 and abs(p - a) / abs(a) <= 0.20
    )
    pct_within = within_20pct / len(actual_vals) * 100

    all_vals = actual_vals + pred_vals
    min_val  = min(all_vals)
    max_val  = max(all_vals)
    margin   = (max_val - min_val) * 0.05
    lo = min_val - margin
    hi = max_val + margin

    fig, ax = plt.subplots(figsize=(9, 8))

    ref = [lo, hi]
    ax.fill_between(
        ref,
        [v * 0.80 for v in ref],
        [v * 1.20 for v in ref],
        alpha=0.12,
        color="#2196F3",
        label="Toleransi ±20%",
    )

    colors = [
        "#4CAF50" if abs(a) > 0 and abs(p - a) / abs(a) <= 0.20 else "#F44336"
        for a, p in zip(actual_vals, pred_vals)
    ]
    ax.scatter(actual_vals, pred_vals, c=colors, alpha=0.75, s=35, zorder=3)

    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.5, label="Ideal (y = x)", zorder=4)

    ax.scatter([], [], c="#4CAF50", s=35, label=f"Dalam toleransi ({within_20pct}/{len(actual_vals)} titik)")
    ax.scatter([], [], c="#F44336", s=35, label="Di luar toleransi")

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(f"Aktual ({label})", fontsize=12)
    ax.set_ylabel(f"Prediksi ({label})", fontsize=12)
    ax.set_title(f"Prediksi vs Aktual — {model_type.upper()}", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)

    stats_text = f"R² = {r2:.4f}\nDalam ±20%: {pct_within:.1f}%"
    ax.text(
        0.98, 0.05, stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8, edgecolor="gray"),
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Scatter plot disimpan ke: {save_path}")

def plot_error_distribution(
    y_true: list[float],
    y_pred: list[float],
    mae_value: float,
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """
    Membuat visualisasi distribusi error prediksi Neural Network.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    y_t = np.array(y_true)
    y_p = np.array(y_pred)
    errors = y_p - y_t

    n = len(errors)
    if n == 0:
        return

    errors_scaled = errors
    mae_scaled = mae_value

    mean_e = np.mean(errors_scaled)
    median_e = np.median(errors_scaled)
    std_e = np.std(errors_scaled)
    min_e = np.min(errors_scaled)
    max_e = np.max(errors_scaled)
    range_e = max_e - min_e

    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.hist(errors_scaled, bins=30, color='skyblue', edgecolor='black', alpha=0.7)

    ticks = np.linspace(min_e, max_e, 10)
    ax1.set_xticks(ticks)
    ax1.set_xticklabels([f"{v:+.1f}" for v in ticks])

    ax1.set_xlabel("Error (hari) [Prediksi - Aktual]", fontsize=12)
    ax1.set_ylabel("Frekuensi", fontsize=12)
    ax1.set_title("Distribusi Error Prediksi", fontsize=14, fontweight='bold')

    ax1.axvspan(-mae_scaled, mae_scaled, color='lightgreen', alpha=0.3, label='±MAE zone')

    offset = max(range_e * 0.005, 1e-5)
    mean_offset = 0.0
    median_offset = 0.0

    if abs(mean_e - 0.0) <= offset * 2:
        mean_offset = offset

    if abs(median_e - 0.0) <= offset * 2 or abs(median_e - mean_e) <= offset * 2:
        median_offset = -offset

    ax1.axvline(0, color='black', linestyle='-', linewidth=2, label='Zero Error')
    ax1.axvline(mean_e + mean_offset, color='orange', linestyle='--', linewidth=2, label='Mean')
    ax1.axvline(median_e + median_offset, color='yellow', linestyle=':', linewidth=2, label='Median')

    ax1.legend(loc='upper left', fontsize=10)

    stats_text = (
        f"n = {n} data\n"
        f"Mean = {mean_e:+.1f} hari\n"
        f"Median = {median_e:+.1f} hari\n"
        f"Std Dev = {std_e:.1f} hari\n"
        f"Range: [{min_e:+.1f}, {max_e:+.1f}] hari"
    )
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray')
    ax1.text(0.95, 0.95, stats_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right', bbox=props)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Error distribution disimpan ke: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def save_feature_importance_bar(
    importance: list[dict],
    save_path: str,
    model_type: str,
    top_n: int = 10,
):
    """Plot top-N feature importance sebagai horizontal bar chart."""
    top = importance[:top_n]
    features = [item["feature"] for item in top][::-1]
    values = [item["relative_importance_pct"] for item in top][::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(features, values, color="#4C72B0", alpha=0.85)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            f" {value:.2f}%",
            va="center",
            fontsize=9,
        )

    ax.set_xlabel("Relative Importance (%)", fontsize=12)
    ax.set_ylabel("Fitur", fontsize=12)
    ax.set_title(
        f"Top {top_n} Feature Importance — {model_type.upper()}",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, axis="x", alpha=0.3)
    ax.margins(x=0.15)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Feature importance chart disimpan ke: {save_path}")


def save_metrics_json(metrics: dict, save_path: str):
    """Simpan metrik evaluasi ke JSON."""
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
    print(f"Metrics JSON disimpan ke: {save_path}")
