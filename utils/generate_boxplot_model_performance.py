#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def generate_boxplot(
    df: pd.DataFrame,
    metric: str,
    output_path: Path,
    figsize=(14, 6),
    dpi=300,
    sort_by_median: bool = True,
):
    """
    Generate boxplot untuk satu metrik across semua golongan daya.
    Kolom wajib pada df: ['capacity', 'feature_type', 'value']
    """
    sns.set_style("white")
    df = df.copy()
    df["capacity"] = df["capacity"].astype(str)
    df["x_label"] = df["capacity"] + " VA - " + df["feature_type"]

    # Peringatan jika setiap kombinasi hanya punya 1 data
    counts = df.groupby("x_label")["value"].count()
    if (counts < 2).any():
        print(
            "[WARNING] Terdapat kombinasi dengan hanya 1 data. "
            "Masukkan data per-fold (misal 5 nilai per kombinasi) agar kotak boxplot terbentuk.",
            file=sys.stderr,
        )

    # Urutkan sumbu-X berdasarkan median
    medians = df.groupby(["x_label", "capacity"], as_index=False)["value"].median()
    if sort_by_median:
        ascending = metric.upper() != "R2"
        medians = medians.sort_values("value", ascending=ascending)
    label_order = medians["x_label"].tolist()

    # Warna kontras per kategori daya
    default_palette = {
        "900": "#2b70a1",
        "1300": "#e87d24",
        "2200": "#2e9e6b",
    }
    capacities = sorted(df["capacity"].unique(), key=lambda x: int(x) if x.isdigit() else x)
    fallback_colors = sns.color_palette("tab10", n_colors=len(capacities))
    color_map = {
        cap: default_palette.get(cap, fallback_colors[i])
        for i, cap in enumerate(capacities)
    }

    fig, ax = plt.subplots(figsize=figsize)

    sns.boxplot(
        data=df,
        x="x_label",
        y="value",
        hue="capacity",
        order=label_order,
        palette=color_map,
        dodge=False,
        width=0.58,
        linewidth=1.0,
        boxprops={"edgecolor": "#444444"},
        medianprops={"color": "#444444", "linewidth": 1.2},
        whiskerprops={"color": "#666666", "linewidth": 0.9},
        capprops={"color": "#666666", "linewidth": 0.9},
        flierprops={
            "marker": "o",
            "markerfacecolor": "none",
            "markeredgecolor": "#555555",
            "markersize": 4.5,
        },
        ax=ax,
    )

    # Rapikan bingkai (spines)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#444444")
        spine.set_linewidth(0.8)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10)
    ax.tick_params(axis="y", labelsize=10)
    ax.set_xlabel("Golongan Daya - Metode Fitur", fontsize=11, labelpad=8)
    ax.set_ylabel(metric.upper(), fontsize=11, labelpad=8)

    # Legend rapi tanpa duplikat
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ordered_handles = [by_label[c] for c in capacities if c in by_label]
    ordered_labels = [f"{c} VA" for c in capacities if c in by_label]

    legend = ax.legend(
        ordered_handles,
        ordered_labels,
        title="Kategori Daya",
        loc="upper right",
        frameon=True,
        fontsize=9.5,
        title_fontsize=10,
    )
    legend.get_frame().set_edgecolor("#cccccc")
    legend.get_frame().set_linewidth(0.8)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate boxplot untuk membandingkan All Features vs Feature Selection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Contoh penggunaan:
  python generate_boxplot_model_performance.py --metric MAE \\
    --entry 900 "All Features" 3.79 3.90 3.67 3.48 2.45 \\
    --entry 900 "Feature Selection" 3.53 2.85 3.32 3.31 2.51 \\
    --entry 1300 "All Features" 3.98 2.24 3.34 3.89 3.08 \\
    --entry 1300 "Feature Selection" 2.87 1.67 3.19 3.16 3.04 \\
    --output boxplot_mae_comparison.png
        """,
    )
    parser.add_argument(
        "--metric",
        required=True,
        choices=["MAE", "RMSE", "MAPE", "R2", "mae", "rmse", "mape", "r2"],
        help="Metrik yang akan diplot",
    )
    parser.add_argument(
        "--entry",
        action="append",
        nargs="+",
        help='Format: <capacity> "<feature_type>" <val_fold_1> <val_fold_2> ... (bisa diulang)',
    )
    parser.add_argument(
        "--output",
        default="boxplot_comparison.png",
        help="Path output file PNG (default: boxplot_comparison.png)",
    )
    parser.add_argument(
        "--no-sort",
        action="store_true",
        help="Jangan urutkan sumbu-X berdasarkan median",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    metric = args.metric.upper()

    if not args.entry:
        print("Error: Berikan input melalui --entry.", file=sys.stderr)
        sys.exit(1)

    rows = []
    for item in args.entry:
        if len(item) < 4:
            print(
                f"Error pada --entry {' '.join(item)}: butuh minimal <capacity> <feature_type> dan 2 nilai fold.",
                file=sys.stderr,
            )
            sys.exit(1)
        capacity, feature_type = item[0], item[1]
        for val in item[2:]:
            rows.append({"capacity": capacity, "feature_type": feature_type, "value": float(val)})
    df = pd.DataFrame(rows)

    output_path = Path(args.output)
    generate_boxplot(df, metric, output_path, sort_by_median=not args.no_sort)
    print(f"Boxplot berhasil disimpan: {output_path.absolute()}")


if __name__ == "__main__":
    main()
