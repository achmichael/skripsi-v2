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
    Kolom wajib pada df: ['capacity', 'combination', 'value']
    (Setiap kombinasi harus memiliki >1 baris observasi, misal per-fold CV)
    """
    sns.set_style("white")
    df = df.copy()
    df["capacity"] = df["capacity"].astype(str)
    df["x_label"] = df["capacity"] + " VA - K" + df["combination"].astype(str)

    # Peringatan jika setiap kombinasi hanya punya 1 data (kotak akan gepeng)
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
        ascending = metric.upper() != "R2"  # Error: terkecil di kiri; R2: tertinggi di kiri
        medians = medians.sort_values("value", ascending=ascending)
    label_order = medians["x_label"].tolist()

    # Warna kontras per kategori daya (mirip referensi)
    default_palette = {
        "900": "#2b70a1",   # Biru
        "1300": "#e87d24",  # Oranye
        "2200": "#2e9e6b",  # Hijau
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
        dodge=False,          # WAJIB False agar kotak tepat di tengah & lebar penuh
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

    # Rapikan bingkai (spines) seperti gambar referensi
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#444444")
        spine.set_linewidth(0.8)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10)
    ax.tick_params(axis="y", labelsize=10)
    ax.set_xlabel("Kombinasi", fontsize=11, labelpad=8)
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
        description="Generate boxplot untuk satu metrik dari data per-fold.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Contoh penggunaan (input nilai per-fold untuk setiap kombinasi):
  python generate_boxplot.py --metric MAE \\
    --entry 900 1 2.85 3.12 2.91 3.04 3.03 \\
    --entry 900 2 3.10 3.35 3.18 3.29 3.28 \\
    --entry 1300 1 1.95 2.18 2.02 2.15 2.10 \\
    --output boxplot_mae.png

Atau dari file CSV (kolom: Kategori, Kombinasi, MAE, RMSE, MAPE, R2):
  python generate_boxplot.py --csv hasil_per_fold.csv --metric MAE --output boxplot_mae.png
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
        help="Format: <capacity> <kombinasi> <val_fold_1> <val_fold_2> ... (bisa diulang)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        help="Path file CSV berisi skor mentah per-fold",
    )
    parser.add_argument(
        "--output",
        default="boxplot.png",
        help="Path output file PNG (default: boxplot.png)",
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

    if args.csv:
        raw_df = pd.read_csv(args.csv)
        raw_df["Kategori"] = raw_df["Kategori"].ffill()
        # Cocokkan nama kolom secara case-insensitive
        col_map = {c.upper(): c for c in raw_df.columns}
        if metric not in col_map:
            print(f"Error: Kolom '{metric}' tidak ditemukan di CSV.", file=sys.stderr)
            sys.exit(1)

        df = pd.DataFrame({
            "capacity": raw_df[col_map["KATEGORI"]].astype(str).str.replace(" VA", "", regex=False).str.strip(),
            "combination": raw_df[col_map["KOMBINASI"]],
            "value": pd.to_numeric(raw_df[col_map[metric]]),
        })
    elif args.entry:
        rows = []
        for item in args.entry:
            if len(item) < 4:
                print(
                    f"Error pada --entry {' '.join(item)}: butuh minimal <capacity> <kombinasi> dan 2 nilai fold.",
                    file=sys.stderr,
                )
                sys.exit(1)
            capacity, comb = item[0], item[1]
            for val in item[2:]:
                rows.append({"capacity": capacity, "combination": comb, "value": float(val)})
        df = pd.DataFrame(rows)
    else:
        print("Error: Berikan input melalui --entry atau --csv.", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output)
    generate_boxplot(df, metric, output_path, sort_by_median=not args.no_sort)
    print(f"Boxplot berhasil disimpan: {output_path.absolute()}")


if __name__ == "__main__":
    main()
