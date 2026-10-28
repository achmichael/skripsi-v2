from pyexpat import errors
from matplotlib.style import available
import numpy as np
import pandas as pd
from utils.helper import (
    calculate_rate,
    categorize_nominals
)
from core.config.config import (
    configs,
    DEVICES
)

from utils.helper import (
    calculate_always_on_device,
    calculate_daily_device,
    calculate_other_device,
    calculate_weekly_device
)

def extractor(df: pd.DataFrame, capacity: str):
    if df.empty:
        raise ValueError("Data kosong.")

    feature_columns = configs["models_config"][capacity]["features"]

    target_col = configs["models_config"][capacity]["target"]
    use_log = configs["models_config"][capacity]["use_log_transform"]

    if "Daya_Listrik_Rumah_VA" in df.columns:
        df["Tarif_PLN_Eksak_Rp"] = df.apply(calculate_rate, 1)
    else:
        df["Tarif_PLN_Eksak_Rp"] = 1444.70

    if "Nominal_Token_Terakhir_Rp" in df.columns and "Total_Energi_Semua_kWhPerHari" in df.columns:
        df["Estimasi_kWh_Didapat"] = ((df["Nominal_Token_Terakhir_Rp"] - 2500) * 1.05) / df["Tarif_PLN_Eksak_Rp"]
        df["Estimasi_Fisika_Durasi_Hari"] = df["Estimasi_kWh_Didapat"] / (df["Total_Energi_Semua_kWhPerHari"] + 0.1)
    else:
        df["Estimasi_Fisika_Durasi_Hari"] = 0.0

    if "Frekuensi_Isi_Token_Per_Bulan" in df.columns:
        freq = df["Frekuensi_Isi_Token_Per_Bulan"].replace(0, np.nan)
        df["Durasi_Dari_Frekuensi"] = (30.0 / freq).clip(1, 60)
        df["Durasi_Dari_Frekuensi"] = df["Durasi_Dari_Frekuensi"].fillna(30.0)

    if ("Nominal_Token_Terakhir_Rp" in df.columns and
        "Estimasi_Tarif_Per_kWh_Rp" in df.columns and
        "Total_Energi_Semua_kWhPerHari" in df.columns):
        kwh_beli = df["Nominal_Token_Terakhir_Rp"] / (
            df["Estimasi_Tarif_Per_kWh_Rp"].replace(0, 1444.70)
        )
        df["Rasio_Token_vs_Energi"] = kwh_beli / (
            df["Total_Energi_Semua_kWhPerHari"] + 0.1
        )

    if "Nominal_Token_Terakhir_Rp" in df.columns:
        df["Token_Nominal_Kategori"] = df["Nominal_Token_Terakhir_Rp"].apply(
            categorize_nominals
        )

    if ("Total_Energi_Semua_kWhPerHari" in df.columns and
        "Nominal_Token_Terakhir_Rp" in df.columns):
        df["Energi_Per_Nominal"] = (
            df["Total_Energi_Semua_kWhPerHari"] /
            (df["Nominal_Token_Terakhir_Rp"] / 1000.0 + 0.01)
        )

    if ("Estimasi_Fisika_Durasi_Hari" in df.columns and
        "Durasi_Dari_Frekuensi" in df.columns):
        df["Fisika_vs_Frekuensi_Gap"] = (
            df["Estimasi_Fisika_Durasi_Hari"] - df["Durasi_Dari_Frekuensi"]
        )
        df["Rasio_Fisika_vs_Frekuensi"] = df["Estimasi_Fisika_Durasi_Hari"] / (df["Durasi_Dari_Frekuensi"] + 1e-8)
        df["Rasio_Fisika_vs_Frekuensi"] = df["Rasio_Fisika_vs_Frekuensi"].clip(0.1, 10.0)

    if "Daya_Listrik_Rumah_VA" in df.columns and "Total_Energi_Semua_kWhPerHari" in df.columns:
        df["Daya_x_TotalEnergi"] = (
            df["Daya_Listrik_Rumah_VA"] * df["Total_Energi_Semua_kWhPerHari"]
        )

    # NOW check for available columns
    available_columns = list(df.columns)

    missing = []
    resolved_columns = []

    for col in feature_columns:
        if col in available_columns:
            resolved_columns.append(col)
            continue

        expanded = [c for c in available_columns if c.startswith(f"{col}_")]

        if expanded:
            resolved_columns.extend(expanded)
        else:
            missing.append(col)

    if missing:
        raise ValueError(f"Kolom fitur tidak ditemukan di dataset: {missing[:5]}")

    feature_columns = resolved_columns

    if target_col not in df.columns:
        raise ValueError(f"Kolom target tidak ditemukan di dataset: {target_col}")

    x_df = df[feature_columns].apply(pd.to_numeric, errors="coerce")
    if x_df.isna().any().any():
        bad_cols = x_df.columns[x_df.isna().any()].tolist()
        raise ValueError(f"Non numeric values found in columns: {bad_cols}")
    x_data = x_df.values.tolist()

    y_raw = df[target_col].astype(float).values

    if use_log:
        y_data = np.log1p(np.maximum(y_raw, 0)).tolist()
    else:
        y_data = y_raw.tolist()

    return x_data, y_data, feature_columns, target_col
