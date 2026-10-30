from matplotlib.pylab import axis
from pyexpat import errors
from matplotlib.style import available
import numpy as np
import pandas as pd

from utils.helper import (
    calculate_rate,
    categorize_nominals,
    pbjt_rate
)

from core.config.config import (
    configs,
    DEVICES, ENERGY_COLS, PBJT_CITY, PBJT_DEFAULT
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

    total_energi = sum(df[col] for col in ENERGY_COLS if col in df.columns)

    df["Total_Energi_Semua_kWhPerHari"] = total_energi

    if "Daya_Listrik_Rumah_VA" in df.columns:
        df["Tarif_PLN_Eksak_Rp"] = df.apply(calculate_rate, 1)
    else:
        df["Tarif_PLN_Eksak_Rp"] = 1444.70

    if "Kota/Kabupaten" in df.columns:
        pbjt = df.apply(
            lambda row: pbjt_rate(row["Kota/Kabupaten"], row.get("Daya_Listrik_Rumah_VA")),
            axis=1,
        )
    else:
        pbjt = PBJT_DEFAULT

    if "Nominal_Token_Terakhir_Rp" in df.columns and "Total_Energi_Semua_kWhPerHari" in df.columns:
        # 1 = Tarif dasar = 100%
        df["Estimasi_kWh_Didapat"] = ((df["Nominal_Token_Terakhir_Rp"] - 2500) / (1 + pbjt)) / df["Tarif_PLN_Eksak_Rp"]
        df["Estimasi_Fisika_Durasi_Hari"] = df["Estimasi_kWh_Didapat"] / (df["Total_Energi_Semua_kWhPerHari"] + 0.1)
    else:
        df["Estimasi_Fisika_Durasi_Hari"] = 0.0
    if "Frekuensi_Isi_Token_Per_Bulan" in df.columns:
        freq = df["Frekuensi_Isi_Token_Per_Bulan"].replace(0, np.nan)
        df["Durasi_Dari_Frekuensi"] = (30.0 / freq).clip(1, 60)
        df["Durasi_Dari_Frekuensi"] = df["Durasi_Dari_Frekuensi"].fillna(30.0)

    if ("Estimasi_Fisika_Durasi_Hari" in df.columns and
        "Durasi_Dari_Frekuensi" in df.columns):
        df["Rasio_Fisika_vs_Frekuensi"] = df["Estimasi_Fisika_Durasi_Hari"] / (df["Durasi_Dari_Frekuensi"] + 1e-8)
        df["Rasio_Fisika_vs_Frekuensi"] = df["Rasio_Fisika_vs_Frekuensi"].clip(0.1, 10.0)

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
