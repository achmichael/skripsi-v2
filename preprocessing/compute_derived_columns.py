import pandas as pd
import numpy as np
from core.config.config import (
    DAILY_USAGE_HOURS,
    AC_WATT_BY_PK,
    KULKAS_WATT_BY_SIZE,
    WEEKLY_USAGE_FREQUENCY,
    DAILY_DEVICE_WATT,
    WEEKLY_DEVICE_WATT,
    WEEKLY_DEVICE_DURATION_HOURS
)

def compute_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def safe_float(val):
        try:
            return float(val) if pd.notna(val) else 0.0
        except (ValueError, TypeError):
            return 0.0

    devices = [
        ('Kulkas', 'always_on'),
        ('TV', 'daily'),
        ('AC', 'daily_with_pk'),
        ('Kipas', 'daily'),
        ('RiceCooker', 'daily'),
        ('MesinCuci', 'weekly')
    ]

    for device, device_type in devices:
        jumlah_col = f"{device}_Jumlah"
        kategori_col = f"{device}_Kategori"
        watt_col = f"{device}_EstimasiWattPerUnit"
        jam_col = f"{device}_EstimasiJamPerHari"
        kwh_col = f"{device}_Energi_kWhPerHari"

        if jumlah_col not in df.columns:
            continue

        jumlah = df[jumlah_col].apply(safe_float)

        if device_type == 'always_on':
            if kategori_col in df.columns:
                watt = df[kategori_col].map(KULKAS_WATT_BY_SIZE).fillna(
                    KULKAS_WATT_BY_SIZE.get("Tidak tahu", 100)
                )
            else:
                watt = 100.0

            jam = (jumlah > 0).astype(float) * 24.0
            wh_per_hari = jumlah * watt * jam

        elif device_type == 'daily_with_pk':
            pk_col = f"{device}_PK_Kategori"
            if pk_col in df.columns:
                watt = df[pk_col].map(AC_WATT_BY_PK).fillna(
                    AC_WATT_BY_PK.get("Tidak tahu", 700)
                )
            else:
                watt = 700.0

            if kategori_col in df.columns:
                jam = df[kategori_col].map(DAILY_USAGE_HOURS).fillna(0.0)
            else:
                jam = 0.0

            wh_per_hari = jumlah * watt * jam

        elif device_type == 'daily':
            watt = DAILY_DEVICE_WATT.get(device, 0)

            if kategori_col in df.columns:
                jam = df[kategori_col].map(DAILY_USAGE_HOURS).fillna(0.0)
            else:
                jam = 0.0

            wh_per_hari = jumlah * watt * jam

        elif device_type == 'weekly':
            watt = WEEKLY_DEVICE_WATT.get(device, 350)
            frekuensi_col = f"{device}_EstimasiFrekuensiPerMinggu"
            durasi_col = f"{device}_EstimasiDurasiSekaliPakaiJam"

            if kategori_col in df.columns:
                frekuensi = df[kategori_col].map(WEEKLY_USAGE_FREQUENCY).fillna(0.0)
            else:
                frekuensi = 0.0

            durasi = WEEKLY_DEVICE_DURATION_HOURS.get(device, 1.5)

            wh_per_hari = (jumlah * watt * frekuensi * durasi) / 7.0

            df[frekuensi_col] = frekuensi
            df[durasi_col] = durasi

            jam = frekuensi

        df[watt_col] = watt if isinstance(watt, pd.Series) else pd.Series([watt] * len(df), index=df.index)
        if device_type != 'weekly':
            df[jam_col] = jam if isinstance(jam, pd.Series) else pd.Series([jam] * len(df), index=df.index)
        else:
            df[jam_col] = jam if isinstance(jam, pd.Series) else pd.Series([jam] * len(df), index=df.index)

        df[kwh_col] = (wh_per_hari / 1000.0).round(6)
        
    return df
