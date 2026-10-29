from typing import Any, Dict, List, Optional
from transformation.feature import fit_standard_scaler, transform_standard
from pandas.io.pytables import dropna_doc
from locale import normalize
import pandas as pd
import numpy as np
from core.models.model_900_va import Model900VA
from core.models.model_1300_va import Model1300VA
from core.models.model_2200_va import Model2200VA
from core.config.config import (
    DAILY_USAGE_HOURS,
    DAILY_DEVICE_WATT,
    AC_WATT_BY_PK,
    KULKAS_WATT_BY_SIZE,
    WEEKLY_USAGE_FREQUENCY,
    OTHER_DEFAULT_WATT,
    OTHER_USAGE_TO_DAILY_HOURS,
    WEEKLY_DEVICE_WATT,
    WEEKLY_DEVICE_DURATION_HOURS, PBJT_DEFAULT,
    PBJT_CITY
)

def create_model(capacity, layers):
    np.random.seed(42)

    match str(capacity):
        case "900":
            model = Model900VA(layer_sizes=layers)
        case "1300":
            model = Model1300VA(layer_sizes=layers)
        case "2200":
            model = Model2200VA(layer_sizes=layers)
        case _:
            raise ValueError(
                f"Kapasitas listrik tidak didukung: {capacity}"
            )

    return model

def mode_imputation(df, cols):
    for col in cols:
        if col in df.columns:
            mode_value = df[col].mode()

            if not mode_value.empty:
                df[col] = df[col].fillna(mode_value[0])

    return df

def one_hot_encode(df, config):
    for col in config["preprocessing"]["one_hot_cols"]:
        if col not in df.columns:
            continue

        categories = config["preprocessing"]["one_hot_categories"].get(col, sorted(df[col].dropna().unique()))

        for cat in categories:
            col_name = f"{col}_{cat}"
            df[col_name] = (df[col] == cat).astype(int)

        df = df.drop(columns=[col])

    return df

def ordinal_encode(df, config):
    for col, mapping in config["preprocessing"]["ordinal_encoding"].items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    return df

def calculate_rate(row):
    power = row.get("Daya_Listrik_Rumah_VA", 900)
    is_subsidi = str(row.get("Status_Subsidi_Listrik", "")).strip().lower() == "subsidi"

    match power:
        case p if p <= 450:
            return 415.0
        case 900:
            return 605.0 if is_subsidi else 1352.0
        case 1300 | 2200:
            return 1444.70
        case p if p > 2200:
            return 1699.53
        case _:
            return 1444.70

def pbjt_rate(city, power):
    if pd.isna(city):
        return PBJT_DEFAULT

    name = str(city).strip().lower()
    for key, value in PBJT_CITY.items():
        if key.lower() != name:
            continue

        if isinstance(value, dict):
            try:
                va = float(power)
            except (TypeError, ValueError):
                return PBJT_DEFAULT

            if va <= 1000:
                return value["250 - 1.000 VA"]
            if va <= 3500:
                return value["1.001 - 3.500 VA"]
            return value["> 3.500 VA"]

        return value

    return PBJT_DEFAULT

def frequency_mapping(df, configs):
    mappings = {}
    for col in configs["preprocessing"]["frequency_cols"]:
        if col in df.columns:
            frequencies = df[col].value_counts(
                True,
                True
            )

            mappings[col] = frequencies.to_dict()

    return mappings

def frequency_encode(df, mappings, unknown_value=0.0):
    df = df.copy()

    for col, mapping in mappings.items():
        encoded = df[col].map(mapping)

        df[col] = encoded.fillna(unknown_value).astype(np.float64)

    return df

def categorize_nominals(x):
    if x <= 20_000:   return 0
    if x <= 50_000:   return 1
    if x <= 100_000:  return 2
    if x <= 200_000:  return 3
    return 4


def selective_feature_scaling(
    x_train,
    x_val,
    x_test,
    feature_columns,
    encoding_cols,
):
    scaling_indices = [
        i
        for i, col in enumerate(feature_columns)
        if col not in encoding_cols
    ]

    # copy the data so that the original data remains unchanged
    x_train_scaled = [row.copy() for row in x_train]
    x_val_scaled = [row.copy() for row in x_val]
    x_test_scaled = [row.copy() for row in x_test]

    if scaling_indices:
        train_numeric = [
            [row[i] for i in scaling_indices]
            for row in x_train
        ]

        scaler = fit_standard_scaler(train_numeric)

        train_transformed = transform_standard(train_numeric, scaler)

        val_numeric = [
            [row[i] for i in scaling_indices]
            for row in x_val
        ]

        test_numeric = [
            [row[i] for i in scaling_indices]
            for row in x_test
        ]

        val_transformed = transform_standard(val_numeric, scaler)
        test_transformed = transform_standard(test_numeric, scaler)

        # place the scaled results back into their original feature positions
        for dataset, transformed in [
            (x_train_scaled, train_transformed),
            (x_val_scaled, val_transformed),
            (x_test_scaled, test_transformed),
        ]:
            for row_idx, row in enumerate(transformed):
                for col_idx, feature_idx in enumerate(scaling_indices):
                    dataset[row_idx][feature_idx] = row[col_idx]
    else:
        scaler = {"mean": [], "std": []}

    scaler_data = {
        "scaler": scaler,
        "scaling_indices": scaling_indices,
    }

    return x_train_scaled, x_val_scaled, x_test_scaled, scaler_data


def transform_selective(x_data, scaler_data):
    scaling_indices = scaler_data["scaling_indices"]
    scaled = [row.copy() for row in x_data]

    if not scaling_indices:
        return scaled

    numeric = [[row[i] for i in scaling_indices] for row in x_data]
    transformed = transform_standard(numeric, scaler_data["scaler"])

    for row_idx, row in enumerate(transformed):
        for col_idx, feature_idx in enumerate(scaling_indices):
            scaled[row_idx][feature_idx] = row[col_idx]

    return scaled

def calculate_energy_wh(
    quantity: float,
    watt: float,
    hours_per_day: float
) -> float:
    return round(
        max(0.0, quantity) * max(0.0, watt) * max(0.0, hours_per_day), 3
    )

def wh_to_kwh(energy_wh: float) -> float:
    return round(energy_wh / 1000, 6)

def calculate_always_on_device(
    quantity: float,
    watt_per_unit: float,
) -> Dict[str, float]:
    # untuk kulkas diasumsikan hidup selama 24 jam jika jumlahnya lebih dari 0
    hour = 24.0 if quantity > 0 else 0.0
    energy_wh = calculate_energy_wh(quantity, watt_per_unit, hour)

    return {
        "quantity": quantity,
        "watt_per_unit": watt_per_unit,
        "hours_per_day": hour,
        "energy_wh_per_day": energy_wh,
        "energy_kwh_per_day": wh_to_kwh(energy_wh)
    }

def calculate_daily_device(
    quantity: float,
    watt_per_unit: float,
    usage_category: str
) -> Dict[str, float]:
    hour = DAILY_USAGE_HOURS.get(usage_category, 0.0)
    energy_wh = calculate_energy_wh(quantity, watt_per_unit, hour)

    return {
        "quantity": quantity,
        "watt_per_unit": watt_per_unit,
        "hours_per_day": hour,
        "energy_wh_per_day": energy_wh,
        "energy_kwh_per_day": wh_to_kwh(energy_wh)
    }

def calculate_weekly_device(
    quantity: float,
    watt_per_unit: float,
    frequency_per_week: float,
    duration_per_use: float
) -> Dict[str, float]:
    energy_wh = (
        max(0.0, quantity) * max(0.0, watt_per_unit) * max(0.0, frequency_per_week) * max(0.0, duration_per_use) / 7.0
    )

    energy_wh = round(energy_wh, 3)

    return {
        "quantity": quantity,
        "watt_per_unit": watt_per_unit,
        "frequency_per_week": frequency_per_week,
        "duration_per_use": duration_per_use,
        "energy_wh_per_day": energy_wh,
        "energy_kwh_per_day": wh_to_kwh(energy_wh)
    }

def calculate_other_device(
    type: str,
    quantity: float,
    usage_category: str,
) -> Dict[str, Any]:
    watt = OTHER_DEFAULT_WATT.get(type, OTHER_DEFAULT_WATT["Lainnya"])
    hour = OTHER_USAGE_TO_DAILY_HOURS.get(usage_category, 0.0)

    energy_wh = calculate_energy_wh(quantity, watt, hours_per_day=hour)

    return {
        "type": type,
        "quantity": quantity,
        "watt_per_unit": watt,
        "hour_per_day": hour,
        "energy_wh_per_day": energy_wh,
        "energy_kwh_per_day": wh_to_kwh(energy_wh)
    }
