from transformation.feature import fit_standard_scaler, transform_standard
from pandas.io.pytables import dropna_doc
from locale import normalize
import pandas as pd
import numpy as np
from core.models.model_900_va import Model900VA
from core.models.model_1300_va import Model1300VA
from core.models.model_2200_va import Model2200VA

def create_model(capacity, layers):
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
    no_scaling_indices = {
        i
        for i, col in enumerate(feature_columns)
        if col not in encoding_cols
    }

    scaling_indices = [
        i
        for i in range(len(feature_columns))
        if i not in no_scaling_indices
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

    return x_train_scaled, x_val_scaled, x_test_scaled
