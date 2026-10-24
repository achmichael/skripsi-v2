import math
import random

import numpy as np
import pandas as pd

from core.config.config import configs
from utils.helper import (
    mode_imputation,
    ordinal_encode,
    frequency_mapping,
    frequency_encode,
    calculate_rate
)

def handle_missing_value(df: pd.DataFrame) -> pd.DataFrame:
    nullable_cols = configs["preprocessing"]["nullable_cols"]

    for col in nullable_cols:
        if col in df.columns:
            df[col] = df[col].replace({
                "Tidak tahu": np.nan,
                "Tidak diisi": np.nan
            })

    if "Daya_Listrik_Rumah_VA" in df.columns:
        df["Daya_Listrik_Rumah_VA"] = df["Daya_Listrik_Rumah_VA"].replace({
            "Tidak tahu": 900,
            "> 5500": 7700,
        })
        df["Daya_Listrik_Rumah_VA"] = pd.to_numeric(
            df["Daya_Listrik_Rumah_VA"], errors="coerce"
        )

    if "Status_Subsidi_Listrik" in df.columns:
        df["Status_Subsidi_Listrik"] = df["Status_Subsidi_Listrik"].map({
            "Subsidi": 0,
            "Non Subsidi": 1,
        }).astype(float)

    imputed = mode_imputation(df, nullable_cols)

    return imputed

def preprocess(path: str):
    df = pd.read_csv(path, encoding="utf-8-sig")

    handled = handle_missing_value(df)

    mapped = frequency_encode(handled, frequency_mapping(df, configs))

    encoded = ordinal_encode(mapped, configs)

    return encoded

def train_test_split(x_data: list[list[float]], y_data: list[list[float]], test_ratio: float = 0.2, seed: int = 42):
    combined = list(zip(x_data, y_data))

    random.seed(seed)
    random.shuffle(combined)

    test_size = int(len(combined) * test_ratio)

    test_data = combined[:test_size]
    train_data = combined[test_size:]

    x_train = [item[0] for item in train_data]
    y_train = [item[1] for item in train_data]

    x_test = [item[0] for item in test_data]
    y_test = [item[1] for item in test_data]

    return x_train, y_train, x_test, y_test

