import numpy as np

def fit_standard_scaler(x_data: list[list[float]]) -> dict:
    num_features = len(x_data[0]) if x_data else 0
    means = []
    stds = []

    for feature_index in range(num_features):
        column_values = [row[feature_index] for row in x_data]
        mean = float(np.mean(column_values))
        std = float(np.std(column_values))

        if std == 0.0:
            std = 1.0

        means.append(mean)
        stds.append(std)

    return {
        "mean": means,
        "std": stds
    }


def transform_standard(x_data: list[list[float]], scaler: dict) -> list[list[float]]:
    scaled_data = []
    for row in x_data:
        scaled_row = []

        for i, value in enumerate(row):
            mean_value = scaler["mean"][i]
            std_value = scaler["std"][i]
            scaled_value = (value - mean_value) / std_value
            scaled_row.append(scaled_value)

        scaled_data.append(scaled_row)

    return scaled_data
