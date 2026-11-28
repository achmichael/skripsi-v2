def fit_minmax_scaler(x_data: list[list[float]]) -> dict:
    num_features = len(x_data[0]) if x_data else 0
    mins = []
    maxs = []

    for feature_index in range(num_features):
        column_values = [row[feature_index] for row in x_data]
        min_val = float(min(column_values))
        max_val = float(max(column_values))

        mins.append(min_val)
        maxs.append(max_val)

    return {
        "min": mins,
        "max": maxs
    }


def transform_minmax(x_data: list[list[float]], scaler: dict) -> list[list[float]]:
    scaled_data = []
    for row in x_data:
        scaled_row = []

        for i, value in enumerate(row):
            min_val = scaler["min"][i]
            max_val = scaler["max"][i]
            
            if max_val == min_val:
                scaled_value = 0.0
            else:
                scaled_value = (value - min_val) / (max_val - min_val)
            
            scaled_row.append(scaled_value)

        scaled_data.append(scaled_row)

    return scaled_data
