def fit_minmax_scaler(y_data: list[float]) -> dict:
    if not y_data:
        raise ValueError("y_data tidak boleh kosong")
    
    min_val = min(y_data)
    max_val = max(y_data)

    return {
        "min": min_val,
        "max": max_val
    }

def transform_minmax(y_data: list[float], scaler: dict) -> list[float]:
    scaled_data = []

    min_value = scaler["min"]
    max_value = scaler["max"]

    if max_value == min_value:
        return [0.0] * len(y_data)

    for value in y_data:
        scaled_value = (value - min_value) / (max_value - min_value)
        scaled_data.append(scaled_value)

    return scaled_data
