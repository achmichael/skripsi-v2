import numpy as np

def inverse_transform_target(y_scaled: list[float], scaler: dict, use_log: bool = False) -> list[float]:
    min_val = scaler["min"]
    max_val = scaler["max"]
    
    y_original = []
    for value in y_scaled:
        raw = value * (max_val - min_val) + min_val
        
        if use_log:
            raw = min(raw, 709.0)
            final_val = float(np.expm1(raw))
            y_original.append(final_val)
        else:
            y_original.append(raw)
    
    return y_original
