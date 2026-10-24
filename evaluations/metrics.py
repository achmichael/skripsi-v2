import numpy as np

def mae(y_true, y_pred):
    base_condition(y_true, y_pred)

    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    if y_true.shape != y_pred.shape:
        raise ValueError("Dimensi y_true dan y_pred tidak sama")

    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    if y_true.shape != y_pred.shape:
        raise ValueError("Dimensi y_true dan y_pred tidak sama")

    nonzero = y_true != 0

    if not np.any(nonzero):
        raise ValueError("MAPE tidak bisa digunakan untuk nilai target 0")

    return float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100)

def r2(y_true, y_pred):
    base_condition(y_true, y_pred)

    errors = y_true - y_pred

    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0:
        r2 = float("nan")
    else:
        r2 = 1 - (ss_res / ss_tot)

    return r2


def base_condition(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    if y_true.shape != y_pred.shape:
        raise ValueError("Dimensi y_true dan y_pred tidak sama")
