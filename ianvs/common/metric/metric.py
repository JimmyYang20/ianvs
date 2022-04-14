import numpy as np

def smape(y_pred, y_true):
    y_pred = np.array(y_pred).reshape(-1,1)
    y_true = np.array(y_true).reshape(-1,1)
    return np.mean(np.nan_to_num(np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))))

def max_error_rate(y_pred, y_true):
    return max(np.nan_to_num(np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))))