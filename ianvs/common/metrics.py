import sys
import numpy as np


def smape(y_true, y_pred):
    y_true = np.array(y_true).reshape(-1, 1)
    y_pred = np.array(y_pred).reshape(-1, 1)
    return np.mean(np.nan_to_num(np.abs(y_true - y_pred) / (np.abs(y_pred) + np.abs(y_true))))


def max_error_rate(y_true, y_pred):
    return max(np.nan_to_num(np.abs(y_true - y_pred) / (np.abs(y_pred) + np.abs(y_true))))


def get_metric_func(metric_name):
    """ get metric func """
    return getattr(sys.modules[__name__], metric_name)
