from .metric import *

def get_metric(name):
    if name == 'smape':
        return smape
    if name == 'max_error_rate':
        return max_error_rate