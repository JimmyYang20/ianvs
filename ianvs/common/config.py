import os
from .utils import singleton

@singleton
class BaseConfig():
    """The base config"""
    random_seed = 42
    train_test_ratio = os.getenv("TRAIN_TEST_RATIO", 0.2)
    result_path = './result'
    model_path = './model'

@singleton
class SceneConfig():
    train_dataset_url = os.getenv("TRAIN_DATASET_URL")
    test_dataset_url = os.getenv("TEST_DATASET_URL")
