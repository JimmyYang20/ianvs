import os
from inspect import getfullargspec

import joblib
import pandas as pd
from xgboost import XGBClassifier as Model

from sedna.datasources import CSVDataParse

DATACONF = {
    "ATTRIBUTES": ["Season", "Cooling startegy_building level"],
    "LABEL": "Thermal preference",
}


def parse_kwargs(func, **kwargs):
    """ get valid parameters in kwargs """
    if not callable(func):
        return kwargs
    need_kw = getfullargspec(func)
    if need_kw.varkw == 'kwargs':
        return kwargs
    return {k: v for k, v in kwargs.items() if k in need_kw.args}


def dataset_feature_process(df: pd.DataFrame):
    if "City" in df.columns:
        df.drop(["City"], axis=1, inplace=True)
    for feature in df.columns:
        if feature in ["Season", ]:
            continue
        df[feature] = df[feature].apply(lambda x: float(x) if x else 0.0)
    df['Thermal preference'] = df['Thermal preference'].apply(
        lambda x: int(float(x)) if x else 1)
    return df


os.environ['BACKEND_TYPE'] = 'SKLEARN'

class Estimator:
    def __init__(self, **kwargs):
        varkw = parse_kwargs(Model, **kwargs)
        self.model = Model(**varkw)

    def train(self, train_data, valid_data=None):
        try:
            train_data.x = train_data.x.drop(DATACONF["ATTRIBUTES"], axis=1)
        except Exception as err:
            print(f"drop err: {err}")
        self.model.fit(train_data.x, train_data.y)

    def predict(self, data):
        # data -> image urls
        return self.model.predict(data)

    def load(self, model_url):
        self.model = joblib.load(model_url)

    def save(self, model_path):
        return joblib.dump(self.model, model_path)

    def evaluate(self, data, **kwargs):
        return self.model.evaluate(data, **kwargs)

    def dataset_feature_process(self, url, type="default"):
        data = CSVDataParse(data_type=type)
        data.parse(url, label="Thermal preference")
        return data
