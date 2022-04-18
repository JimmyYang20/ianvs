import os
from ianvs.common import utils
from FPN_TensorFlow.interface import Estimator as Model
from sedna.common.class_factory import ClassType, ClassFactory

os.environ['BACKEND_TYPE'] = 'TENSORFLOW'

__all__ = ["BaseModel"]


@ClassFactory.register(ClassType.GENERAL, "estimator")
class BaseModel:
    def __init__(self, **kwargs):
        varkw = utils.parse_kwargs(Model, **kwargs)
        self.model = Model(**varkw)

    def train(self, train_data, valid_data=None):
        return self.model.train(train_data)

    def predict(self, data):
        # data -> image urls
        return self.model.predict(data)

    def load(self, model_url):
        self.model.load(model_url)

    def save(self, model_path):
        return self.model.save(model_path)

    def evaluate(self, data, **kwargs):
        return self.model.evaluate(data, **kwargs)

    def task_division(self, samples, threshold):
        return self.model.task_division(samples, threshold)


