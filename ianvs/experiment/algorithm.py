import joblib
import json
import os
from inspect import getfullargspec
from sedna.core.lifelong_learning import LifelongLearning
from sedna.core.incremental_learning import IncrementalLearning
from ianvs.common.config import BaseConfig
from ianvs.common.utils import parse_kwargs
from copy import deepcopy

class Algorithm:
    def __init__(self):
        self.paradigm = Paradigm()
        self.basemodel = BaseModel()

    def generate_job(self):
        if self.basemodel.hyperparameters:
            #TODO: 根据超参为job起名, 需重构写法
            model_kw = []
            for config_param, value in self.basemodel.hyperparameters.items():
                if isinstance(value, int):
                    model_kw += [config_param, str(value)]
                elif isinstance(value, float):
                    model_kw += [config_param, str(value)]
                else:
                    model_kw += [config_param, '_'.join(map(str, value))]
            estimator = SednaMultiTask(self.basemodel.name, **self.basemodel.hyperparameters)
            estimator_name = [self.basemodel.name] + model_kw
            estimator.__name__ = '_'.join(estimator_name)
        else:
            estimator = SednaMultiTask(self.basemodel.name)
            estimator.__name__ = self.basemodel.name
        if self.paradigm.kind == 'incrementallearning' or self.paradigm.kind == 'lifelonglearning':
            paradigm = job_dict.get(self.paradigm.kind)

            #处理范式超参数
            need_kw = getfullargspec(paradigm)
            input_param = {}
            for i in deepcopy(self.paradigm.funcs):
                if i.get('kind') in need_kw.args:
                    if 'parameters' in i.keys():
                        i['param'] = json.dumps(i.pop('parameters'))
                    input_param[i.pop('kind')] = i

            job = paradigm(estimator=estimator, **input_param)
            job.model_name = estimator.__name__
            job.paradigm_name = self.paradigm.kind
        else:
            raise NotImplementedError
        return job




class BaseModel:
    def __init__(self):
        self.name = ""
        self.hyperparameters = {}
        self.hyperparameter_file = ""
        self.multi_parameters = []


class Paradigm:
    def __init__(self):
        self.kind = ""
        self.incremental_rounds = 2
        self.funcs = []

class SednaMultiTask:
    '''
    Algorithm
    '''
    def __init__(self, base_model, random_seed=42, **kwargs):
        self.__name__ = base_model
        if base_model == 'ada_boost':
            from sklearn.ensemble import AdaBoostClassifier as Model
            self.model = Model(random_state=random_seed)
        elif base_model == 'svm':
            from sklearn.svm import SVC as Model
            self.model = Model(random_state=random_seed)
        elif base_model == 'mlp':
            from sklearn.neural_network import MLPClassifier as Model
            self.model = Model(random_state=random_seed)
        elif base_model == 'random_forest':
            from sklearn.ensemble import RandomForestClassifier as Model
            self.model = Model(random_state=random_seed)
        elif base_model == 'xgboost':
            from xgboost import XGBClassifier as Model
            varkw = parse_kwargs(Model, **kwargs)
            self.model = Model(**varkw)
        else:
            raise NotImplementedError

    def train(self, train_data, valid_data=None):
        self.model.fit(train_data.x, train_data.y)

    def predict(self, data):
        '''Model inference'''
        return self.model.predict(data)

    def load(self, model_url):
        print('model_url: ', model_url)
        self.model = joblib.load(model_url)
        return self

    def save(self, model_path):
        print('model_path: ', model_path)
        return joblib.dump(self.model, model_path)


job_dict = {'lifelonglearning': LifelongLearning,
            'incrementallearning': IncrementalLearning
            }