import json
import os
from copy import deepcopy
from inspect import getfullargspec

from ianvs.common.constant import ParadigmKind
from ianvs.common import utils
from sedna.common.class_factory import ClassFactory, ClassType


class Paradigm:
    def __init__(self):
        self.name: str = ""
        self.train_data_ratio: float = 0.8
        self.basemodel: BaseModel = BaseModel()
        self.modules: list = []

    def check_fields(self):
        if not self.name:
            raise ValueError(f"not found paradigm name({self.name}).")
        self.basemodel.check_fields()
        for m in self.modules:
            m.check_fields()

    def build(self):
        base_job, feature_process = self.basemodel.load()
        modules = self.modules
        job = base_job
        if self.name == ParadigmKind.LIFELONG_LEARNING.value:
            from sedna.core.lifelong_learning import LifelongLearning
            need_kw = getfullargspec(LifelongLearning)

            module_funcs = {}
            for module in deepcopy(modules):

                if module.name in need_kw.args:
                    func = {"method": module.method}
                    if module.parameters:
                        func["param"] = json.dumps(module.parameters)
                    module_funcs[module.name] = func
            job = LifelongLearning(estimator=base_job, **module_funcs)

        return job, feature_process


class BaseModel:
    def __init__(self):
        self.name: str = ""
        self.url: str = ""
        self.feature_process: str = ""
        self.hyperparameters: dict = {}

    def check_fields(self):
        if not self.name:
            raise ValueError(f"not found basemodel name({self.name}).")
        if not self.url:
            raise ValueError(f"not found basemodel url({self.url}.")
        if not os.path.exists(self.url):
            raise ValueError(f"basemodel url({self.url}) does not exist on the local host.")

    def load(self):
        utils.load_module(self.url)
        try:
            estimator = ClassFactory.get_cls(type_name=ClassType.GENERAL, t_cls_name=self.name)(
                **self.hyperparameters)
        except Exception as err:
            raise Exception(f"basemodel (url={self.url}) loads class(name={self.name}), error: {err}.")

        features_process_func = None
        if self.feature_process:
            try:
                features_process_func = ClassFactory.get_cls(type_name=ClassType.GENERAL,
                                                             t_cls_name=self.feature_process)
            except Exception as err:
                raise Exception(f"basemodel (url={self.url}) loads dataset feature process failed, error: {err}.")

        return estimator, features_process_func


class Module:
    def __init__(self):
        self.name: str = ""
        self.method: str = ""
        self.parameters: dict = {}

    def check_fields(self):
        if not self.name:
            raise ValueError(f"not found module name({self.name}).")
        if not self.method:
            raise ValueError(f"not found module method({self.method}).")
