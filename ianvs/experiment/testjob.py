import copy
from itertools import product

from .testenv import TestEnv
from .testcase import TestCase
from .algorithm import Algorithm, Paradigm, BaseModel

import os
import csv
from tabulate import tabulate
import pandas as pd

class TestJob:
    def __init__(self, config):
        self.name = ""
        self.test_env = TestEnv()
        self.algorithm_list = []
        self.parse(config)
        self._run()

    def run(self):
        self.prepare_job()
        self.run_job()
        pass

    def run_job(self):
        pass

    def prepare_job(self):
        #移除
        # env preare_env()
        # build_testcases()
        pass

    def _run(self):
        self.test_env.build()
        self.test_cases = []
        job_res = []
        for algorithm in self.algorithm_list:
        #单独使用函数构建testcase
            self.test_cases.append(TestCase(self.test_env, algorithm))
        for test_case in self.test_cases:
            job_res.append(test_case.run())
        story_generator(job_res, self.test_env.output_url, self.test_cases, self.test_env.metrics)
        print()

        pass

    def parse(self, test_job_config):
        for key in test_job_config:
            v = test_job_config[key]
            if key == str.lower(TestEnv.__name__):
                self.dict_to_object(self.test_env, v)
            elif key == str.lower(Algorithm.__name__):
                self.parse_algorithms_config(v)
            else:
                self.__dict__[key] = test_job_config[key]

    def parse_algorithms_config(self, algorithms_config):
        def parse_basemodels_config(basemodels_config):
            basemodel_list = []
            for b in basemodels_config:
                parameters_list = []
                for parameter in b.multi_parameters:
                    parameters_list.append(b.hyperparameters[parameter])
                for parameter_list in product(*parameters_list):
                    basemodel = copy.deepcopy(b)
                    for i in range(len(b.multi_parameters)):
                        basemodel.hyperparameters[b.multi_parameters[i]] = parameter_list[i]
                    basemodel_list.append(basemodel)
            return basemodel_list

        def parse_algorithms_config(key, cls):
            objects = []
            configs = algorithms_config[key]
            if len(configs) > 0:
                for config in configs:
                    obj = cls()
                    self.dict_to_object(obj, config)
                    objects.append(obj)
            if cls == BaseModel:
                return parse_basemodels_config(objects)
            return objects

        paradigms = parse_algorithms_config(str.lower(Paradigm.__name__) + "s", Paradigm)
        basemodels = parse_algorithms_config(str.lower(BaseModel.__name__) + "s", BaseModel)

        for p in paradigms:
            for b in basemodels:
                algorithm = Algorithm()
                algorithm.paradigm = p
                algorithm.basemodel = b
                self.algorithm_list.append(algorithm)

    def dict_to_object(self, obj, object_config):
        for k in object_config:
            obj.__dict__[k] = object_config[k]

def story_generator(test_res, res_path, test_cases, metrics, ranking=True):
    """
    Benchmark result, provides the insight regarding the algorithm performance

    Parameters
    ----------
    test_res : Dict
        Multiple distributed collaborative AI algorithm in certain test environment

    res_path : String
        Any valid string path is acceptable.

    algorithms : Instance
        Distributed collaborative AI algorithm

    metrics : List
        Measure algorithm performance

    ranking : Bool
        Show and save the ranking list or not.
    """
    rank_path = os.path.join(res_path, 'rank')
    if not os.path.exists(rank_path):
        os.mkdir(rank_path)

    metrics = [j for j in metrics]

    for i, res in enumerate(test_res):

        if ranking:
            rank_file = os.path.join(rank_path, 'ranking_list.csv')
            if not os.path.isfile(rank_file):
                with open(rank_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Paradigm'] + ['Base_model'] + metrics)
            values = [res[j] for j in metrics]
            with open(rank_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([test_cases[i].algorithm.paradigm_name] + [test_cases[i].algorithm.model_name] + values)

    if ranking:
        rank_file = os.path.join(rank_path, 'ranking_list.csv')
        table = pd.read_csv(rank_file)
        table.sort_values(by=['smape'],inplace=True)
        table.to_csv(rank_file, index=None)
        print(tabulate(table, headers='keys', tablefmt="grid"))
