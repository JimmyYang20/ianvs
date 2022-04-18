import os
import uuid
from copy import deepcopy
from inspect import isfunction

from .pipeline import Pipeline
from ianvs.common.metrics import get_metric_func
from ..common import utils


class TestCase:
    def __init__(self, testenv, algorithm):
        """
        Distributed collaborative AI algorithm in certain test environment
        Parameters
        ----------
        test_env : instance
            The test environment of  distributed collaborative AI benchmark
            including samples, dataset setting, metrics
        algorithm : instance
            Distributed collaborative AI algorithm
        """
        self.testenv = testenv
        self.algorithm = algorithm

    def build(self, metrics, workspace):
        self.id = self._get_id()
        self.output_dir = self._get_output_dir(workspace)
        self.metrics = metrics
        self.pipeline = self._build_pipeline()

    def _build_pipeline(self):
        pipeline = Pipeline(self.testenv, self.algorithm, self.output_dir)
        return pipeline.build(self.algorithm.paradigm)

    def _get_output_dir(self, workspace):
        output_dir = os.path.join(workspace, self.algorithm.name)
        flag = True
        while flag:
            output_dir = os.path.join(workspace, self.algorithm.name, str(self.id))
            if not os.path.exists(output_dir):
                flag = False
        return output_dir

    def _get_id(self):
        return uuid.uuid1()

    def _get_metric_funcs(self, metrics: list):
        metrics_funcs = []

        try:
            for metric_dict in metrics:
                metric = utils.get_metric(metric_dict)
                if isfunction(metric):
                    metrics_funcs.append(metric)
                elif isinstance(metric, str):
                    func = get_metric_func(metric)
                    metrics_funcs.append(func)
        except Exception as err:
            raise NotImplementedError(f"get metrics func failed, error: {err}.")
        return metrics_funcs

    def run(self):
        try:
            res = self.pipeline.run()
        except Exception as err:
            raise Exception(f"pipeline (paradigm={self.algorithm.paradigm}) runs failed, error: {err}")
        try:
            eval_result = self._eval(res)
            return eval_result
        except Exception as err:
            raise Exception(f"eval pipeline(paradigm name={self.algorithm.paradigm})'s result failed, error: {err}")

    def _eval(self, result):
        """ eval pipeline's result """
        eval_dataset = self.testenv.dataset.get_eval_dataset()
        metric_funcs = self._get_metric_funcs(self.metrics)
        metric_res = {}
        for metric in metric_funcs:
            metric_res[metric.__name__] = metric(eval_dataset.y, result)
        return metric_res
