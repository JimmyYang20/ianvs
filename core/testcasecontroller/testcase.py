import os
import uuid

from ianvs.testcasecontroller.paradigm import Paradigm


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

    def prepare(self, metrics, workspace):
        self.id = self._get_id()
        self.output_dir = self._get_output_dir(workspace)
        self.metrics = metrics

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

    def run(self):
        try:
            paradigm_pipeline = Paradigm(self.algorithm.kind, self.testenv, self.algorithm, self.output_dir)
            res = paradigm_pipeline.run()
        except Exception as err:
            raise Exception(f"(paradigm={self.algorithm.paradigm}) pipeline runs failed, error: {err}")
        return res
