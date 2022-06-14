import os
import uuid


from ianvs.experiment.pipeline import Pipeline



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

    def run(self):
        try:
            res = self.pipeline.run()
        except Exception as err:
            raise Exception(f"pipeline (paradigm={self.algorithm.paradigm}) runs failed, error: {err}")

