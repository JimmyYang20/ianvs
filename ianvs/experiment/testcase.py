import os
from copy import deepcopy
from ianvs.experiment.pipeline import pipeline_generator

class TestCase():
    def __init__(self, test_env, algorithm, **kwargs):
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
        self.env = deepcopy(test_env)
        self.algorithm = algorithm.generate_job()

        if self.algorithm.paradigm_name == 'lifelonglearning':
            self.algorithm.kb_url = os.path.join(self.env.output_url, self.algorithm.model_name)

    def run(self):
        self.pipeline = pipeline_generator(self.env, self.algorithm)
        res = self.pipeline.run()
        # Create the result file by story setting
        metrics = [i for i in self.env.metrics]
        metric_res = {i: {} for i in metrics}
        for metric in self.env.metrics_fuc:
            metric_res[metric.__name__] = metric(res, self.env.test_data.y)
        return metric_res
