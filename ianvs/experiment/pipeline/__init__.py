from ianvs.common.constant import ParadigmKind
from ianvs.experiment.pipeline.lifelonglearning import LifelongLearning


class Pipeline:
    def __init__(self, test_env, algorithm, workspace):
        self.test_env = test_env
        self.algorithm = algorithm
        self.workspace = workspace

    def build(self, kind):
        if kind == ParadigmKind.LIFELONG_LEARNING.value:
            pipeline = LifelongLearning(self.test_env, self.algorithm, self.workspace)
            return pipeline

