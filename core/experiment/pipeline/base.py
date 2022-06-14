from ianvs.common.constant import ParadigmKind
from ianvs.common import utils
from ianvs.experiment.testenv import TestEnv
from ianvs.experiment.pipeline.singletasklearning import SingleTaskLearning
from ianvs.experiment.pipeline.incrmentallearning import IncrementalLearning


class Base:
    def __init__(self, test_env: TestEnv, algorithm, workspace):
        self.test_env = test_env
        self.dataset = test_env.dataset
        self.algorithm = algorithm
        self.workspace = workspace

    def build(self, kind):
        if kind == ParadigmKind.SingleTaskLearning.value:
            pipeline = SingleTaskLearning(self.test_env, self.algorithm, self.workspace)
        elif kind == ParadigmKind.IncrementalLearning.value:
            pipeline = IncrementalLearning(self.test_env, self.algorithm, self.workspace)
        return pipeline

    def load_data(self, file: str, data_type: str, label=None, use_raw=True, feature_process=None):
        from sedna.datasources import CSVDataParse, TxtDataParse
        from ianvs.common.constant import DatasetFormat
        format = utils.get_file_format(file)

        if format == DatasetFormat.CSV.value:
            data = CSVDataParse(data_type=data_type, func=feature_process)
            data.parse(file, label=label)
        elif format == DatasetFormat.TXT.value:
            data = TxtDataParse(data_type=data_type, func=feature_process)
            data.parse(file, use_raw=use_raw)

        return data

    def eval_overall(self, result):
        """ eval overall results """
        metric_funcs = []
        for metric_dict in self.metrics:
            metric = utils.get_metric_func(metric_dict=metric_dict)
            if callable(metric):
                metric_funcs.append(metric)

        eval_dataset_file = self.dataset.eval_dataset
        eval_dataset = self.load_data(eval_dataset_file, data_type="eval overall", label=self.dataset.label)
        metric_res = {}
        for metric in metric_funcs:
            metric_res[metric.__name__] = metric(eval_dataset.y, result)
        return metric_res
