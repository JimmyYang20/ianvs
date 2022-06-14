import copy
import os
import pandas as pd

from ianvs.common import utils
from ianvs.experiment.testenv import TestEnv
from sedna.common.constant import KBResourceConstant
from ianvs.common.constant import DatasetFormat
from sedna.core.lifelong_learning import LifelongLearning
from sedna.datasources import CSVDataParse, TxtDataParse


class LifelongLearning:
    """ Lifelong_learning pipeline """

    def __init__(self, testenv: TestEnv, algorithm, workspace):
        self.testenv = testenv
        self.algorithm = algorithm
        self.workspace = workspace
        self.prepare()

    def prepare(self):
        self.incremental_rounds = self.testenv.incremental_rounds
        self.dataset_format = self.testenv.dataset.format
        self.dataset_train_ratio = self.algorithm.dataset_train_ratio
        self.job, self.feature_process = self.algorithm.build()
        self.local_task_index_map = {}

    def run(self):
        rounds = self.incremental_rounds + 1
        dataset = self.testenv.dataset

        try:
            dataset_files = self._preprocess_dataset()
            if len(dataset_files) != rounds:
                raise Exception(
                    f"sum({len(dataset_files)}) of dataset files not equal rounds({rounds}) of lifelonglearning job.")
        except Exception as err:
            raise Exception(f"preprocess dataset failed, error: {err}.")

        job = self.job
        for r in range(1, rounds + 1):
            self._set_local_knowledgebase_config(job, r)

            train_dataset_file, eval_dataset_file = dataset_files[r - 1]
            train_dataset = self._process_dataset(train_dataset_file, dataset.label, "train")
            if r == 1:
                job.train(train_dataset)
            else:
                pre_task_index_url = self.local_task_index_map.get(r - 1)
                job.update(train_dataset, task_index_url=pre_task_index_url)

            eval_dataset = self._process_dataset(eval_dataset_file, dataset.label, "eval")
            model_eval = copy.deepcopy(self.testenv.model_eval)
            metric_info = model_eval.pop("model_metric")

            metric_res, index_file = job.evaluate(eval_dataset, metrics=utils.get_metric(metric_info),
                                                  metrics_param=metric_info.get("parameters"), **model_eval)


            inference_dataset = self._process_dataset(self.testenv.dataset.eval_dataset, dataset.label, "test")
            res, is_unseen_task, tasks = job.inference(inference_dataset, index_file=index_file)
            self._save_inference_result(res, r)
        return res

    def _save_inference_result(self, inference_result, round):
        output_dir = os.path.join(self.workspace, "precision", str(round))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if self.dataset_format == DatasetFormat.CSV.value:
            prediction_file = os.path.join(output_dir, "precison.csv")
            pd.DataFrame(inference_result).to_csv(prediction_file, index=False, encoding="utf-8", sep=" ", mode="w")
        elif self.dataset_format == DatasetFormat.TXT.value:
            pass

    def _set_local_knowledgebase_config(self, paradigm_job: LifelongLearning, round):
        output_dir = os.path.join(self.workspace, "knowledgeable", str(round))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        paradigm_job.config.output_url = output_dir

        task_index = os.path.join(output_dir, KBResourceConstant.KB_INDEX_NAME.value)
        paradigm_job.config.task_index = task_index

        self.local_task_index_map[round] = task_index

    def _preprocess_dataset(self):
        dataset = self.testenv.dataset
        output_dir = os.path.join(dataset.output_dir, "lifelonglearning")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        dataset_train_ratio = self.dataset_train_ratio
        splitting_times = self.incremental_rounds + 1
        dataset_files = dataset.splitting_more_times(dataset.train_dataset, dataset.format, dataset_train_ratio,
                                                     output_dir, times=splitting_times)
        return dataset_files

    def _process_dataset(self, file, label, type):
        feature_process = self.feature_process
        if self.dataset_format == DatasetFormat.CSV.value:
            dataset = CSVDataParse(data_type=type, func=feature_process)
            dataset.parse(file, label=label)
        elif self.dataset_format == DatasetFormat.TXT.value:
            dataset = TxtDataParse(data_type=type, func=feature_process)
            dataset.parse(file)
        else:
            raise ValueError(f"dataset(file={file})'s format({self.dataset_format}) is not supported.")

        return dataset
