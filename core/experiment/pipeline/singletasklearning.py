import copy
import os
import shutil

import pandas as pd

from ianvs.experiment.testenv import TestEnv
from ianvs.common.constant import DatasetFormat
from sedna.datasources import CSVDataParse, TxtDataParse


class SingleTaskLearning:
    """ SingleTaskLearning pipeline """

    def __init__(self, testenv: TestEnv, algorithm, workspace):
        self.testenv = testenv
        self.algorithm = algorithm
        self.workspace = workspace
        self.prepare()

    def prepare(self):
        self.dataset_format = self.testenv.dataset.format
        self.job, self.feature_process = self.algorithm.build()

    def run(self):
        dataset = self.testenv.dataset
        job = self.job

        # model training
        train_dataset = self._process_dataset(self.testenv.dataset.train_dataset, dataset.label, "train")
        train_output_dir = os.path.join(self.workspace, "output/train")
        model_path = self.algorithm.initial_model_url
        if model_path:
            if os.path.isfile(model_path):
                new_model_path = os.path.join(train_output_dir, os.path.basename(model_path))
                shutil.copy(model_path, new_model_path)
            elif os.path.isdir(model_path):
                new_model_path = train_output_dir
                shutil.copytree(model_path, new_model_path)
            hyperparameters = {"initial_model_url": new_model_path}

        job.train(train_dataset, **hyperparameters)

        # save model
        model_path = job.save(train_output_dir)

        inference_dataset = self._process_dataset(self.testenv.dataset.eval_dataset, dataset.label, "inference")
        inference_output_dir = os.path.join(self.workspace, "output/inference")
        hyperparameters = {"inference_output_dir": inference_output_dir}
        job.load(model_path)
        res = job.predict(inference_dataset.x, **hyperparameters)
        return res

    def _save_inference_result(self, inference_result):
        output_dir = os.path.join(self.workspace, "precision")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if self.dataset_format == DatasetFormat.CSV.value:
            prediction_file = os.path.join(output_dir, "precison.csv")
            pd.DataFrame(inference_result).to_csv(prediction_file, index=False, encoding="utf-8", sep=" ", mode="w")
        elif self.dataset_format == DatasetFormat.TXT.value:
            pass

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
