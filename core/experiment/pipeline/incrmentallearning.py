import os

from . import Pipeline
from ianvs.experiment.testenv import TestEnv
from ianvs.experiment.algorithm import Algorithm


class IncrementalLearning(Pipeline):
    """ IncrementalLearning pipeline """

    def __init__(self, test_env: TestEnv, algorithm: Algorithm, workspace: str):
        super(IncrementalLearning, self).__init__(test_env, algorithm, workspace)
        self.prepare()

    def prepare(self):
        self.dataset_format = self.test_env.dataset.format
        self.dataset_train_ratio = self.algorithm.dataset_train_ratio

    def run(self):
        rounds = self.test_env.incremental_rounds + 1

        try:
            dataset_files = self.preprocess_dataset(splitting_times=rounds)
        except Exception as err:
            raise Exception(f"preprocess dataset failed, error: {err}.")

        for r in range(1, rounds + 1):
            train_dataset_file, eval_dataset_file = dataset_files[r - 1]

            train_output_dir = os.path.join(self.workspace, f"output/train/{r}")
            os.environ["MODEL_URL"] = train_output_dir
            if r == 1:
                model_url = self.algorithm.initial_model_url
            os.environ["BASE_MODEL_URL"] = model_url

            job, feature_process = self.algorithm.build()
            train_dataset = self.load_data(train_dataset_file, "train", feature_process=feature_process)
            job.train(train_dataset)
            print("test")

            # # else:
            # #     job.update(train_dataset, task_index_url=pre_task_index_url)
            #
            # eval_dataset = self._process_dataset(eval_dataset_file, dataset.label, "eval")
            # model_eval = copy.deepcopy(self.testenv.model_eval)
            # metric_info = model_eval.pop("model_metric")
            #
            # metric_res, index_file = job.evaluate(eval_dataset, metrics=utils.get_metric(metric_info),
            #                                       metrics_param=metric_info.get("parameters"), **model_eval)
            #
            # inference_dataset = self._process_dataset(self.testenv.dataset.eval_dataset, dataset.label, "test")
            # res, is_unseen_task, tasks = job.inference(inference_dataset, index_file=index_file)
            # self._save_inference_result(res, r)
        # return res

    # def _save_inference_result(self, inference_result, round):
    #     output_dir = os.path.join(self.workspace, "precision", str(round))
    #     if not os.path.exists(output_dir):
    #         os.makedirs(output_dir)
    #
    #     if self.dataset_format == DatasetFormat.CSV.value:
    #         prediction_file = os.path.join(output_dir, "precison.csv")
    #         pd.DataFrame(inference_result).to_csv(prediction_file, index=False, encoding="utf-8", sep=" ", mode="w")
    #     elif self.dataset_format == DatasetFormat.TXT.value:
    #         pass

    # def _preprocess_dataset(self):
    #     output_dir = os.path.join(self.workspace, "dataset")
    #
    #     output_dir = os.path.join(self.dataset.output_dir, "lifelonglearning")
    #     if not os.path.exists(output_dir):
    #         os.makedirs(output_dir)
    #
    #     dataset_train_ratio = self.dataset_train_ratio
    #     splitting_times = self.incremental_rounds + 1
    #     dataset_files = dataset.splitting_more_times(dataset.train_dataset, dataset.format, dataset_train_ratio,
    #                                                  output_dir, times=splitting_times)
    #     return dataset_files
    #
    # def _process_dataset(self, file, label, type, feature_process=None):
    #     if self.dataset_format == DatasetFormat.CSV.value:
    #         dataset = CSVDataParse(data_type=type, func=feature_process)
    #         dataset.parse(file, label=label)
    #     elif self.dataset_format == DatasetFormat.TXT.value:
    #         dataset = TxtDataParse(data_type=type, func=feature_process)
    #         dataset.parse(file)
    #     else:
    #         raise ValueError(f"dataset(file={file})'s format({self.dataset_format}) is not supported.")
    #
    #     return dataset
