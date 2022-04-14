import os
import pandas as pd
from sedna.common.constant import KBResourceConstant


class PipeLine():
    """
    Benchmark evaluate the distributed collaborative AI algorithm performance in certain test environment

    Parameters
    ----------
    test_case : Instance
        Distributed collaborative AI algorithm in certain test environment
    """
    def __init__(self, test_env, algorithm):
        self.algorithm = algorithm
        self.train_data = test_env.train_data
        self.test_data = test_env.test_data
        self.metrics = test_env.metrics
        self.res_path = test_env.output_url

class LLPipeLine(PipeLine):
    '''
    Lifelong_learning pipeline
    '''
    def __init__(self, test_env, algorithm):
        super(LLPipeLine, self).__init__(test_env=test_env, algorithm=algorithm)
        algorithm.config.output_url = algorithm.kb_url
        algorithm.config.task_index = os.path.join(algorithm.kb_url,\
                                                   KBResourceConstant.KB_INDEX_NAME.value)

    def evaluate(self, result_path):
        res, is_unseen_task, tasks = self.algorithm.inference(self.test_data)
        # Check if there is no result folder to save experiment result, if not then create one
        if result_path is None:
            result_path = BaseConfig.result_path
            if not os.path.exists(result_path):
                os.mkdir(result_path)

        prediction_path = self.algorithm.paradigm_name + '_' + self.algorithm.model_name \
                          + '_prediction.csv'

        res_path = os.path.join(result_path, prediction_path)
        pd.DataFrame(res).to_csv(res_path, header=False, index=False)

        # Create the result file by story setting
        metrics = [i.__name__ for i in self.metrics]
        metric_res = {i: {} for i in metrics}

        for metric in self.metrics:
            metric_res[metric.__name__] = metric(res, self.test_data.y)

        return metric_res

    def run(self, **kwargs):
        #TODO: 增量
        self.algorithm.train(self.train_data, **kwargs)
        res, is_unseen_task, tasks = self.algorithm.inference(self.test_data)
        # Check if there is no result folder to save experiment result, if not then create one
        if not os.path.exists(self.res_path):
            os.mkdir(self.res_path)
        prediction_path = os.path.join(self.res_path, 'prediction')

        prediction_file = self.algorithm.paradigm_name + '_' + self.algorithm.model_name \
                          + '_prediction.csv'
        if not os.path.exists(prediction_path):
            os.mkdir(prediction_path)

        res_path = os.path.join(prediction_path, prediction_file)
        pd.DataFrame(res).to_csv(res_path, header=False, index=False)
        return res

class ILPipeLine(PipeLine):
    '''
    Incremental learning pipeline
    '''
    def __init__(self, test_env, algorithm):
        super(ILPipeLine, self).__init__(test_env=test_env, algorithm=algorithm)


    def evaluate(self, result_path, **kwargs):
        res = self.algorithm.inference(self.test_data, **kwargs)
        if result_path is None:
            result_path = BaseConfig.result_path
            if not os.path.exists(result_path):
                os.mkdir(result_path)

        # prediction_file = self.algorithm.paradigm_name + '_' + self.algorithm.model_name \
        #                   + '_prediction.csv'
        #
        # res_path = os.path.join(result_path, prediction_file)
        # pd.DataFrame(res).to_csv(res_path, header=None, index=None)

        # Create the result file by story setting
        # metrics = [i.__name__ for i in self.metrics]
        # metric_res = {i: {} for i in metrics}
        #
        # for metric in self.metrics:
        #     metric_res[metric.__name__] = metric(res, self.test_data.y)

        return res[0].get('metrics')

    def run(self, **kwargs):
        self.algorithm.train(self.train_data, **kwargs)
        res, is_unseen_task, tasks = self.algorithm.inference(self.test_data)
        # Check if there is no result folder to save experiment result, if not then create one
        result_path = kwargs.get('result_path')
        if result_path is None:
            result_path = BaseConfig.result_path
            if not os.path.exists(result_path):
                os.mkdir(result_path)

        prediction_path = self.algorithm.paradigm_name + '_' + self.algorithm.model_name \
                          + '_prediction.csv'

        res_path = os.path.join(result_path, prediction_path)
        pd.DataFrame(res).to_csv(res_path, header=False, index=False)

        # Create the result file by story setting
        metrics = [i.__name__ for i in self.metrics]
        metric_res = {i: {} for i in metrics}

        for metric in self.metrics:
            metric_res[metric.__name__] = metric(res, self.test_data.y)

        return metric_res

def pipeline_generator(test_env, algorithm):
    if algorithm.paradigm_name == 'lifelonglearning':
        pipeline = LLPipeLine
    elif algorithm.paradigm_name == 'incrementallearning':
        pipeline = ILPipeLine
    else:
        raise
    return pipeline(test_env, algorithm)