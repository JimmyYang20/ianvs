from sklearn.model_selection import train_test_split
from sedna.datasources import CSVDataParse
from sedna.datasources import BaseDataSource
from ianvs.common.metric import get_metric

class TestEnv:
    def __init__(self):
        self.dataset_url = ""
        self.train_ratio = 0.8
        self.label = []
        self.model_eval = {}
        self.metrics = []
        self.rank = {"switch": False}
        self.visualization_models = "off"
        self.output_url = "./test/"

    #输入输出
    def build(self):
        if self.dataset_url.split('.')[1] == 'csv':
            dataset = CSVDataParse(data_type='train')
            dataset.parse(self.dataset_url, label=self.label)
        else:
            raise NotImplementedError
        # Split the data to train set and test set
        train_data = BaseDataSource(data_type='train')
        test_data = BaseDataSource(data_type='test')
        train_data.x, test_data.x, train_data.y, test_data.y = train_test_split(dataset.x,
                                                                                dataset.y,
                                                                                test_size=1-self.train_ratio,
                                                                                random_state=42,
                                                                                shuffle=False)
        self.train_data = train_data
        self.test_data = test_data
        self.metrics_fuc = [get_metric(i) for i in self.metrics]

