from common.utils import parse_kwargs
from sedna.core.lifelong_learning import LifelongLearning
from sedna.core.incremental_learning import IncrementalLearning

job_dict = {'lifelong_learning': LifelongLearning,
            'incremental_learning': IncrementalLearning
            }

def paradigm_parameter_parse(**config):
    paradigm_name = config.pop('paradigm')
    paradigm_algorithm = job_dict.get(paradigm_name)
    paradigm_config = parse_kwargs(paradigm_algorithm, **config)
    return paradigm_name, paradigm_config

