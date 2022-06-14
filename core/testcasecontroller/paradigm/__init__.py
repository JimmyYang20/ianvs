from .singletask_learning import SingleTaskLearning
from .incremental_learning import IncrementalLearning
from ianvs.common.constant import ParadigmKind


def Paradigm(kind, test_env, algorithm, workspace):
    if kind == ParadigmKind.SingleTaskLearning.value:
         return SingleTaskLearning(test_env, algorithm, workspace)
    elif kind == ParadigmKind.IncrementalLearning.value:
        return IncrementalLearning(test_env, algorithm, workspace)
