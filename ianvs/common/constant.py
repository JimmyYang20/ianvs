from enum import Enum


class DatasetFormat(Enum):
    """
    dataset format
    """
    CSV = "csv"
    TXT = "txt"


class ParadigmKind(Enum):
    """
    paradigm kind
    """
    LIFELONG_LEARNING = "lifelonglearning"
