from functools import wraps
from inspect import getfullargspec

def singleton(cls):
    """
    Set class to singleton class/
    :param cls: class
    :return: instance
    """
    __instance__ = {}

    @wraps(cls)
    def get_instance(*args, **kwargs):
        """Get class instance and save it into glob list"""
        if cls not in __instance__:
            __instance__[cls] = cls(*args, **kwargs)
        return __instance__[cls]
    return get_instance

def parse_kwargs(func, **kwargs):
    if not callable(func):
        return kwargs
    need_kw = getfullargspec(func)
    if need_kw.varkw == 'kwargs':
        return kwargs
    return {k: v for k, v in kwargs.items() if k in need_kw.args}

def module_to_class(module):
    class empty():
        def __init__(self):
            pass

    new_class = empty()
    for setting in dir(module):
       setattr(new_class, setting, getattr(module, setting))

    return new_class
