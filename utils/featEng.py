import numpy as np


def isExist(x):
    if type(x) is str:
        if x == "nan":
            return 0
        else:
            return 1
    elif type(x) is float:
        if np.isnan(x):
            return 0
        else:
            return 1
    raise TypeError('Type not supported')
