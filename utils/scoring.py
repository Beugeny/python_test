import pandas as pd
from sklearn.metrics import mean_squared_error
import math


def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5


def rmse_scoring(estimator, x, y):
    # print(len(x), len(y))
    y_pred = estimator.predict(x)
    return rmse(y, y_pred)


# A function to calculate Root Mean Squared Logarithmic Error (RMSLE)
def rmsle(y_true, y_pred):
    if type(y_pred) is pd.DataFrame:
        y_pred = y_pred.values
    elif type(y_pred) is pd.Series:
        y_pred = y_pred.values
    if type(y_true) is pd.DataFrame:
        y_true = y_true.values
    elif type(y_true) is pd.Series:
        y_true = y_true.values

    assert len(y_true) == len(y_pred)

    for i, pred in enumerate(y_pred):
        if y_pred[i] + 1 < 0:
            y_pred[i] = 0

    terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y_true[i] + 1)) ** 2.0 for i, pred in enumerate(y_pred)]
    return (sum(terms_to_sum) * (1.0 / len(y_true))) ** 0.5


def rmlse_scoring(estimator, x, y):
    y_pred = estimator.predict(x)
    return rmsle(y, y_pred)
