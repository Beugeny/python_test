from pandas import Series
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures


def eng(df):
    res = pd.DataFrame()
    if len(df["LotArea"]) != len(df["LotArea"].dropna()):
        raise ValueError('Lot area values can not be nan')

    res["LotArea"] = df["LotArea"]

    poly = PolynomialFeatures(4, include_bias=False)
    res = poly.fit_transform(res)

    res = preprocessing.scale(res)

    return res
