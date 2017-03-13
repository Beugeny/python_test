from pandas import Series
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures


def eng(df):
    # Этот метод не должен удалять ни один столбец
    res = pd.DataFrame()
    if len(df["LotArea"]) != len(df["LotArea"].dropna()):
        raise ValueError('Lot area values can not be nan')

    res["LotArea"] = df["LotArea"]


    # poly = PolynomialFeatures(3, include_bias=False)
    # res = poly.fit_transform(res)

    # scaler = preprocessing.StandardScaler().fit(res)
    # res = scaler.transform(res)

    return res
