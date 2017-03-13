from pandas import Series
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures


def eng(df):
    # Этот метод не должен удалять ни один столбец
    res = pd.DataFrame()
    if len(df["LotArea"]) != len(df["LotArea"].dropna()):
        raise ValueError('Lot area values can not be nan')

    if len(df["HouseStyle"]) != len(df["HouseStyle"].dropna()):
        raise ValueError('HouseStyle values can not be nan')

    res["LotArea"] = df["LotArea"]

    mapping = {'2.5Fin': 1, '2.5Unf': 2, '1.5Fin': 3, '1Story': 4, '2Story': 5, 'SLvl': 6, 'SFoyer': 7, '1.5Unf': 8}
    res["HouseStyle"] = df.replace({'HouseStyle': mapping})["HouseStyle"]
    # print(set(res["HouseStyle"]))

    # poly = PolynomialFeatures(3, include_bias=False)
    # res = poly.fit_transform(res)
    # res = poly.fit_transform(res)

    scaler = preprocessing.StandardScaler().fit(res)
    res = scaler.transform(res)

    return res
