from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

from utils.scoring import rmse_scoring, rmlse_scoring


def regression_res(y_pred, y_true):
    A = ((y_pred - y_true) ** 2).sum()
    B = ((y_true - y_true.mean()) ** 2).sum()
    print("sq_error={0}".format(A))
    # print("sum(y-mean(y))^2={0}".format(B))
    print("score={0}".format(1 - A / B))


def cross_val(clf, X, y):
    # print("Cross Validation: \n{0}".format(clf))
    return cross_val_score(clf, X, y, cv=4, scoring=rmlse_scoring).mean()


def fit_predict(X, y):
    poly = PolynomialFeatures(2, include_bias=False)
    # X = poly.fit_transform(X)
    # X = preprocessing.scale(X)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    mdl = pd.DataFrame()
    mdl["Models"] = pd.Series([
        LinearRegression(),
        DecisionTreeRegressor(),
        RandomForestRegressor(),
        SVR(),
        KNeighborsRegressor(),
    ])
    mdl.index = ["LinReg", "DecisionTree", "RandForest", "SVR", "KNeig"]
    mdl["names"] = mdl.index.values;
    mdl["score_train"] = mdl["Models"].apply(
        lambda x: cross_val(x, x_train, y_train))
    mdl["score_test"] = mdl["Models"].apply(
        lambda x: cross_val(x, x_test, y_test))

    mdl["Models"].apply(lambda x: x.fit(x_train, y_train))  # TODO cross validation fit
    mdl["predict_train"] = mdl["Models"].apply(lambda x: x.predict(x_train))
    mdl["predict_test"] = mdl["Models"].apply(lambda x: x.predict(x_test))
    print(mdl[["score_train", "score_test"]])

    plt.plot(mdl["score_train"].values, 'ro')
    plt.xticks(np.array(range(0, len(mdl["score_train"]))), mdl.index)
    plt.show()

    best_score = mdl.loc[mdl["score_train"] == mdl["score_train"].min()]

    clf = best_score["Models"].values[0]
    clf_name = best_score["names"].values[0]
    train_score = best_score["score_train"].values[0]
    test_score = best_score["score_test"].values[0]

    print("Best clf={0}. rmse={1}/{2}".format(clf_name, train_score, test_score))

    return {"clf": clf,
            "predict_train": best_score["predict_train"].values[0],
            "predict_test": best_score["predict_test"].values[0],
            "score_train": train_score,
            "score_test": test_score,
            "y_train": y_train,
            "y_test": y_test}
