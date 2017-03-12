from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt


def regression_res(y_pred, y_true):
    A = ((y_pred - y_true) ** 2).sum()
    B = ((y_true - y_true.mean()) ** 2).sum()
    print("sq_error={0}".format(A))
    # print("sum(y-mean(y))^2={0}".format(B))
    print("score={0}".format(1 - A / B))


def fit_predict(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    mdl = pd.DataFrame()
    mdl["Models"] = pd.Series([
        LinearRegression(),
        DecisionTreeRegressor(),
        RandomForestRegressor(),
        LogisticRegression(),
        SVR(),
        KNeighborsRegressor(),
    ])
    mdl.index = ["LinReg", "DecisionTree", "RandForest", "LogReg", "SVR", "KNeig"]

    for x in mdl["Models"].values:
        x.fit(X_train, y_train)
    mdl["predict_train"] = mdl["Models"].apply(lambda x: x.predict(X_train))
    mdl["predict_test"] = mdl["Models"].apply(lambda x: x.predict(X_test))
    mdl["score_train"] = mdl["Models"].apply(lambda x: x.score(X_train, y_train))
    mdl["score_test"] = mdl["Models"].apply(lambda x: x.score(X_test, y_test))
    print(mdl)

    plt.plot(mdl["score_train"].values, 'ro')
    plt.xticks(np.array(range(0, len(mdl["score_train"]))), mdl.index)
    plt.show()

    max_score = mdl.loc[mdl["score_train"] == mdl["score_train"].max()]

    return {"clf": max_score["Models"].values[0],
            "predict_train": max_score["predict_train"].values[0],
            "predict_test": max_score["predict_test"].values[0],
            "score_train": max_score["score_train"].values[0],
            "score_test": max_score["score_test"].values[0],
            "y_train":y_train,
            "y_test": y_test}
