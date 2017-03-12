from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import pandas as pd
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
    mdl = pd.DataFrame()
    mdl["Models"] = pd.Series([
        LinearRegression(),
        DecisionTreeRegressor(),
        RandomForestRegressor(),
        LogisticRegression(),
        SVR(),
        KNeighborsRegressor(),
    ])
    mdl.index = mdl["Models"].apply(lambda x: type(x)).values

    for x in mdl["Models"].values:
        x.fit(X, y)

    mdl["fit_result"] = mdl["Models"].apply(lambda x: x.predict(X, y))
    mdl["score"] = mdl["Models"].apply(lambda x: x.score(X, y))

    print(mdl)

    plt.plot(mdl["score"])
    plt.show()
