import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
import numpy as np

from utils.scoring import rmlse_scoring, rmsle


def select_model(x_train, y_train):
    mdl = pd.DataFrame()
    mdl["Models"] = pd.Series([
        GradientBoostingRegressor(random_state=123),
        # LinearRegression(),
        # DecisionTreeRegressor(),
        # ch_meta_rand_forest(x_train, y_train),
        # SVR(),
        ch_meta_heigbors(x_train, y_train),
    ])
    mdl.index = ["Boosting", "KNeig"]
    mdl["names"] = mdl.index.values

    mdl["score_train"] = mdl["Models"].apply(
        lambda x: cross_val(x, x_train, y_train))

    mdl["Models"].apply(lambda x: x.fit(x_train, y_train))  # TODO cross validation fit
    mdl["predict_train"] = mdl["Models"].apply(lambda x: x.predict(x_train))

    print(mdl["score_train"])
    best_score = mdl.loc[mdl["score_train"] == mdl["score_train"].min()]

    clf = best_score["Models"].values[0]
    clf_name = best_score["names"].values[0]
    train_score = best_score["score_train"].values[0]

    print("Best clf={0}. rmse={1}".format(clf_name, train_score))

    return {"clf": clf,
            "predict_train": best_score["predict_train"].values[0],
            "score_train": train_score}


def cross_val(clf, X, y):
    return cross_val_score(clf, X, y, cv=5, scoring=rmlse_scoring).mean()


def ch_meta_heigbors(x_train, y_train):
    return grid_search_cv(KNeighborsRegressor(),
                          [{'n_neighbors': np.arange(2, 15), 'weights': ["uniform", "distance"]}], x_train, y_train)


def ch_meta_svr(x_train, y_train):
    return grid_search_cv(SVR(), [
        {'kernel': ["rbf", "sigmoid"], "C": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 100000, 1000000]}], x_train,
                          y_train)


def ch_meta_rand_forest(x_train, y_train):
    return grid_search_cv(RandomForestRegressor(random_state=123), [
        {'n_estimators': list(np.arange(5, 20, 2)), "min_samples_split": [2, 3, 4, 5]}], x_train,
                          y_train)


def grid_search_cv(clf, tuned_params, X, y):
    sc1 = make_scorer(rmsle, greater_is_better=False)
    grd = GridSearchCV(clf, tuned_params, cv=5, scoring=sc1, verbose=1)

    grd.fit(X, y)
    print("Best={0}, score={1}".format(grd.best_params_, cross_val(grd.best_estimator_, X, y)))
    print(grd.best_estimator_)
    return grd.best_estimator_
