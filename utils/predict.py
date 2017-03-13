from sklearn import preprocessing
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
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

from utils.scoring import rmse_scoring, rmlse_scoring, rmsle


def cross_val(clf, X, y):
    # print("Cross Validation: \n{0}".format(clf))
    return cross_val_score(clf, X, y, cv=4, scoring=rmlse_scoring).mean()


def grid_search_cv(clf, tuned_params, X, y):
    sc1 = make_scorer(rmsle, greater_is_better=False)
    grd = GridSearchCV(clf, tuned_params, cv=2, scoring=sc1, verbose=1)

    grd.fit(X, y)
    print("Best={0}, score={1}".format(grd.best_params_, cross_val(grd.best_estimator_, X, y)))
    print(grd.best_estimator_)
    return grd.best_estimator_


def regression_fit(X, y):
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
    mdl["names"] = mdl.index.values

    # SVR
    # mdl["Models"].iloc[3] = grid_search_cv(SVR(),
    #                                        [{'kernel': ["rbf", "sigmoid"],
    #                                          # "degree": np.arange(1, 10),
    #                                          "C": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 100000,1000000]}],
    #                                        x_train, y_train)

    # KNeig
    mdl["Models"].iloc[4] = grid_search_cv(KNeighborsRegressor(),
                                           [{'n_neighbors': np.arange(2, 15), 'weights': ["uniform", "distance"]}],
                                           x_train, y_train)

    mdl["score_train"] = mdl["Models"].apply(
        lambda x: cross_val(x, x_train, y_train))

    mdl["Models"].apply(lambda x: x.fit(x_train, y_train))  # TODO cross validation fit
    mdl["score_test"] = mdl["Models"].apply(
        lambda x: rmlse_scoring(x, x_test, y_test))

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

    # TODO
    # 1 Вынести подбор метапараметров отдельно
    # 2 Результат 1 уровня формировать как новые фичи образованные предсказаниями классификатора
    # 3 Результат 2 уровня Как новая модель от данных 1 уровня (должна проходить все те же этапы)
    # 4 Попробовать найти метапараметры разных моделей, может что-то будет лучше

    return {"clf": clf,
            "predict_train": best_score["predict_train"].values[0],
            "predict_test": best_score["predict_test"].values[0],
            "score_train": train_score,
            "score_test": test_score,
            "y_train": y_train,
            "y_test": y_test}
