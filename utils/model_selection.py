import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from utils.prediction import ch_meta_rand_forest, ch_meta_heigbors, cross_val


def select_model(x_train, y_train):
    mdl = pd.DataFrame()
    mdl["Models"] = pd.Series([
        LinearRegression(),
        DecisionTreeRegressor(),
        ch_meta_rand_forest(x_train, y_train),
        SVR(),
        ch_meta_heigbors(x_train, y_train),
    ])
    mdl.index = ["LinReg", "DecisionTree", "RandForest", "SVR", "KNeig"]
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


