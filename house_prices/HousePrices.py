import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from utils.predict import regression_res
from utils.plotter import plot_predict_accuracity
from utils.predict import fit_predict

import house_prices.feature_eng as fe

f = "D:\\resources\\machineLearning\\python_test\\data\\house_prices\\"
df_test = pd.read_csv(f + "test.csv")
df_train = pd.read_csv(f + "train.csv")

print(df_train.columns.values)
print(len(df_train))

X_train_data, X_test_data, y_train, y_test = train_test_split(df_train.drop("SalePrice", axis=1), df_train["SalePrice"],
                                                              test_size=0.33,
                                                              random_state=42)

X_train = fe.eng(X_train_data)
X_test = fe.eng(X_test_data)
fit_predict(X_train,y_train)


# clf = LinearRegression()
# clf.fit(X_train, y_train)
#
# res = clf.predict(X_train)
# regression_res(res, y_train)
# plot_predict_accuracity(res, y_train, "Train")
#
# res = clf.predict(X_test)
# regression_res(res, y_test)
# plot_predict_accuracity(res, y_test, "Test")
