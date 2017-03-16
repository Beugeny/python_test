import stat
import numpy as np
from scipy import stats

import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns

import house_prices.feature_eng as fe
from utils.model_selection import select_model
import utils.plotter as plotter
from utils.plotter import plot_predict_accuracity, corr_plot, corr_zoom, normal_plot
from utils.prediction import predict_model
from utils.submit import submit
from scipy.stats import norm
import utils.normalizer as normalizer

# f = "C:\\data\\projects\\python\\data\\house_price\\"
f = "D:\\resources\\machineLearning\\python_test\\data\\house_price\\"
df_test = pd.read_csv(f + "test.csv")
df_train = pd.read_csv(f + "train.csv")

print("Total Rows train={0}".format(len(df_train.columns)))
print("Total Rows test={0}".format(len(df_test.columns)))
print("Train data len={0}".format(len(df_train)))

crl = df_train
corr_values = fe.numeric_correlation(crl, "SalePrice")
X = fe.eng(df_train, corr_values)
X['SalePrice'] = df_train['SalePrice']

print(len(X))
X=X[(X.GrLivArea < 4000) | (X.SalePrice > 300000)]
print(len(X))


# plotter.normal_plot(X['GrLivArea'])
# plotter.pair_plot(X[['GrLivArea', 'SalePrice', '1stFlrSF']])
# plotter.normal_plot(X['SalePrice'])
# plotter.normal_plot(X['1stFlrSF'])


# Normalize data
X['GrLivArea'] = np.log(X['GrLivArea'])
X['SalePrice'] = np.log(X['SalePrice'])
X['1stFlrSF'] = np.log(X['1stFlrSF'])

x_train, x_test, y_train, y_test = train_test_split(X.drop("SalePrice", axis=1),
                                                    X["SalePrice"],
                                                    test_size=0.33, random_state=42)
print(x_train.columns)

res = select_model(x_train, y_train)
plot_predict_accuracity(res["predict_train"], y_train, "Train")

predict_model(res["clf"], x_test, y_test)
plot_predict_accuracity(res["clf"].predict(x_test), y_test, "Test")

x_super_test = fe.eng(df_test, corr_values)
x_super_test['GrLivArea'] = np.log(x_super_test['GrLivArea'])
x_super_test['1stFlrSF'] = np.log(x_super_test['1stFlrSF'])
super_result = res["clf"].predict(x_super_test)

submit(df_test["Id"], np.exp(super_result), f, "Id", "SalePrice")

# See other data columns (Maby not numeric)
# two level predictions
# Maybe prediction with several models (maybe not)
